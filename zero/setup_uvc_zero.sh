#!/usr/bin/env bash
# setup_uvc_zero.sh — turn a Pi Zero (W / 2 W) + CSI camera into a USB webcam (UVC)
# Run ONCE on the Pi Zero:   sudo bash setup_uvc_zero.sh
# Works for any number of Zeros: each one shows up on the Pi 5 under its own hostname
#   /dev/v4l/by-id/usb-<hostname>_Tactile_UVC_Cam_<serial>-video-index0
# so give every Zero its own hostname in Raspberry Pi Imager (camera, camera2, ...).
# When done: reboot, then plug the "USB" (data) microUSB port into the Pi 5 / PC.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then echo "Run with sudo"; exit 1; fi
USER_HOME=$(getent passwd "${SUDO_USER:-root}" | cut -d: -f6)
CFG=/boot/firmware/config.txt
[[ -f $CFG ]] || CFG=/boot/config.txt

echo "== 1. Enable USB gadget (dwc2) in $CFG"
# Note: the stock config.txt already has "dtoverlay=dwc2,dr_mode=host" under [cm5]
# (applies to the CM5 only). Do not be fooled by that line: check for dr_mode=otg/peripheral.
if ! grep -qE "^dtoverlay=dwc2,dr_mode=(otg|peripheral)" "$CFG"; then
    printf '\n[all]\ndtoverlay=dwc2,dr_mode=otg\n' >> "$CFG"
fi
grep -q "^camera_auto_detect=1" "$CFG" || echo "[WARN] camera_auto_detect=1 missing in $CFG — the camera may not be detected"

echo "== 2. Install build packages"
apt-get update
# rpicam-apps too: installing libcamera-dev can pull a newer libcamera, and an old
# rpicam-apps then fails with "symbol lookup error ... generateConfiguration" (test tool only)
apt-get install -y git meson ninja-build build-essential pkg-config \
                   libcamera-dev libjpeg-dev v4l-utils rpicam-apps

echo "== 3. Build & install uvc-gadget"
SRC="$USER_HOME/uvc-gadget"
if [[ ! -d $SRC ]]; then
    sudo -u "${SUDO_USER:-root}" git clone https://gitlab.freedesktop.org/camera/uvc-gadget.git "$SRC"
fi
cd "$SRC"
# New libcamera (Trixie) uses std::span -> needs C++20, while uvc-gadget is
# still C++17 + libcamera::Span. Small patch so it builds:
#   - Span<...> -> std::span<...>
#   - FrameDurationLimits filled via an initializer_list (a C++20 std::span
#     cannot be built from { a, b })
LCS=lib/libcamera-source.cpp
if [[ -f $LCS ]] && grep -q "Span<" "$LCS"; then
    perl -0pi -e 's/Span<const int64_t,\s*2>\(\{\s*frame_time,\s*frame_time\s*\}\)/{ static_cast<int64_t>(frame_time), static_cast<int64_t>(frame_time) }/g' "$LCS"
    perl -pi -e 's/(?<![:\w])Span</std::span</g' "$LCS"
    grep -q "#include <span>" "$LCS" || sed -i '0,/^#include/s//#include <span>\n#include/' "$LCS"
    echo "[patch] $LCS adapted for new libcamera (std::span)"
fi
chown -R "${SUDO_USER:-root}": "$SRC"
rm -rf build
sudo -u "${SUDO_USER:-root}" meson setup build --prefix=/usr -Dcpp_std=c++20 -Dwerror=false
sudo -u "${SUDO_USER:-root}" ninja -C build
ninja -C build install
ldconfig
UVC_BIN=$(command -v uvc-gadget || true)
[[ -n $UVC_BIN ]] || { echo "[ERROR] uvc-gadget is not installed"; exit 1; }
echo "[OK] uvc-gadget installed at $UVC_BIN"

# Camera: IMX219 clones (e.g. 200 deg lenses) are often NOT found by camera_auto_detect.
# If libcamera sees no camera, load the imx219 driver directly (takes effect after reboot).
if ! grep -q "^dtoverlay=imx219" "$CFG"; then
    if ! command -v cam >/dev/null 2>&1 || ! cam -l 2>/dev/null | grep -qi imx219; then
        if ! dmesg | grep -qi "imx219"; then
            sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' "$CFG"
            printf '\n[all]\ndtoverlay=imx219\n' >> "$CFG"
            echo "[camera] no camera auto-detected -> dtoverlay=imx219 forced in $CFG (after reboot)"
            echo "         (other sensor? replace imx219 by its overlay, e.g. ov5647)"
        fi
    fi
fi

echo "== 4. Write gadget script /usr/local/bin/rpi-uvc-gadget.sh"
cat > /usr/local/bin/rpi-uvc-gadget.sh <<'EOS'
#!/usr/bin/env bash
# rpi-uvc-gadget.sh — register the Pi Zero as a UVC camera via configfs, then run uvc-gadget.
# Install on the Zero:  sudo install -m 755 rpi-uvc-gadget.sh /usr/local/bin/rpi-uvc-gadget.sh
#
# IMPORTANT: configfs resolves RELATIVE symlink targets against the process's working
# directory (not the link location), so every link here uses an ABSOLUTE PATH.
set -e
modprobe libcomposite

CFS=/sys/kernel/config/usb_gadget
G=$CFS/g1
UVC_BIN=$(command -v uvc-gadget)

if [[ -d $G ]]; then
    if [[ -n "$(cat $G/UDC 2>/dev/null)" ]]; then
        echo "gadget already active -> starting uvc-gadget"
        exec "$UVC_BIN" -c 0 uvc.0
    fi
    echo "old (incomplete) gadget found -> reboot first to clear it: sudo reboot"
    exit 1
fi

mkdir -p $G
echo 0x0525 > $G/idVendor
echo 0xa4a2 > $G/idProduct
echo 0x0100 > $G/bcdDevice
echo 0x0200 > $G/bcdUSB
echo 0xEF   > $G/bDeviceClass
echo 0x02   > $G/bDeviceSubClass
echo 0x01   > $G/bDeviceProtocol
mkdir -p $G/strings/0x409
# serial = last 8 digits of the Pi's CPU serial -> unique per Zero (two Zeros on one Pi 5)
SER=$(awk '/^Serial/{print substr($3, length($3)-7)}' /proc/cpuinfo)
echo "${SER:-$(hostname)}" > $G/strings/0x409/serialnumber
echo "$(hostname)"     > $G/strings/0x409/manufacturer
echo "Tactile UVC Cam" > $G/strings/0x409/product
mkdir -p $G/configs/c.1/strings/0x409
echo "UVC" > $G/configs/c.1/strings/0x409/configuration
echo 500   > $G/configs/c.1/MaxPower

F=$G/functions/uvc.0
mkdir -p $F

# frame <format-dir> <w> <h> <interval100ns...>   (interval = 1e7 / fps)
frame() {
    local fmt=$1 w=$2 h=$3; shift 3
    local d=$F/streaming/$fmt/${h}p
    mkdir -p $d
    echo $w > $d/wWidth
    echo $h > $d/wHeight
    echo $(( w * h * 2 )) > $d/dwMaxVideoFrameBufferSize
    echo $(( w * h * 16 * 5 ))   > $d/dwMinBitRate
    echo $(( w * h * 16 * 120 )) > $d/dwMaxBitRate
    echo $1 > $d/dwDefaultFrameInterval
    # MUST be a single write(): configfs replaces the content on every write, so
    # printf (which writes once per argument) would keep only the last interval.
    cat > $d/dwFrameInterval <<EOI
$(printf '%s\n' "$@")
EOI
}

# Uncompressed YUYV (best for speckle): 640x480 @ 15 (default)/10,
# 384x288 @ 30/15, 320x256 @ 60/30/15. Heights must be multiples of 32
# (320x240 gives EINVAL in uvc-gadget).
frame uncompressed/u 640 480 666666 1000000
frame uncompressed/u 384 288 333333 666666
frame uncompressed/u 320 256 166666 333333 666666
# MJPEG (faster over USB, but compression artifacts)
frame mjpeg/m 640  480 333333 666666

mkdir -p $F/streaming/header/h
ln -s $F/streaming/uncompressed/u $F/streaming/header/h/u
ln -s $F/streaming/mjpeg/m        $F/streaming/header/h/m
ln -s $F/streaming/header/h $F/streaming/class/fs/h
ln -s $F/streaming/header/h $F/streaming/class/hs/h
ln -s $F/streaming/header/h $F/streaming/class/ss/h

mkdir -p $F/control/header/h
ln -s $F/control/header/h $F/control/class/fs/h
ln -s $F/control/header/h $F/control/class/ss/h

# 2048 = the maximum that fits the Pi Zero dwc2 FIFO (3072 -> 'No suitable fifo found').
echo 2048 > $F/streaming_maxpacket

ln -s $F $G/configs/c.1/uvc.0
ls /sys/class/udc | head -n1 > $G/UDC

echo "gadget active -> running $UVC_BIN"
exec "$UVC_BIN" -c 0 uvc.0
EOS
chmod +x /usr/local/bin/rpi-uvc-gadget.sh

echo "== 5. systemd service (starts automatically at boot)"
cat > /etc/systemd/system/uvc-gadget.service <<'EOS'
[Unit]
Description=Tactile UVC webcam gadget
After=local-fs.target

[Service]
ExecStart=/usr/local/bin/rpi-uvc-gadget.sh
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOS
systemctl daemon-reload
systemctl enable uvc-gadget.service

echo
echo "== Check that the camera is detected:"
rpicam-hello --list-cameras 2>/dev/null || libcamera-hello --list-cameras 2>/dev/null || true
echo
echo "DONE. Now:  sudo reboot"
echo "Then move the cable to the 'USB' microUSB port (middle) and plug it into the Pi 5 / PC."
echo "Logs if something goes wrong:  journalctl -u uvc-gadget -b"
echo "On the Pi 5 this camera is:  /dev/v4l/by-id/usb-$(hostname)_Tactile_UVC_Cam_*-video-index0"
echo "Use it with:  python3 run_sensor_fast.py --usb=$(hostname)"
