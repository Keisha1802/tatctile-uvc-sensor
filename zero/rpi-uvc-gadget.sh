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
# USB bandwidth reserved per 125 us microframe (640x480 YUYV @ 15 fps needs ~9.2 MB/s).
# 2048 (2 x 1024 B) = 16.4 MB/s, the maximum for the Zero's dwc2 FIFO (3072 -> 'No suitable
# fifo found'). 1536 (2 x 768 B) also works, but on the Jetson Orin Nano devkit it did NOT
# stop the frame drops of several cameras behind its internal USB hub (even two cameras
# drop there): give each camera its own USB port/root port instead.
echo ${UVC_MAXPACKET:-2048} > $F/streaming_maxpacket

ln -s $F $G/configs/c.1/uvc.0
ls /sys/class/udc | head -n1 > $G/UDC

echo "gadget active -> running $UVC_BIN"
exec "$UVC_BIN" -c 0 uvc.0
