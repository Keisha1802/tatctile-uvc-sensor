#!/usr/bin/env bash
# setup_raspi.sh — host setup on a Raspberry Pi 5 (Raspberry Pi OS Bookworm / Trixie, 64-bit)
#
#   bash install/setup_raspi.sh            # packages + video group
#   bash install/setup_raspi.sh --usb-power  # also allow 1.6 A on the USB ports (needed when
#                                            # Pi Zero cameras are powered from the Pi 5)
# Python packages come from apt (system OpenCV/NumPy/Picamera2), no virtualenv needed.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] apt packages"
sudo apt-get update
sudo apt-get install -y python3-opencv python3-numpy v4l-utils
# Picamera2 is only needed for a CSI camera on the Pi 5 itself (optional with USB cameras)
sudo apt-get install -y python3-picamera2 || echo "[WARN] python3-picamera2 not installed (only needed for CSI)"

echo "[2/3] camera permission"
sudo usermod -aG video "$USER"

# uvcvideo: drop INCOMPLETE frames in the driver (nodrop=0). With nodrop=1 a frame that
# lost USB packets is still delivered, and its missing part shows up green / glitchy.
echo "options uvcvideo nodrop=0" | sudo tee /etc/modprobe.d/uvcvideo-tactile.conf >/dev/null
if [[ -e /sys/module/uvcvideo/parameters/nodrop ]]; then
  echo 0 | sudo tee /sys/module/uvcvideo/parameters/nodrop >/dev/null || true
fi
echo "      uvcvideo nodrop=0 (incomplete frames dropped, no green frames)"


if [[ "${1:-}" == "--usb-power" ]]; then
  echo "[3/3] usb_max_current_enable=1"
  CFG=/boot/firmware/config.txt; [[ -f $CFG ]] || CFG=/boot/config.txt
  if ! grep -q '^usb_max_current_enable=1' "$CFG"; then
    echo 'usb_max_current_enable=1' | sudo tee -a "$CFG" >/dev/null
    echo "      added to $CFG (reboot needed)"
  else
    echo "      already set in $CFG"
  fi
else
  echo "[3/3] skipped USB power setting (add --usb-power if the Zero cameras reset/freeze)"
fi

python3 - <<'EOF'
import cv2, numpy
print(f"[OK] OpenCV {cv2.__version__}, NumPy {numpy.__version__}, "
      f"guidedFilter: {'contrib' if hasattr(getattr(cv2, 'ximgproc', None), 'guidedFilter') else 'numpy fallback'}")
EOF
echo
echo "Cameras:"; ls /dev/v4l/by-id/ 2>/dev/null | grep -i tactile || echo "  (no Tactile UVC Cam plugged in yet)"
echo
echo "ROS 2: Raspberry Pi OS has no ROS 2 apt packages. Use Ubuntu 24.04 on the Pi 5"
echo "(then: bash install/setup_ubuntu.sh --ros) or a ROS 2 Docker container."
echo
echo "Done. Log out/in once (video group), then:  ./run.sh usb"
