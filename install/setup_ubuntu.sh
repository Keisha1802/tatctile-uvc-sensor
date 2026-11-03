#!/usr/bin/env bash
# setup_ubuntu.sh — native host setup on Ubuntu 22.04 / 24.04 (x86-64 PC/laptop or arm64,
# incl. Ubuntu on a Raspberry Pi 5). The Pi Zero cameras are plain UVC webcams, so no
# driver is needed: the kernel's uvcvideo handles them.
#
#   bash install/setup_ubuntu.sh          # Python venv (.venv) + OpenCV contrib + video group
#   bash install/setup_ubuntu.sh --ros    # also install ROS 2 (Humble on 22.04, Jazzy on 24.04)
set -euo pipefail
cd "$(dirname "$0")/.."
WANT_ROS=0; [[ "${1:-}" == "--ros" ]] && WANT_ROS=1

. /etc/os-release
echo "Ubuntu ${VERSION_ID} (${VERSION_CODENAME})"

echo "[1/4] apt packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip v4l-utils libgl1 libglib2.0-0 fonts-dejavu-core

# Raspberry Pi 5 running Ubuntu: let the USB ports supply 1.6 A (Pi Zero cameras reset /
# drop packets otherwise) and keep USB autosuspend off for the cameras
if grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
  CFG=/boot/firmware/config.txt
  if [[ -f $CFG ]] && ! grep -q '^usb_max_current_enable=1' "$CFG"; then
    echo 'usb_max_current_enable=1' | sudo tee -a "$CFG" >/dev/null
    echo "      Pi 5: usb_max_current_enable=1 added to $CFG (reboot needed)"
  fi
fi
echo 'ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="0525", ATTR{idProduct}=="a4a2", TEST=="power/control", ATTR{power/control}="on"' \
  | sudo tee /etc/udev/rules.d/90-tactile-uvc.rules >/dev/null
sudo udevadm control --reload-rules

echo "[2/4] camera permission"
sudo usermod -aG video "$USER"

if [[ $WANT_ROS == 1 ]]; then
  case "$VERSION_CODENAME" in
    jammy) DISTRO=humble ;;
    noble) DISTRO=jazzy ;;
    *) echo "[WARN] no ROS 2 LTS for $VERSION_CODENAME; skipping ROS"; DISTRO="" ;;
  esac
  if [[ -n "$DISTRO" ]]; then
    echo "[3/4] ROS 2 $DISTRO (official apt repository)"
    sudo apt-get install -y software-properties-common curl
    sudo add-apt-repository -y universe
    VER=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
          | grep -F '"tag_name"' | awk -F'"' '{print $4}')
    curl -L -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${VER}/ros2-apt-source_${VER}.${VERSION_CODENAME}_all.deb"
    sudo dpkg -i /tmp/ros2-apt-source.deb
    sudo apt-get update
    sudo apt-get install -y "ros-${DISTRO}-ros-base"
    echo "      -> source /opt/ros/${DISTRO}/setup.bash   before running with 'ros'"
  fi
else
  echo "[3/4] ROS 2 skipped (add --ros to install it)"
fi

echo "[4/4] Python environment (.venv)"
# --system-site-packages: rclpy from /opt/ros stays importable inside the venv
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
# pip's OpenCV ships Qt without fonts -> trackbar names in the Controls window are blank.
# Point its font folder at the system DejaVu fonts.
CV2_DIR=$(.venv/bin/python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')
if [[ -d "$CV2_DIR/qt" && -d /usr/share/fonts/truetype/dejavu ]]; then
  ln -sfn /usr/share/fonts/truetype/dejavu "$CV2_DIR/qt/fonts"
  echo "      OpenCV Qt fonts -> /usr/share/fonts/truetype/dejavu"
fi

.venv/bin/python - <<'EOF'
import cv2, numpy
print(f"[OK] OpenCV {cv2.__version__}, NumPy {numpy.__version__}, "
      f"guidedFilter: {'contrib' if hasattr(getattr(cv2, 'ximgproc', None), 'guidedFilter') else 'numpy fallback'}")
EOF
echo
echo "Cameras:"; ls /dev/v4l/by-id/ 2>/dev/null | grep -i tactile || echo "  (no Tactile UVC Cam plugged in yet)"
echo
echo "Done. Log out/in once (video group), then:  ./run.sh usb"
