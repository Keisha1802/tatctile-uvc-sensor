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
sudo apt-get install -y python3-venv python3-pip v4l-utils libgl1 libglib2.0-0

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

.venv/bin/python - <<'EOF'
import cv2, numpy
print(f"[OK] OpenCV {cv2.__version__}, NumPy {numpy.__version__}, "
      f"guidedFilter: {'contrib' if hasattr(getattr(cv2, 'ximgproc', None), 'guidedFilter') else 'numpy fallback'}")
EOF
echo
echo "Cameras:"; ls /dev/v4l/by-id/ 2>/dev/null | grep -i tactile || echo "  (no Tactile UVC Cam plugged in yet)"
echo
echo "Done. Log out/in once (video group), then:  ./run.sh usb"
