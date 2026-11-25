#!/usr/bin/env bash
# Linux setup for the gripper's Dynamixel servos (U2D2).
# Works on Ubuntu / Debian, x86_64 PC and aarch64 (Jetson, Raspberry Pi).
#
#   cd dynamixel_u2d2
#   bash install_linux.sh            # venv + packages + U2D2 udev rule + serial permission
#   bash install_linux.sh --no-udev  # skip the udev rule / group (no sudo)
#
# Afterwards:  ./run_gripper.sh       (main GUI)   ./run_gripper.sh scan_dxl.py
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
UDEV=1
[[ "${1:-}" == "--no-udev" ]] && UDEV=0

echo "== Dynamixel U2D2 gripper - Linux setup ($(uname -m))"

# 1. system packages (python venv, Qt runtime for the GUI)
if command -v apt-get >/dev/null 2>&1; then
    PKGS="python3 python3-venv python3-pip python3-pyqt5 libxcb-xinerama0"
    MISSING=""
    for p in $PKGS; do dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"; done
    if [[ -n "$MISSING" ]]; then
        echo "-- installing:$MISSING"
        sudo apt-get update -q && sudo apt-get install -y -q $MISSING
    fi
else
    echo "-- not a Debian / Ubuntu system: make sure python3, python3-venv and PyQt5 are installed"
fi

# 2. virtual environment (with the system PyQt5: no 100 MB wheel / no compile on ARM)
if [[ ! -x dxl-env/bin/python ]]; then
    echo "-- creating dxl-env"
    python3 -m venv --system-site-packages dxl-env
fi
dxl-env/bin/python -m pip install -q --upgrade pip
dxl-env/bin/python -m pip install -q -r requirements.txt
dxl-env/bin/python -c "from PyQt5 import QtWidgets" 2>/dev/null || dxl-env/bin/python -m pip install -q --ignore-installed PyQt5   # system Qt broken / missing -> wheel
dxl-env/bin/python -m pip install -q -e . >/dev/null 2>&1 || true     # console tools: dynamixel-scan, ...
dxl-env/bin/python -c "import dynamixel_sdk; from PyQt5 import QtWidgets; print('-- python packages OK (PyQt5 + dynamixel-sdk)')"

# 3. U2D2: fixed name /dev/u2d2, low USB latency, access without sudo
if [[ $UDEV == 1 ]]; then
    RULE=/etc/udev/rules.d/99-u2d2.rules
    echo "-- udev rule $RULE (U2D2 = FTDI 0403:6014 -> /dev/u2d2, latency 1 ms)"
    sudo tee "$RULE" >/dev/null <<'RULES'
# ROBOTIS U2D2 (FTDI FT232H): stable name + 1 ms latency timer (default 16 ms slows every read),
# ModemManager told to leave it alone (its probing breaks packets: "multiple access on port")
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6014", SYMLINK+="u2d2", MODE="0666", ENV{ID_MM_DEVICE_IGNORE}="1"
ACTION=="add", SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
RULES
    sudo udevadm control --reload-rules && sudo udevadm trigger
    if ! id -nG "$USER" | grep -qw dialout; then
        sudo usermod -aG dialout "$USER"
        echo "-- added $USER to 'dialout' (log out and in once)"
    fi
    # brltty grabs FTDI adapters on some Ubuntu versions
    if systemctl list-unit-files 2>/dev/null | grep -q '^brltty'; then
        echo "-- note: if /dev/ttyUSB* disappears right after plugging in, run: sudo apt remove brltty"
    fi
fi

chmod +x run_gripper.sh
echo
echo "Done. Plug the U2D2 in, then:"
echo "   ./run_gripper.sh                 main gripper GUI (chromasense_gui.py)"
echo "   ./run_gripper.sh scan_dxl.py     find the servos"
echo "   DXL_PORT=/dev/ttyUSB1 ./run_gripper.sh   another port"
ls -l /dev/u2d2 2>/dev/null || echo "   (/dev/u2d2 appears when the U2D2 is plugged in)"
