#!/usr/bin/env bash
# Start a gripper script with the project's venv (Linux).
#   ./run_gripper.sh                    main GUI (chromasense_gui.py)
#   ./run_gripper.sh scan_dxl.py        any other script in this folder
#   DXL_PORT=/dev/ttyUSB1 ./run_gripper.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=dxl-env/bin/python
[[ -x $PY ]] || { echo "run 'bash install_linux.sh' first"; exit 1; }
SCRIPT="${1:-chromasense_gui.py}"
shift || true

if [[ -z "${DXL_PORT:-}" ]]; then
    if [[ -e /dev/u2d2 ]]; then
        export DXL_PORT=/dev/u2d2
    else
        P=$(ls /dev/ttyUSB* 2>/dev/null | head -1 || true)
        [[ -n "$P" ]] && export DXL_PORT="$P"
    fi
fi
echo "U2D2 port: ${DXL_PORT:-none found (the GUI starts in simulation mode)}"
if [[ -n "${DXL_PORT:-}" && ! -w "${DXL_PORT}" ]]; then
    echo "!! no write access to $DXL_PORT: run install_linux.sh (dialout group), then log out and in"
fi
for f in /sys/bus/usb-serial/devices/*/latency_timer; do
    [[ -r $f ]] && [[ $(cat "$f") -gt 4 ]] && echo "note: USB latency $(cat "$f") ms on $(basename "$(dirname "$f")") (install_linux.sh sets 1 ms)"
done
# only ONE program may use the U2D2 port: show who holds it (another GUI, a script left running, ModemManager)
if [[ -n "${DXL_PORT:-}" ]]; then
    REAL=$(readlink -f "$DXL_PORT")
    HOLD=$(fuser "$REAL" 2>/dev/null | tr -s ' ')
    if [[ -n "$HOLD" ]]; then
        echo "!! $REAL is in use by:"
        for p in $HOLD; do echo "     PID $p: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-90)"; done
        echo "   close it first, e.g.:  kill $HOLD      (ModemManager: sudo systemctl stop ModemManager)"
        exit 1
    fi
fi
exec "$PY" "$SCRIPT" "$@"
