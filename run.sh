#!/usr/bin/env bash
# run.sh — start the tactile sensor on a Raspberry Pi 5 or Ubuntu with the right Python.
#   ./run.sh usb                                   # one Pi Zero camera
#   ./run.sh dual --usb=camera1 --usb2=camera2     # two cameras
#   ./run.sh usb ros --gel-mm=20                   # + ROS 2 topics (source ROS first)
# Any other script:  ./run.sh --script=preview_uvc.py
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY=python3
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"   # Ubuntu venv
SCRIPT=run_sensor_fast.py
ARGS=()
for a in "$@"; do
  case "$a" in
    --script=*) SCRIPT="${a#--script=}" ;;
    *) ARGS+=("$a") ;;
  esac
done
cd "$ROOT/host"
exec "$PY" "$SCRIPT" "${ARGS[@]}"
