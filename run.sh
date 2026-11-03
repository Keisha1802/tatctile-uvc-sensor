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
# OpenCV's Qt GUI: use the system fonts and X11/XWayland (silences Qt font/Wayland warnings)
[[ -z "${QT_QPA_FONTDIR:-}" && -d /usr/share/fonts/truetype/dejavu ]] && \
  export QT_QPA_FONTDIR=/usr/share/fonts/truetype/dejavu
[[ "${XDG_SESSION_TYPE:-}" == "wayland" && -z "${QT_QPA_PLATFORM:-}" ]] && export QT_QPA_PLATFORM=xcb
# NumPy/BLAS must not start its own thread pool per call (cameras already run in parallel)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
# glibc: reuse freed memory instead of returning every large NumPy temporary to the kernel
# (each new one then costs page faults + zeroing; measured on a Pi 5 / Ubuntu, 4 KB pages)
export MALLOC_MMAP_THRESHOLD_=${MALLOC_MMAP_THRESHOLD_:-268435456} \
       MALLOC_TRIM_THRESHOLD_=${MALLOC_TRIM_THRESHOLD_:-536870912} \
       MALLOC_TOP_PAD_=${MALLOC_TOP_PAD_:-67108864}
cd "$ROOT/host"
exec "$PY" "$SCRIPT" "${ARGS[@]}"
