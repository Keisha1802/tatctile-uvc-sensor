#!/usr/bin/env bash
# tune_zero.sh — steady 15 fps from a Pi Zero W UVC camera WITHOUT changing resolution/fps.
#
# A Zero W (one ARMv6 core) is at its limit at 640x480 YUYV 15 fps. Frames arrive late or
# incomplete when the CPU is clocked down or busy with other things. This script:
#   1. keeps the CPU at its normal 1 GHz all the time (force_turbo=1, NO over-voltage,
#      no overclock) instead of ramping up from 700 MHz when the load arrives
#   2. gives uvc-gadget priority over everything else (nice -15; not realtime, so SSH and
#      WiFi keep working)
#   3. switches off things the camera does not need: Bluetooth, and background timers
#      (apt updates, man-db, e2scrub) that wake up at random and steal CPU
# WiFi/SSH stay on. Everything is undone with --undo.
#
#   sudo bash tune_zero.sh          # apply, then reboot
#   sudo bash tune_zero.sh --undo   # revert, then reboot
#   sudo bash tune_zero.sh --status # show what is applied
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }

CFG=/boot/firmware/config.txt; [[ -f $CFG ]] || CFG=/boot/config.txt
DROPIN_DIR=/etc/systemd/system/uvc-gadget.service.d
DROPIN=$DROPIN_DIR/tactile-tune.conf
STATE=/var/lib/tactile-tune
BEGIN="# >>> tactile-tune"
END="# <<< tactile-tune"
UNITS=(bluetooth.service hciuart.service ModemManager.service triggerhappy.service
       apt-daily.timer apt-daily-upgrade.timer man-db.timer e2scrub_all.timer)

status() {
  echo "config.txt block : $(grep -q "^$BEGIN" "$CFG" && echo applied || echo -)"
  echo "uvc-gadget nice  : $([[ -f $DROPIN ]] && echo applied || echo -)"
  echo "governor         : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo ?)"
  echo "cpu freq (kHz)   : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq 2>/dev/null || echo ?)"
  echo "throttled        : $(vcgencmd get_throttled 2>/dev/null || echo ?)"
  echo "disabled by tune : $(cat $STATE/disabled 2>/dev/null | tr '\n' ' ' || true)"
}

undo() {
  sed -i "/^$BEGIN/,/^$END/d" "$CFG"
  rm -f "$DROPIN"; rmdir "$DROPIN_DIR" 2>/dev/null || true
  if [[ -f $STATE/disabled ]]; then
    while read -r u; do [[ -n $u ]] && systemctl enable "$u" 2>/dev/null || true; done < $STATE/disabled
    rm -f $STATE/disabled
  fi
  systemctl daemon-reload
  echo "[undo] reverted. Reboot: sudo reboot"
}

apply() {
  # 1. CPU always at the normal 1 GHz (no overclock, no over_voltage)
  if ! grep -q "^$BEGIN" "$CFG"; then
    printf '\n%s\n[all]\nforce_turbo=1\ndtoverlay=disable-bt\n%s\n' "$BEGIN" "$END" >> "$CFG"
  fi
  echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true

  # 2. uvc-gadget first in line for the CPU (normal scheduler, so SSH/WiFi still run)
  mkdir -p "$DROPIN_DIR"
  cat > "$DROPIN" <<'EOF'
[Service]
Nice=-15
IOSchedulingClass=best-effort
IOSchedulingPriority=0
EOF

  # 3. background services the camera does not need
  mkdir -p $STATE; : > $STATE/disabled.new
  for u in "${UNITS[@]}"; do
    if systemctl is-enabled "$u" >/dev/null 2>&1; then
      systemctl disable --now "$u" >/dev/null 2>&1 && echo "$u" >> $STATE/disabled.new
    fi
  done
  cat $STATE/disabled.new >> $STATE/disabled 2>/dev/null || mv $STATE/disabled.new $STATE/disabled
  rm -f $STATE/disabled.new
  systemctl daemon-reload
  echo "[tune] applied (CPU fixed at 1 GHz without overclock, uvc-gadget nice -15,"
  echo "       Bluetooth + background timers off). Reboot: sudo reboot"
  echo "       Revert any time: sudo bash tune_zero.sh --undo"
}

case "${1:-}" in
  --undo) undo ;;
  --status) status ;;
  *) apply ;;
esac
