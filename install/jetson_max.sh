#!/usr/bin/env bash
# jetson_max.sh — keep an NVIDIA Jetson at full power for the tactile cameras, also after reboot.
#   1. power mode MAXN (Orin Nano Super: MAXN_SUPER, AGX Orin: MAXN), picked by name
#   2. jetson_clocks: CPU/GPU/EMC fixed at their maximum clock (systemd service)
#   3. USB autosuspend OFF for every USB device (ports/hubs never go to low power)
# It does NOT add USB bandwidth: cameras behind one hub still share 480 Mbit/s.
#
#   sudo bash install/jetson_max.sh            # apply (stays after reboot)
#   sudo bash install/jetson_max.sh --status   # show what is active
#   sudo bash install/jetson_max.sh --undo     # back to the default power behaviour
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
[[ -f /etc/nv_tegra_release ]] || { echo "not a Jetson"; exit 1; }

UNIT=/etc/systemd/system/tactile-max-power.service

status() {
  echo "power mode : $(nvpmodel -q 2>/dev/null | grep -i 'NV Power Mode' || echo ?)"
  echo "service    : $(systemctl is-enabled tactile-max-power 2>/dev/null || echo -)"
  echo "usb autosuspend (global): $(cat /sys/module/usbcore/parameters/autosuspend)"
  echo "usb devices not 'on'    : $(grep -L '^on$' /sys/bus/usb/devices/*/power/control 2>/dev/null | wc -l)"
  jetson_clocks --show 2>/dev/null | grep -E "^cpu0|^GPU|^EMC" || true
}

undo() {
  systemctl disable --now tactile-max-power 2>/dev/null || true
  rm -f "$UNIT"; systemctl daemon-reload
  jetson_clocks --restore 2>/dev/null || true
  echo 2 > /sys/module/usbcore/parameters/autosuspend
  echo "[undo] default power behaviour restored (power mode unchanged: sudo nvpmodel -m <id>)"
}

apply() {
  # power mode whose name starts with MAXN (MAXN_SUPER on the Orin Nano Super, MAXN on AGX)
  local id
  id=$(grep -E "^< *POWER_MODEL" /etc/nvpmodel.conf | grep -i "NAME=MAXN" \
       | sed -E 's/.*ID=([0-9]+).*/\1/' | head -1)
  if [[ -n "$id" ]]; then
    nvpmodel -m "$id" >/dev/null 2>&1 || true        # persists across reboot by itself
    echo "[1] power mode ID $id: $(nvpmodel -q 2>/dev/null | grep -i 'NV Power Mode')"
  else
    echo "[1] no MAXN mode found in /etc/nvpmodel.conf, power mode left as is"
  fi

  cat > "$UNIT" <<'EOF'
[Unit]
Description=Tactile sensor: Jetson clocks at maximum, USB autosuspend off
After=nvpmodel.service multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/bin/jetson_clocks
ExecStart=/bin/sh -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend; for f in /sys/bus/usb/devices/*/power/control; do echo on > "$f" 2>/dev/null; done'

[Install]
WantedBy=multi-user.target
EOF
  # the older single-purpose unit from the chat instructions is replaced by this one
  systemctl disable --now jetson-clocks 2>/dev/null || true
  rm -f /etc/systemd/system/jetson-clocks.service
  # cameras plugged in later: keep them powered too
  echo 'ACTION=="add", SUBSYSTEM=="usb", TEST=="power/control", ATTR{power/control}="on"' \
    > /etc/udev/rules.d/91-tactile-usb-power.rules
  udevadm control --reload-rules
  systemctl daemon-reload
  systemctl enable --now tactile-max-power >/dev/null
  echo "[2] jetson_clocks + [3] USB autosuspend off: active now and at every boot"
  status
}

case "${1:-}" in
  --status) status ;;
  --undo) undo ;;
  *) apply ;;
esac
