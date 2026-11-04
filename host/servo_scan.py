#!/usr/bin/env python3
"""
servo_scan.py  —  find and identify Waveshare/Feetech ST/SC bus servos (e.g. ST3215)
====================================================================================
Talks to the servos through a Waveshare "Servo Driver with ESP32" in SERIAL FORWARDING
mode (USB -> CP210x -> ESP32 -> servo bus), or through any USB bus-servo adapter.

Before using it with the Servo Driver with ESP32:
  join its Wi-Fi ESP32_DEV (password 12345678) with a phone/laptop, open 192.168.4.1,
  press "Start Serial Forwarding". (Without it the ESP32 does not pass the bytes on.)

  python3 host/servo_scan.py                      # boot text of the ESP + scan IDs 0..253
  python3 host/servo_scan.py watch                # live position/load of every servo found:
                                                  #   move or squeeze ONE finger by hand and
                                                  #   see which ID changes (no motion commanded)
  python3 host/servo_scan.py wiggle 3             # servo ID 3 moves +-WIGGLE steps and back
  python3 host/servo_scan.py wiggle all           # each servo in turn, 2 s apart
Options: --port=/dev/ttyUSB0 (default: first CP210x/CH34x/ttyUSB/ttyACM)  --baud=115200
         --max-id=253  --wiggle=60 (steps of 4096 per turn, ~5 deg)  --raw (print the ESP's text)

'watch' never moves anything. 'wiggle' moves a little: check that nothing can be pinched or
crushed (a closed gripper on a tactile sensor!) before you run it.
Protocol: Feetech SCS/STS (PING 0x01, READ 0x02, WRITE 0x03; present position 0x38,
speed 0x3A, load 0x3C, voltage 0x3E, temperature 0x3F, goal position 0x2A).
"""
import glob
import os
import sys
import time

try:
    import serial                                   # pyserial
except ImportError:
    sys.exit("[ERROR] pyserial missing:  .venv/bin/pip install pyserial   (or: pip install pyserial)")

PING, READ, WRITE = 0x01, 0x02, 0x03
ADDR_GOAL, ADDR_POS, ADDR_LOAD, ADDR_VOLT, ADDR_TEMP = 0x2A, 0x38, 0x3C, 0x3E, 0x3F


def busy_exit(port, err):
    """Port could not be opened: say who holds it instead of a traceback."""
    import subprocess
    who = ""
    try:
        dev = os.path.realpath(port)
        r = subprocess.run(["fuser", "-v", dev], capture_output=True, text=True, timeout=3)
        who = (r.stderr + r.stdout).strip()
    except Exception:
        pass
    sys.exit(f"[ERROR] cannot open {port}: {err}\n"
             + (f"        in use by:\n{who}\n" if who else "")
             + "        close the Arduino Serial Monitor / other servo script, or: "
               "sudo apt remove brltty (it grabs CP210x ports), then replug the ESP32")


def find_port():
    for pat in ("/dev/serial/by-id/*CP210*", "/dev/serial/by-id/*CH34*", "/dev/serial/by-id/*",
                "/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("[ERROR] no serial port found (ls /dev/ttyUSB* /dev/serial/by-id/)")


def packet(sid, instr, params=b""):
    body = bytes([sid, len(params) + 2, instr]) + bytes(params)
    return b"\xff\xff" + body + bytes([(~sum(body)) & 0xFF])


def open_port(port, baud, timeout):
    """Open WITHOUT pulsing DTR/RTS: on ESP32 boards (CP2102 + auto-reset transistors) the
    usual open resets the ESP32, which also switches serial forwarding off again."""
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, timeout
    s.dsrdtr = s.rtscts = False
    s.dtr = False
    s.rts = False
    try:
        s.open()
    except serial.SerialException as e:
        busy_exit(port, e)
    return s


class Bus:
    def __init__(self, port, baud, timeout=0.02, ser=None):
        self.s = ser or open_port(port, baud, timeout)
        self.s.timeout = timeout
        self.timeout = timeout

    def _reply(self, sid, n_params):
        """-> params bytes of a status packet from sid, or None."""
        need = 6 + n_params
        buf = b""
        t_end = time.monotonic() + self.timeout + 0.01 * (1 + n_params // 8)
        while time.monotonic() < t_end and len(buf) < 64:
            buf += self.s.read(max(1, need - len(buf)))
            i = buf.find(b"\xff\xff")
            while i >= 0 and len(buf) - i >= need:
                p = buf[i:i + need]
                if p[2] == sid and p[3] == n_params + 2:
                    if (~sum(p[2:-1])) & 0xFF == p[-1]:
                        return p[5:5 + n_params]
                i = buf.find(b"\xff\xff", i + 1)
        return None

    def ping(self, sid):
        self.s.reset_input_buffer()
        self.s.write(packet(sid, PING))
        return self._reply(sid, 0) is not None

    def read(self, sid, addr, n):
        self.s.reset_input_buffer()
        self.s.write(packet(sid, READ, [addr, n]))
        return self._reply(sid, n)

    def write(self, sid, addr, data):
        self.s.reset_input_buffer()
        self.s.write(packet(sid, WRITE, [addr] + list(data)))
        return self._reply(sid, 0) is not None


def s16(lo, hi, sign_bit=15):
    """Feetech values: magnitude + direction bit (bit 15 for position/speed, bit 10 for load)."""
    v = lo | (hi << 8)
    return -(v & ((1 << sign_bit) - 1)) if v & (1 << sign_bit) else v


def state(bus, sid):
    d = bus.read(sid, ADDR_POS, 8)            # pos(2) speed(2) load(2) volt(1) temp(1)
    if d is None or len(d) < 8:
        return None
    return dict(pos=s16(d[0], d[1]), speed=s16(d[2], d[3]), load=s16(d[4], d[5], 10),
                volt=d[6] / 10.0, temp=d[7])


def show_boot(s, secs=1.5):
    """Print whatever the ESP32 sends by itself (firmware banner / web-UI messages)."""
    s.timeout = 0.1
    t_end, out = time.monotonic() + secs, b""
    while time.monotonic() < t_end:
        out += s.read(256)
    txt = out.decode("utf-8", "replace").strip()
    print("[ESP] " + ("\n[ESP] ".join(txt.splitlines()[-15:]) if txt else "(no text: normal in forwarding mode)"))
    return "rst:" in txt or "boot:" in txt


def wait_for_forwarding(s):
    """Linux raises DTR/RTS for a moment when the port is opened, which resets an ESP32
    with auto-reset. Keep THIS port open, let the user switch forwarding on again, go on."""
    print("[WARN] the ESP32 just rebooted (opening the port resets it; Linux does that, not this "
          "script).\n       Keep this running: wait ~5 s until Wi-Fi ESP32_DEV is back, open "
          "192.168.4.1,\n       press 'Start Serial Forwarding', then press Enter here.")
    try:
        input("       Enter = forwarding is on > ")
    except EOFError:
        time.sleep(15)
    s.reset_input_buffer()


def scan(bus, max_id):
    found = []
    print(f"[SCAN] IDs 0..{max_id} ...", flush=True)
    for sid in range(0, max_id + 1):
        if bus.ping(sid):
            found.append(sid)
            st = state(bus, sid)
            extra = (f"position {st['pos']:5d}  load {st['load']:+5d}  {st['volt']:.1f} V  {st['temp']} C"
                     if st else "(state not readable)")
            print(f"  ID {sid:3d}: found   {extra}", flush=True)
    if not found:
        print("[SCAN] no servo answered. Check: 'Start Serial Forwarding' pressed in 192.168.4.1, "
              "servo power on (the board's DC input), --baud (115200 for the ESP32 forwarding, "
              "1000000 for a direct bus adapter).")
    else:
        print(f"[SCAN] {len(found)} servo(s): {found}")
    return found


def watch(bus, ids):
    print("[WATCH] move or squeeze ONE finger by hand; the ID whose numbers change is that finger.")
    print("        Ctrl+C to stop. (No motion is commanded.)")
    ref = {i: state(bus, i) for i in ids}
    try:
        while True:
            row = []
            for i in ids:
                st = state(bus, i)
                if st is None:
                    row.append(f"ID{i}: --")
                    continue
                r = ref.get(i) or st
                dpos = st["pos"] - r["pos"]
                mark = " <<" if abs(dpos) > 15 or abs(st["load"]) > 80 else ""
                row.append(f"ID{i}: pos {st['pos']:5d} ({dpos:+5d}) load {st['load']:+5d}{mark}")
            print("\r" + " | ".join(row) + "   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print()


def wiggle(bus, sid, steps):
    st = state(bus, sid)
    if st is None:
        print(f"  ID {sid}: no answer"); return
    p0 = st["pos"]
    print(f"  ID {sid}: wiggle around {p0} (+-{steps} steps) ...", flush=True)
    for target in (p0 + steps, p0 - steps, p0):
        t = max(0, min(4095, target))
        # goal position (2 B), goal time (2 B, 0), goal speed (2 B): moderate 600 steps/s
        bus.write(sid, ADDR_GOAL, [t & 0xFF, t >> 8, 0, 0, 600 & 0xFF, 600 >> 8])
        time.sleep(0.4)
    st = state(bus, sid)
    print(f"  ID {sid}: back at {st['pos'] if st else '?'}")


def main():
    args = sys.argv[1:]
    port, baud, max_id, steps = None, 115200, 253, 60
    for a in list(args):
        if a.startswith("--port="):
            port = a.split("=", 1)[1]; args.remove(a)
        elif a.startswith("--baud="):
            baud = int(a.split("=", 1)[1]); args.remove(a)
        elif a.startswith("--max-id="):
            max_id = int(a.split("=", 1)[1]); args.remove(a)
        elif a.startswith("--wiggle="):
            steps = int(a.split("=", 1)[1]); args.remove(a)
    raw = "--raw" in args
    args = [a for a in args if a != "--raw"]
    port = port or find_port()
    print(f"[INFO] port {port} ({os.path.realpath(port)}) @ {baud} baud")
    if not os.access(port, os.R_OK | os.W_OK):
        sys.exit(f"[ERROR] no permission for {port}:  sudo usermod -aG dialout $USER  (then log out/in)")
    ser = open_port(port, baud, 0.02)
    if show_boot(ser, 2.5) and "--no-wait" not in sys.argv:
        wait_for_forwarding(ser)
    bus = Bus(port, baud, ser=ser)
    mode = args[0] if args else "scan"
    if mode == "scan":
        scan(bus, max_id)
    elif mode == "watch":
        ids = [int(x) for x in args[1:]] or scan(bus, max_id)
        if ids:
            watch(bus, ids)
    elif mode == "wiggle":
        if len(args) < 2:
            sys.exit("usage: servo_scan.py wiggle <id|all>")
        ids = scan(bus, max_id) if args[1] == "all" else [int(args[1])]
        for i in ids:
            wiggle(bus, i, steps)
            if len(ids) > 1:
                time.sleep(2.0)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
