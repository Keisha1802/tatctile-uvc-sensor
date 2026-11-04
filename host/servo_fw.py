#!/usr/bin/env python3
"""
servo_fw.py  —  identify the 6 bus servos through the lab's own ESP32 Arduino firmware
=======================================================================================
For the sketch on the Waveshare Servo Driver with ESP32 that talks to the PC at 460800 baud:
  ESP32 -> PC : every ~10 ms one line  pos1,load1,vel1, ... ,pos6,load6,vel6
                (pos rad relative to zero_offset, load N*m, vel rad/s; 'ERR,ERR' = no answer)
  PC -> ESP32 : one line  q1,q2,q3,q4,q5,q6   (target angles in rad) -> ALL six servos move
Servo i in the line = servo_ids[i] in the sketch = ID 1..6.

  python3 host/servo_fw.py                 # scan: which IDs answer, angle/load/temperature-free view
  python3 host/servo_fw.py watch           # live: move ONE joint by hand -> '<<' marks its ID
  python3 host/servo_fw.py wiggle 3        # ID 3 moves +-5 deg and back, the others HOLD where they are
Options: --port=/dev/ttyUSB0  --baud=460800  --deg=5 (wiggle size)

'scan' and 'watch' send NOTHING (nothing moves). 'wiggle' sends full 6-servo commands with
the CURRENT angle of every other servo, so only one moves; it refuses when any servo reads
ERR (a missing value would send that servo to 0 rad). Servos need the board's DC input.
Opening the port resets the ESP32 (Linux pulses DTR/RTS): the script waits for its banner.
"""
import glob
import math
import os
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("[ERROR] pyserial missing:  .venv/bin/pip install pyserial")

N = 6


def find_port():
    for pat in ("/dev/serial/by-id/*CP210*", "/dev/serial/by-id/*CH34*", "/dev/ttyUSB*", "/dev/ttyACM*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    sys.exit("[ERROR] no serial port found")


def parse(line):
    """'p,l,v,...' with 'ERR,ERR' for a failed servo -> list of 6 dicts or None per servo."""
    tok = [t.strip() for t in line.strip().split(",")]
    out, i = [], 0
    while len(out) < N and i < len(tok):
        if tok[i] == "ERR":
            out.append(None); i += 2
        else:
            try:
                out.append(dict(pos=float(tok[i]), load=float(tok[i + 1]), vel=float(tok[i + 2])))
            except (ValueError, IndexError):
                return None
            i += 3
    return out if len(out) == N else None


class Link:
    def __init__(self, port, baud):
        self.s = serial.Serial(port, baud, timeout=0.2)
        self.buf = b""

    def lines(self, secs):
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            self.buf += self.s.read(4096)
            *full, self.buf = self.buf.split(b"\n")
            for ln in full:
                yield ln.decode("utf-8", "replace").strip()

    def latest(self, secs=0.5):
        last = None
        for ln in self.lines(secs):
            f = parse(ln)
            if f is not None:
                last = f
        return last

    def send(self, q):
        self.s.write((",".join(f"{v:.4f}" for v in q) + "\n").encode())


def wait_ready(link):
    print("[INFO] waiting for the ESP32 (opening the port resets it) ...", flush=True)
    for ln in link.lines(6.0):
        if "initialized" in ln.lower():
            print(f"[ESP] {ln}")
        elif parse(ln) is not None:
            return True
    return False


def fmt(f):
    return f"{math.degrees(f['pos']):+7.1f} deg  load {f['load']:+5.2f} Nm  vel {f['vel']:+5.2f}"


def scan(link):
    samples = [f for f in (parse(ln) for ln in link.lines(1.0)) if f is not None]
    if not samples:
        print("[SCAN] no feedback lines. Right firmware/baud (--baud=460800)? DC power on?")
        return None
    last = samples[-1]
    ok = [i for i in range(N) if any(s[i] is not None for s in samples)]
    print(f"[SCAN] {len(samples)} feedback lines in 1 s")
    for i in range(N):
        good = sum(s[i] is not None for s in samples)
        if last[i] is not None:
            print(f"  ID {i + 1}: answers ({good}/{len(samples)})   {fmt(last[i])}")
        else:
            print(f"  ID {i + 1}: NO ANSWER ({good}/{len(samples)})  -> not connected, other ID, or no power")
    print(f"[SCAN] answering: {[i + 1 for i in ok]}")
    return last


def watch(link):
    ref = link.latest()
    if ref is None:
        print("[WATCH] no feedback"); return
    print("[WATCH] move ONE joint by hand; '<<' marks it. Ctrl+C to stop. (Nothing is sent.)")
    try:
        for ln in link.lines(1e9):
            f = parse(ln)
            if f is None:
                continue
            cells = []
            for i in range(N):
                if f[i] is None or ref[i] is None:
                    cells.append(f"ID{i + 1}:  ERR ")
                    continue
                d = math.degrees(f[i]["pos"] - ref[i]["pos"])
                mark = "<<" if abs(d) > 2.0 or abs(f[i]["load"] - ref[i]["load"]) > 0.15 else "  "
                cells.append(f"ID{i + 1}:{d:+6.1f}{mark}")
            print("\r" + " ".join(cells) + "  (deg from start)", end="", flush=True)
    except KeyboardInterrupt:
        print()


def wiggle(link, sid, deg):
    f = link.latest()
    if f is None:
        sys.exit("[WIGGLE] no feedback")
    if any(x is None for x in f):
        bad = [i + 1 for i in range(N) if f[i] is None]
        sys.exit(f"[WIGGLE] refused: servo(s) {bad} read ERR; a command would send them to 0 rad")
    q0 = [x["pos"] for x in f]
    k = sid - 1
    print(f"[WIGGLE] ID {sid} around {math.degrees(q0[k]):+.1f} deg (+-{deg} deg); the others hold")
    for off in (+deg, -deg, 0.0):
        q = list(q0)
        q[k] = q0[k] + math.radians(off)
        link.send(q)
        time.sleep(0.5)
    g = link.latest()
    if g and g[k]:
        print(f"[WIGGLE] ID {sid} back at {math.degrees(g[k]['pos']):+.1f} deg")


def main():
    args = sys.argv[1:]
    port, baud, deg = None, 460800, 5.0
    for a in list(args):
        if a.startswith("--port="):
            port = a.split("=", 1)[1]; args.remove(a)
        elif a.startswith("--baud="):
            baud = int(a.split("=", 1)[1]); args.remove(a)
        elif a.startswith("--deg="):
            deg = float(a.split("=", 1)[1]); args.remove(a)
    port = port or find_port()
    if not os.access(port, os.R_OK | os.W_OK):
        sys.exit(f"[ERROR] no permission for {port}:  sg dialout -c \"...\"  or log out/in once")
    print(f"[INFO] {port} @ {baud}")
    link = Link(port, baud)
    if not wait_ready(link):
        sys.exit("[ERROR] no feedback lines within 6 s: check the baud (sketch: 460800) and the DC power")
    mode = args[0] if args else "scan"
    if mode == "scan":
        scan(link)
    elif mode == "watch":
        scan(link); watch(link)
    elif mode == "wiggle" and len(args) > 1:
        wiggle(link, int(args[1]), deg)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
