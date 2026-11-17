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
  python3 host/servo_fw.py diag 20         # 20 s feedback check: rate, gaps, ERR per servo, ESP32 reboots
  python3 host/servo_fw.py diag 20 hold    # same while holding the current angles (torque on, no motion)
  python3 host/servo_fw.py sweep 1 -60     # ID1 slowly (5 deg/s) to -60 deg and back, others hold:
                                          #   prints the angle where a servo reads ERR / feedback stalls
  python3 host/servo_fw.py wiggle 3        # ID 3 moves +-5 deg and back, the others HOLD where they are
  python3 host/servo_fw.py zero            # CURRENT pose = new zero: prints zero_offset[] for the sketch
                                           # (sends nothing; --zero=a,..,f if the sketch's values changed)
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
# the sketch clamps every command: servos 1-3 to +-110 deg, servos 4-6 to +-120 deg
LIMIT_DEG = [110, 110, 110, 120, 120, 120]
# zero_offset[] currently in the sketch (raw encoder ticks 0..4095); --zero=a,b,c,d,e,f if changed
ZERO_SKETCH = [2104, 2051, 2013, 1343, 2663, 3033]


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


def open_or_explain(port, baud, timeout):
    try:
        return serial.Serial(port, baud, timeout=timeout)
    except serial.SerialException as e:
        busy_exit(port, e)


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
        self.s = open_or_explain(port, baud, 0.2)
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
            d = math.degrees(last[i]["pos"])
            warn = (f"   <-- OUTSIDE the sketch limit +-{LIMIT_DEG[i]} deg: any command snaps it there"
                    if abs(d) > LIMIT_DEG[i] else "")
            print(f"  ID {i + 1}: answers ({good}/{len(samples)})   {fmt(last[i])}{warn}")
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
    out = [(i + 1, math.degrees(q0[i])) for i in range(N) if abs(math.degrees(q0[i])) > LIMIT_DEG[i] - 1]
    if out:
        txt = ", ".join(f"ID {i} at {d:+.0f} deg (limit +-{LIMIT_DEG[i - 1]})" for i, d in out)
        sys.exit(f"[WIGGLE] refused: {txt}. ANY command would snap it to the limit at full speed. "
                 "Move it by hand inside the limit, or re-check zero_offset in the sketch.")
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


def rezero(link, old):
    """New zero_offset so that the CURRENT pose reads 0 rad. Sends nothing.
    raw = zero + rad*4095/(2*pi) (same conversion as the sketch). The feedback has 2
    decimals (0.01 rad = 6.5 ticks), so the result is within about +-3 ticks (+-0.3 deg)."""
    samples = [f for f in (parse(ln) for ln in link.lines(2.0)) if f is not None]
    if not samples:
        sys.exit("[ZERO] no feedback")
    new = []
    for i in range(N):
        vals = [x[i]["pos"] for x in samples if x[i] is not None]
        if not vals:
            sys.exit(f"[ZERO] ID {i + 1} gives no reading: not re-zeroed (fix it first)")
        rad = sum(vals) / len(vals)
        raw = int(round(old[i] + rad * 4095.0 / (2.0 * math.pi))) % 4096
        new.append(raw)
        print(f"  ID {i + 1}: now {math.degrees(rad):+7.1f} deg from old zero {old[i]:4d} "
              f"-> new zero {raw:4d}")
    print("\n[ZERO] put this line in the sketch (replace the old one), then upload:\n")
    print(f"int zero_offset[num_servos] = {{{','.join(str(v) for v in new)}}};\n")
    print("Accuracy about +-3 ticks (+-0.3 deg): the feedback has 2 decimals. Keep the arm still "
          "while this runs.")


def diag(link, secs, hold):
    """Record the feedback stream (nothing moves): line rate, gaps, ERR per servo, ESP32 reboots,
    garbage lines. With hold=True it sends the CURRENT angles every 50 ms (torque on, no motion) -
    the situation in which servo_gui stops with 'feedback stopped'."""
    print(f"[DIAG] {secs:.0f} s, {'HOLD (sending current angles, torque on, no motion)' if hold else 'listen only (nothing sent)'}")
    t0 = last = time.monotonic()
    n_ok, n_bad, gaps, boots, other = 0, 0, [], [], []
    err = [0] * N
    loads = [[] for _ in range(N)]
    hold_q, t_send, sent_stop = None, 0.0, ""
    def stream():                                   # unbuffered: real arrival time of every line
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            link.buf += link.s.read(max(1, link.s.in_waiting))
            *full, link.buf = link.buf.split(b"\n")
            for x in full:
                yield x.decode("utf-8", "replace").strip()
    for ln in stream():
        now = time.monotonic()
        if not ln:
            continue
        f = parse(ln)
        if f is None:
            low = ln.lower()
            if "rst:" in low or low.startswith("ets ") or "initialized" in low:
                boots.append((now - t0, ln[:70]))
                print(f"[DIAG] {now - t0:6.2f}s ESP32 boot text: {ln[:70]}")
            else:
                n_bad += 1
                if len(other) < 8:
                    other.append((now - t0, ln[:70]))
            continue
        if now - last > 0.15:
            gaps.append((last - t0, now - last))
            print(f"[DIAG] {last - t0:6.2f}s no feedback for {now - last:.2f} s"
                  + ("   <-- servo_gui stops here (> 0.5 s)" if now - last > 0.5 else ""))
        last = now
        n_ok += 1
        for i, x in enumerate(f):
            if x is None:
                err[i] += 1
            else:
                loads[i].append(abs(x["load"]))
        if hold:
            if any(x is None for x in f):
                if not sent_stop:
                    sent_stop = f"{now - t0:.2f}s"
                    print(f"[DIAG] {now - t0:6.2f}s ID {[i + 1 for i, x in enumerate(f) if x is None]} ERR "
                          f"-> stop sending (a command would send it to 0 rad)")
                hold = False
            else:
                if hold_q is None:
                    hold_q = [x["pos"] for x in f]
                if now - t_send > 0.05:
                    link.send(hold_q); t_send = now
    dur = time.monotonic() - t0
    print("\n[DIAG] ---- summary ----")
    print(f"  feedback lines: {n_ok}  ({n_ok / dur:.1f} /s)")
    print(f"  gaps > 0.15 s: {len(gaps)}" + (f", longest {max(g for _, g in gaps):.2f} s" if gaps else "")
          + f";  > 0.5 s (servo_gui stop): {sum(1 for _, g in gaps if g > 0.5)}")
    print(f"  ESP32 reboots: {len(boots)}" + ("   <-- power dip (brown-out) or reset" if boots else ""))
    print(f"  garbage / other lines: {n_bad}" + "".join(f"\n     {t:6.2f}s {l}" for t, l in other))
    for i in range(N):
        pct = 100.0 * err[i] / max(1, n_ok)
        L = loads[i]
        print(f"  ID{i + 1}: ERR {err[i]:5d} ({pct:5.1f} %)" + (f"   load max {max(L):.2f} Nm" if L else "")
              + ("   <-- does not answer reliably" if pct > 1 else ""))
    if sent_stop:
        print(f"  hold stopped at {sent_stop} because of ERR")


def sweep(link, sid, to_deg, speed=5.0, load_lim=0.6):
    """Move ONE servo slowly from where it is to to_deg and back (others hold), and record at which
    angle a servo reads ERR / the feedback stalls. On ERR the commands keep the last values (hold)."""
    f = None
    for ln in link.lines(1.0):
        g = parse(ln)
        if g is not None and all(x is not None for x in g):
            f = g
    if f is None:
        sys.exit("[SWEEP] no complete feedback (a servo reads ERR now) - fix that first")
    cmd = [math.degrees(x["pos"]) for x in f]
    i = sid - 1
    start = cmd[i]
    lim = LIMIT_DEG[i] - 2
    to_deg = max(-lim, min(lim, to_deg))
    print(f"[SWEEP] ID{sid}: {start:+.1f} -> {to_deg:+.1f} -> {start:+.1f} deg at {speed:.0f} deg/s, "
          f"stop if |load| > {load_lim:.2f} Nm.  Ctrl+C = stop (servos hold)")
    legs = [to_deg, start]
    t_last = time.monotonic(); t_send = 0.0; events = []; n = 0; n_err = 0
    last_act = start; err_now = False; t_ok = time.monotonic()

    def fresh():                                    # unbuffered: what arrived since the last call
        link.buf += link.s.read(link.s.in_waiting) if link.s.in_waiting else b""
        *full, link.buf = link.buf.split(b"\n")
        return [x.decode("utf-8", "replace").strip() for x in full]
    try:
        while legs:
            time.sleep(0.005)
            for ln in fresh():
                g = parse(ln)
                if g is None:
                    continue
                now = time.monotonic()
                if now - t_last > 0.15:
                    events.append((cmd[i], last_act, f"feedback stalled {now - t_last:.2f} s"))
                    print(f"[SWEEP] cmd {cmd[i]:+6.1f}  act {last_act:+6.1f}: feedback stalled {now - t_last:.2f} s")
                t_last = now; n += 1
                bad = [k + 1 for k, x in enumerate(g) if x is None]
                err_now = bool(bad)
                if bad:
                    n_err += 1
                    events.append((cmd[i], last_act, f"ERR ID {bad}"))
                    print(f"[SWEEP] cmd {cmd[i]:+6.1f}  act {last_act:+6.1f}: ERR ID {bad}")
                    continue
                t_ok = now
                last_act = math.degrees(g[i]["pos"])
                L = abs(g[i]["load"])
                if L > load_lim:
                    print(f"[SWEEP] load {L:.2f} Nm at {last_act:+.1f} deg -> going back"); legs = legs[1:]
                    events.append((cmd[i], last_act, f"load {L:.2f} Nm"))
                    cmd[i] = last_act
                    break
            now = time.monotonic()
            if now - t_ok > 2.0:
                print(f"[SWEEP] no complete feedback for 2 s at cmd {cmd[i]:+.1f} deg -> stop (servos hold)")
                events.append((cmd[i], last_act, "lost for > 2 s -> stopped"))
                break
            if legs and now - t_send > 0.05 and (err_now or now - t_last > 0.15):
                link.send([math.radians(c) for c in cmd]); t_send = now      # ERR / stall: hold, no advance
            elif legs and now - t_send > 0.05:
                d = legs[0] - cmd[i]
                step = speed * (now - t_send if t_send else 0.05)
                cmd[i] += max(-step, min(step, d))
                if abs(legs[0] - cmd[i]) < 1e-6:
                    legs = legs[1:]
                link.send([math.radians(c) for c in cmd]); t_send = now
    except KeyboardInterrupt:
        print("\n[SWEEP] stopped (servos hold)")
    print(f"\n[SWEEP] ---- summary ----\n  feedback lines {n}, ERR lines {n_err}")
    if not events:
        print(f"  ID{sid} swept {start:+.1f} <-> {to_deg:+.1f} deg: no ERR, no stall, no overload")
    for c, a, e in events:
        print(f"  at cmd {c:+6.1f} deg (act {a:+6.1f}): {e}")


def main():
    args = sys.argv[1:]
    port, baud, deg, zero = None, 460800, 5.0, list(ZERO_SKETCH)
    for a in list(args):
        if a.startswith("--port="):
            port = a.split("=", 1)[1]; args.remove(a)
        elif a.startswith("--baud="):
            baud = int(a.split("=", 1)[1]); args.remove(a)
        elif a.startswith("--deg="):
            deg = float(a.split("=", 1)[1]); args.remove(a)
        elif a.startswith("--zero="):
            zero = [int(v) for v in a.split("=", 1)[1].split(",")]; args.remove(a)
            if len(zero) != N:
                sys.exit("--zero needs 6 values (zero_offset of the sketch)")
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
    elif mode == "sweep" and len(args) > 2:
        sweep(link, int(args[1]), float(args[2]), speed=deg)          # --deg= sets the speed (deg/s)
    elif mode == "diag":
        rest = [a for a in args[1:] if a != "hold"]
        diag(link, float(rest[0]) if rest else 20.0, "hold" in args[1:])
    elif mode == "zero":
        rezero(link, zero)
    elif mode == "wiggle" and len(args) > 1:
        wiggle(link, int(args[1]), deg)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
