#!/usr/bin/env python3
"""
servo_grasp.py  —  automatic grasp: every finger closes until ITS sensor touches, then holds a depth
====================================================================================================
No manual sliders. Each moving joint (the IDs that differ between the two TAUGHT poses, e.g.
ID 1-3) is one finger with its own tactile sensor and its own controller:

  CLOSE     the finger closes at --speed deg/s until its sensor reports contact
  SQUEEZE   its position is adjusted so that its depth_max_mm -> --depth-ref
            (too shallow -> close a bit, too deep -> open a bit). All fingers use the same
            reference, so the object ends up pressed evenly and centred by itself: a finger
            that touched first does not push harder while the others are still closing.
  MISS      closed pose reached without contact (nothing under that finger)
  slip on any sensor -> reference +20 % for all fingers (at most once per 1.5 s)

The commands are still angles (the lab ESP32 sketch only takes angles); the sensors decide them.

1) Teach the two poses once (servos hold where servo_gui.py left them):
     ./run.sh --script=servo_gui.py          drag to the OPEN pose, q
     ./run.sh --script=servo_grasp.py teach open
     ./run.sh --script=servo_gui.py          drag to the fully CLOSED pose (empty hand), q
     ./run.sh --script=servo_grasp.py teach closed          -> host/servo_grasp.json

2) Sensors with UDP output (another terminal; no ROS needed):
     ./run.sh triple --usb=camera1 --usb2=camera2 --usb3=camera3 --udp

3) Grasp (window keys:  g grasp   a active   o open   h hold   t tare   q quit):
     g / a first TARE the sensors (their reading now = zero), so press them with nothing touching.
     a = ACTIVE: the fingers stay where they are; push a sensor and that finger gives way /
         follows so that its depth stays at --depth-ref (active compliance, per finger).
     ./run.sh --script=servo_grasp.py --depth-ref=0.2
     ./run.sh --script=servo_grasp.py --map=1:camera2,2:camera1,3:camera3   # which sensor is on which finger
     ./run.sh --script=servo_grasp.py --together          # old behaviour: all fingers as one (any contact)
     ./run.sh --script=servo_grasp.py --no-tactile        # fallback: contact = servo load (per finger)

Options:
  --map=ID:CAM,...   finger (servo ID) -> sensor. Default: the n-th moving ID -> the n-th camera
                     name in sorted order (ID1->camera1, ID2->camera2, ID3->camera3). Check it by
                     pressing one sensor by hand: the window shows which finger lights up.
  --depth-ref=MM     contact depth every finger holds (default 0.30 mm, above the tare value)
  --contact-depth=MM contact only when the depth is this much above the tare value (default 0.10)
  --contact-area=F   ... AND this fraction of the image more than at tare (default 0.02 = 2 %)
  --contact-frames=N ... for N sensor frames in a row (default 3). Filters phantom touches.
  --min-fps=F        warn when a sensor runs slower than this (default 10)
  --depth-max=MM     a finger deeper than this opens a bit (default 0.8 mm)
  --speed=DEG/S      closing speed per finger (default 8)
  --squeeze=DEG/S    max speed while regulating the contact (default 3)
  --together         one controller for all fingers (contact = any sensor, depth = deepest)
  --udp=PORT         listen port (default 9870)
  --load-stop=NM     safety: stop sending if any |load| > this for 0.3 s (default 3.0, 0 = off)
  --no-tactile       contact from each finger's servo load instead of the sensors
  --contact-load=NM  load that counts as contact with --no-tactile (default 0.6)
  --port=/dev/ttyUSB0 --baud=460800

Safety (same as servo_gui.py): stops sending on ERR / lost feedback (0.5 s), sustained load
above --load-stop, a joint lagging its command by > 25 deg. A finger whose sensor stops
sending (0.5 s) holds where it is. Commands are clamped 2 deg inside the sketch limits.
"""
import json
import math
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np

import servo_fw as fw
from servo_gui import Reader

N = fw.N
LIM = np.array([l - 2 for l in fw.LIMIT_DEG], float)
TICK = 0.05
POSES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_grasp.json")
WIN = "Grasp (g grasp, a active, o open, h hold, t tare, q quit)"


def opt(args, name, default, cast=float):
    for a in args:
        if a.startswith(f"--{name}="):
            return cast(a.split("=", 1)[1])
    return default


class Tactile(threading.Thread):
    """Latest JSON message per camera from the sensors (run_sensor_* --udp)."""

    def __init__(self, port):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.2)
        self.lock, self.last, self.events = threading.Lock(), {}, []

    def run(self):
        while True:
            try:
                m = json.loads(self.sock.recvfrom(65536)[0])
            except (socket.timeout, ValueError):
                continue
            m["rx"] = time.monotonic()
            with self.lock:
                self.last[m["cam"]] = m
                self.events += [(m["cam"], e) for e in m.get("events", [])]

    def snapshot(self, max_age=0.5):
        """-> {cam: (contact, depth_mm)} for cameras heard within max_age, and new events."""
        now = time.monotonic()
        with self.lock:
            cur = {c: dict(contact=m.get("contact", 0) > 0.5, depth=float(m.get("depth_max_mm", 0.0)),
                           area=float(m.get("area_frac", 0.0)), fps=float(m.get("fps", 0.0)))
                   for c, m in self.last.items() if now - m["rx"] < max_age}
            ev, self.events = self.events, []
        return cur, ev

    def names(self):
        with self.lock:
            return sorted(self.last)


def load_poses():
    try:
        with open(POSES) as f:
            p = json.load(f)
        return p.get("open"), p.get("closed")
    except (OSError, ValueError):
        return None, None


def teach(which, rd):
    f, _ = rd.get()
    if f is None or any(x is None for x in f):
        sys.exit("[TEACH] no complete feedback (ERR servo / no DC power?)")
    pose = [round(math.degrees(x["pos"]), 2) for x in f]
    out = [i + 1 for i in range(N) if abs(pose[i]) > LIM[i]]
    if out:
        sys.exit(f"[TEACH] ID {out} outside the sketch limits; move it back first")
    p = {}
    if os.path.exists(POSES):
        try:
            p = json.load(open(POSES))
        except ValueError:
            pass
    p[which] = pose
    tmp = POSES + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(p, fh, indent=1)
    os.replace(tmp, POSES)
    print(f"[TEACH] {which} = " + "  ".join(f"ID{i + 1} {d:+.1f}" for i, d in enumerate(pose)))
    print(f"[TEACH] saved {POSES}")


PH_COL = {"open": (150, 150, 150), "close": (240, 180, 60), "squeeze": (90, 200, 90),
          "miss": (60, 190, 240), "hold": (200, 200, 200), "nodata": (60, 60, 230)}


def draw(st, fingers):
    W, H = 900, 130 + 78 * len(fingers) + 66
    img = np.full((H, W, 3), (34, 28, 26), np.uint8)

    def put(s, x, y, c=(220, 220, 220), sc=0.55, th=1):
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)

    mcol = {"IDLE": (160, 160, 160), "GRASP": (90, 200, 90), "ACTIVE": (230, 120, 220), "OPEN": (240, 160, 80),
            "HOLD": (200, 200, 200), "STOPPED": (60, 60, 230)}[st["mode"]]
    put(st["mode"], 14, 36, mcol, 1.0, 2)
    put(f"ref {st['depth_ref']:.2f} mm   max {st['depth_max']:.2f} mm   max |load| {st['load']:.2f} Nm"
        + (f" (stop {st['load_stop']:.1f})" if st["load_stop"] > 0 else ""), 300, 34, (200, 200, 200), 0.5)
    put(st["msg"], 14, 66, (60, 190, 240))
    dmax = max(st["depth_max"], 1e-6)
    for k, fg in enumerate(fingers):
        y = 100 + 78 * k
        put(f"ID {fg['id']}", 14, y + 14, (235, 235, 235), 0.6)
        slow = fg["fresh"] and 0 < fg["fps"] < st["min_fps"]
        put((fg["cam"] or "-") + (f" {fg['fps']:.0f}fps" if fg["fresh"] else ""), 14, y + 36,
            (60, 60, 230) if slow else (170, 170, 170), 0.45)
        ph = fg["phase"]
        put(ph.upper(), 110, y + 14, PH_COL.get(ph, (200, 200, 200)), 0.5)
        # live servo position on the open -> closed line: green = actual, blue = sent, orange = target
        x0, x1, yb = 210, 560, y + 7
        o, c = fg["open_deg"], fg["closed_deg"]
        sx = lambda d: int(x0 + np.clip((d - o) / (c - o), -0.05, 1.05) * (x1 - x0))
        cv2.line(img, (x0, yb), (x1, yb), (80, 80, 80), 2)
        for xx in (x0, x1):
            cv2.line(img, (xx, yb - 7), (xx, yb + 7), (120, 120, 120), 1)
        put("open", x0 - 2, yb - 11, (120, 120, 120), 0.35)
        put("closed", x1 - 34, yb - 11, (120, 120, 120), 0.35)
        if fg["tgt_deg"] is not None:
            cv2.circle(img, (sx(fg["tgt_deg"]), yb), 8, (60, 160, 240), 2)
        if fg["cmd_deg"] is not None:
            cv2.circle(img, (sx(fg["cmd_deg"]), yb), 4, (230, 160, 60), -1)
        if fg["act"] is not None:
            cv2.circle(img, (sx(fg["act"]), yb), 6, (120, 220, 120), -1)
            lc = (90, 200, 90) if abs(fg["load"]) < 1.0 else (60, 190, 240) if abs(fg["load"]) < 2.0 else (60, 60, 230)
            txt = f"act {fg['act']:+6.1f}"
            if fg["cmd_deg"] is not None:
                txt += f"  sent {fg['cmd_deg']:+6.1f}  tgt {fg['tgt_deg']:+6.1f} deg"
            put(txt, x0, y + 32, (220, 220, 220), 0.45)
            put(f"load {fg['load']:+.2f} Nm", x0, y + 52, lc, 0.45)
            put(f"closed {100 * fg['u_act']:.0f} %", x0 + 150, y + 52, (170, 170, 170), 0.45)
        else:
            put("ERR / no feedback", x0, y + 32, (60, 60, 230), 0.45)
        # depth vs reference
        x0, x1 = 590, W - 20
        cv2.rectangle(img, (x0, y), (x1, y + 14), (70, 70, 70), 1)
        cv2.rectangle(img, (x0, y), (x0 + int((x1 - x0) * min(1, fg["depth"] / dmax)), y + 14),
                      (90, 200, 90) if fg["contact"] else (120, 120, 120), -1)
        xr = x0 + int((x1 - x0) * min(1, st["depth_ref"] / dmax))
        cv2.line(img, (xr, y - 4), (xr, y + 18), (60, 190, 240), 2)
        put(f"depth {fg['depth']:.2f} mm" + ("  CONTACT" if fg["contact"] else ""), x0, y + 34,
            (90, 200, 90) if fg["contact"] else (170, 170, 170), 0.45)
    # every servo, live
    yy = H - 40
    put("all:", 14, yy, (150, 150, 150), 0.45)
    for i, (a, l) in enumerate(st["servo"]):
        put(f"ID{i + 1} " + (f"{a:+6.1f}" if a is not None else "  ERR"), 60 + 138 * i, yy,
            (220, 220, 220) if a is not None else (60, 60, 230), 0.45)
    put("g grasp  a active  o open  h hold  t tare  q quit    servo: green = actual, blue = sent, ring = target"
        "   depth: green above tare, line = ref", 14, H - 14, (150, 150, 150), 0.4)
    return img


def main():
    args = sys.argv[1:]
    port = opt(args, "port", None, str) or fw.find_port()
    baud = opt(args, "baud", 460800, int)
    link = fw.Link(port, baud)
    print(f"[INFO] {port} @ {baud}")
    if not fw.wait_ready(link):
        sys.exit("[ERROR] no feedback within 6 s (DC power on? sketch at 460800?)")
    rd = Reader(link); rd.start()
    time.sleep(0.3)

    if args and args[0] == "teach":
        if len(args) < 2 or args[1] not in ("open", "closed"):
            sys.exit("usage: servo_grasp.py teach open|closed")
        teach(args[1], rd)
        return

    P_open, P_closed = load_poses()
    if P_open is None or P_closed is None:
        sys.exit(f"[ERROR] teach both poses first (missing in {POSES}):\n"
                 f"  ./run.sh --script=servo_grasp.py teach open\n  ./run.sh --script=servo_grasp.py teach closed")
    P_open, P_closed = np.array(P_open, float), np.array(P_closed, float)
    span = P_closed - P_open
    moving = [j for j in range(N) if abs(span[j]) > 2.0]
    if not moving:
        sys.exit("[ERROR] open and closed poses are (almost) the same; teach them again")

    no_tact = "--no-tactile" in args
    together = "--together" in args
    speed, squeeze = opt(args, "speed", 8.0), opt(args, "squeeze", 3.0)
    contact_load = opt(args, "contact-load", 0.6)
    c_depth = opt(args, "contact-depth", 0.10)          # mm above the tare value
    c_area = opt(args, "contact-area", 0.02)            # fraction of the image above the tare value
    c_frames = int(opt(args, "contact-frames", 3))      # consecutive sensor messages
    min_fps = opt(args, "min-fps", 10.0)
    st = dict(mode="IDLE", msg="g = grasp", load=0.0, depth_ref=opt(args, "depth-ref", 0.30),
              depth_max=opt(args, "depth-max", 0.80), load_stop=opt(args, "load-stop", 3.0), t_slip=0.0,
              min_fps=opt(args, "min-fps", 10.0), servo=[(None, 0.0)] * N)
    ref0 = st["depth_ref"]
    tact = None if no_tact else Tactile(opt(args, "udp", 9870, int))
    if tact:
        tact.start()
        print(f"[INFO] listening for tactile data on UDP {tact.sock.getsockname()[1]} (sensor with --udp)")

    # finger = one moving joint; u = 0 (open) .. 1 (closed) along its own taught range
    user_map = {}
    for item in filter(None, opt(args, "map", "", str).split(",")):
        i, c = item.split(":")
        user_map[int(i)] = c
    fingers = [dict(id=j + 1, j=j, cam=user_map.get(j + 1), u=0.0, u_cmd=0.0, phase="open",
                    contact=False, depth=0.0, d0=0.0, a0=0.0, n_on=0, fps=0.0, raw=None,
                    open_deg=float(P_open[j]), closed_deg=float(P_closed[j]), act=None, load=0.0,
                    u_act=0.0, cmd_deg=None, tgt_deg=None, fresh=False) for j in moving]

    def tare():
        """Current sensor reading of every finger = zero (removes a static phantom offset).
        Only call it while the fingers touch nothing."""
        for fg in fingers:
            r = fg["raw"]
            fg["d0"], fg["a0"], fg["n_on"] = (r["depth"], r["area"], 0) if r else (0.0, 0.0, 0)
        print("[TARE] " + "  ".join(f"ID{fg['id']} d0={fg['d0']:.2f}mm a0={100 * fg['a0']:.1f}%" for fg in fingers))

    def assign_cams():
        """Default map: n-th moving ID -> n-th camera heard (sorted names)."""
        if tact is None or together:
            return
        names = tact.names()
        for k, fg in enumerate(fingers):
            if fg["id"] not in user_map:
                fg["cam"] = names[k] if k < len(names) else None
        print("[MAP] " + "  ".join(f"ID{fg['id']}->{fg['cam']}" for fg in fingers))

    cmd = None
    sending, t_start, heavy_n = False, 0.0, 0
    t_next = time.monotonic()
    cv2.namedWindow(WIN)

    def stop(why):
        nonlocal sending
        if sending:
            print(f"[STOP] {why}")
        sending = False
        st["mode"], st["msg"] = "STOPPED", why

    def start(mode, msg):
        nonlocal sending, cmd, t_start, heavy_n
        f, _ = rd.get()
        if f is None or any(x is None for x in f):
            st["msg"] = "cannot start: a servo reads ERR or no feedback"
            return False
        act = np.array([math.degrees(x["pos"]) for x in f])
        if not sending:
            cmd, t_start, heavy_n = act.copy(), time.monotonic(), 0
            for fg in fingers:                       # start from where each finger is now
                fg["u"] = float(np.clip((act[fg["j"]] - P_open[fg["j"]]) / span[fg["j"]], 0, 1))
        sending = True
        st["mode"], st["msg"] = mode, msg
        print(f"[{mode}] {msg}")
        return True

    while True:
        now = time.monotonic()
        f, t_fb = rd.get()
        st["load"] = max([abs(x["load"]) for x in f if x is not None] or [0.0]) if f else 0.0
        st["servo"] = [((math.degrees(x["pos"]), x["load"]) if x is not None else (None, 0.0))
                       for x in f] if f else [(None, 0.0)] * N
        for fg in fingers:
            a, l = st["servo"][fg["j"]]
            fg["act"], fg["load"] = a, l
            fg["u_act"] = float(np.clip((a - P_open[fg["j"]]) / span[fg["j"]], 0, 1)) if a is not None else 0.0

        # ---- per-finger contact / depth ----
        if tact:
            cur, events = tact.snapshot()
            for cam, e in events:
                print(f"[TACTILE] {cam}: {e}")
                if e == "slip_start" and st["mode"] == "GRASP" and now - st["t_slip"] > 1.5:
                    st["t_slip"] = now
                    st["depth_ref"] = min(st["depth_max"] * 0.9, st["depth_ref"] * 1.2)
                    st["msg"] = f"slip on {cam} -> all fingers squeeze harder (ref {st['depth_ref']:.2f} mm)"
            for fg in fingers:
                if together:
                    v = max(cur.values(), key=lambda r: r["depth"]) if cur else None
                else:
                    v = cur.get(fg["cam"])
                fg["raw"], fg["fresh"] = v, v is not None
                if v is None:
                    fg["contact"], fg["depth"], fg["n_on"], fg["fps"] = False, 0.0, 0, 0.0
                    continue
                fg["fps"] = v["fps"]
                fg["depth"] = max(0.0, v["depth"] - fg["d0"])
                on = fg["depth"] >= c_depth and (v["area"] - fg["a0"]) >= c_area
                fg["n_on"] = fg["n_on"] + 1 if on else 0
                # contact: deep AND large enough for c_frames frames; released below half the depth
                if fg["contact"]:
                    fg["contact"] = fg["depth"] >= 0.5 * c_depth
                else:
                    fg["contact"] = fg["n_on"] >= c_frames
        else:
            for fg in fingers:
                x = f[fg["j"]] if f else None
                fg["fresh"] = x is not None
                fg["contact"] = x is not None and abs(x["load"]) > contact_load
                fg["depth"] = 0.0

        # ---- control ----
        if sending:
            if f is None or now - t_fb > 0.5:
                stop("feedback stopped -> not sending")
            elif any(x is None for x in f):
                stop("a servo reads ERR -> not sending")
            else:
                act = np.array([math.degrees(x["pos"]) for x in f])
                heavy = st["load_stop"] > 0 and st["load"] > st["load_stop"] and now - t_start > 1.0
                heavy_n = heavy_n + 1 if heavy else 0
                lag = [i + 1 for i in range(N) if abs(act[i] - cmd[i]) > 25]
                if heavy_n >= 6:
                    stop(f"load > {st['load_stop']:.1f} Nm -> stopped (servos hold)")
                elif lag:
                    stop(f"ID {lag} not following (> 25 deg behind) -> stopped")
                elif now >= t_next:
                    dt = TICK
                    for fg in fingers:
                        rate = 1.0 / abs(span[fg["j"]])       # u per degree
                        if st["mode"] == "OPEN":
                            fg["u"] = max(0.0, fg["u"] - speed * rate * dt)
                            fg["phase"] = "open"
                        elif st["mode"] == "HOLD":
                            fg["phase"] = "hold"
                        elif st["mode"] == "ACTIVE":
                            # stay put; a touched finger keeps ITS depth at the reference:
                            # pushed harder -> gives way, pushed less -> follows the object
                            if not fg["fresh"] or no_tact:
                                fg["phase"] = "nodata" if not fg["fresh"] else "hold"
                            elif not fg["contact"]:
                                fg["phase"] = "hold"
                            else:
                                fg["phase"] = "squeeze"
                                d = fg["depth"]
                                e = -1.0 if d > st["depth_max"] else float(
                                    np.clip((st["depth_ref"] - d) / max(st["depth_ref"], 1e-3), -1, 1))
                                if abs(e) < 0.1:
                                    e = 0.0
                                fg["u"] = float(np.clip(fg["u"] + e * squeeze * rate * dt, 0, 1))
                        elif st["mode"] == "GRASP":
                            if not fg["fresh"]:
                                fg["phase"] = "nodata"         # its sensor is silent: hold
                            elif fg["phase"] in ("open", "close", "miss", "hold", "nodata") and not fg["contact"]:
                                if fg["u"] >= 1.0:
                                    fg["phase"] = "miss"
                                else:
                                    fg["phase"] = "close"
                                    fg["u"] = min(1.0, fg["u"] + speed * rate * dt)
                            else:
                                if fg["phase"] != "squeeze":
                                    print(f"[CONTACT] ID {fg['id']} ({fg['cam'] or 'load'})")
                                fg["phase"] = "squeeze"
                                if no_tact:
                                    continue                     # load fallback: hold at first contact
                                ref, d = st["depth_ref"], fg["depth"]
                                if d > st["depth_max"]:
                                    e = -1.0
                                elif not fg["contact"]:
                                    e = 1.0                      # lost the contact: close again
                                else:
                                    e = float(np.clip((ref - d) / max(ref, 1e-3), -1, 1))
                                    if abs(e) < 0.1:
                                        e = 0.0                  # dead band +-10 %
                                fg["u"] = float(np.clip(fg["u"] + e * squeeze * rate * dt, 0, 1))
                    if st["mode"] == "GRASP":
                        ph = [fg["phase"] for fg in fingers]
                        if all(p == "squeeze" for p in ph):
                            st["msg"] = "all fingers in contact: holding depth " + f"{st['depth_ref']:.2f} mm"
                        elif all(p in ("squeeze", "miss") for p in ph) and "miss" in ph:
                            st["msg"] = "finger(s) " + ",".join(str(fg["id"]) for fg in fingers
                                                                if fg["phase"] == "miss") + " closed without contact"
                        elif "nodata" in ph:
                            st["msg"] = "no data from " + ",".join(str(fg["cam"]) for fg in fingers
                                                                   if fg["phase"] == "nodata") + " -> that finger holds"
                    if st["mode"] == "OPEN" and all(fg["u"] <= 0 for fg in fingers) \
                            and np.all(np.abs(cmd - P_open) < 0.5):
                        st["mode"], st["msg"] = "IDLE", "open. g = grasp"
                        st["depth_ref"] = ref0
                    # target pose -> ramp the command (each joint <= speed deg/s)
                    tgt = P_open.copy()
                    for fg in fingers:
                        tgt[fg["j"]] = P_open[fg["j"]] + fg["u"] * span[fg["j"]]
                    tgt = np.clip(tgt, -LIM, LIM)
                    step = max(speed, squeeze) * dt
                    cmd = cmd + np.clip(tgt - cmd, -step, step)
                    link.send([math.radians(c) for c in cmd])
                    t_next = now + TICK
        for fg in fingers:
            fg["cmd_deg"] = float(cmd[fg["j"]]) if (cmd is not None and sending) else None
            fg["tgt_deg"] = float(P_open[fg["j"]] + fg["u"] * span[fg["j"]]) if sending else None
            fg["u_cmd"] = (float(np.clip((cmd[fg["j"]] - P_open[fg["j"]]) / span[fg["j"]], 0, 1))
                           if cmd is not None else 0.0)

        cv2.imshow(WIN, draw(st, fingers))
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            stop("quit")
            break
        if k in (ord("g"), ord("a")):
            if tact and not any(fg["fresh"] for fg in fingers) and not together:
                assign_cams()
            if tact and not tact.snapshot()[0]:
                st["msg"] = "no tactile data: start the sensors with --udp (or use --no-tactile)"
            else:
                if not sending or st["mode"] in ("IDLE", "OPEN", "STOPPED"):
                    assign_cams()
                    time.sleep(0.15)                     # one fresh message per camera
                    cur, _ = tact.snapshot() if tact else ({}, [])
                    for fg in fingers:
                        fg["raw"] = (cur.get(fg["cam"]) if not together
                                     else (max(cur.values(), key=lambda r: r["depth"]) if cur else None))
                        fg["contact"] = False
                    tare()
                for fg in fingers:
                    fg["phase"] = "open"
                slow = [f"{fg['cam']} {fg['fps']:.0f} fps" for fg in fingers if fg["fresh"] and 0 < fg["fps"] < min_fps]
                if k == ord("g"):
                    start("GRASP", f"each finger closes at {speed:.0f} deg/s until its sensor touches")
                else:
                    start("ACTIVE", "fingers stay; touch a sensor: that finger keeps the reference depth")
                if slow:
                    st["msg"] = "WARNING low fps: " + ", ".join(slow) + " -> USB bandwidth (own port per Zero)"
                    print("[WARN] " + st["msg"])
        elif k == ord("t"):
            tare(); st["msg"] = "tare: current sensor readings = zero"
        elif k == ord("o"):
            start("OPEN", "opening")
        elif k == ord("h") and sending:
            st["mode"], st["msg"] = "HOLD", "holding here (g = grasp again, o = open)"
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
