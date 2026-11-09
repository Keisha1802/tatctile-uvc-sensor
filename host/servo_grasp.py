#!/usr/bin/env python3
"""
servo_grasp.py  —  automatic grasp with the tactile sensor: close until touch, then hold a set depth
==================================================================================================
No manual sliders: the fingers move along a straight line between two TAUGHT poses
(open -> closed, all 6 servos, so no need to know which ID is which finger or which way is
"close"). The position is still what goes to the servos (the lab ESP32 sketch only takes
angles), but the tactile sensor decides where along that line to stop:

  APPROACH   close at --speed deg/s until the sensor reports contact
  SQUEEZE    regulate the contact: target position moves so that depth_max_mm -> --depth-ref
             (too shallow -> close a bit, too deep -> open a bit; slip -> target +20 %)
  OPEN       back to the open pose, slowly

1) Teach the two poses once (servos hold where servo_gui.py left them):
     ./run.sh --script=servo_gui.py          drag to the OPEN pose, SPACE (stop sending), q
     python3 servo_grasp.py teach open
     ./run.sh --script=servo_gui.py          drag to the fully CLOSED pose (empty hand), q
     python3 servo_grasp.py teach closed     -> saved in servo_grasp.json

2) Start the sensor with the UDP output (another terminal; no ROS needed):
     python3 run_sensor_fast.py usb --udp                         (or dual ... --udp)

3) Grasp:
     python3 servo_grasp.py                  window keys:  g grasp   o open   h hold   q quit
     python3 servo_grasp.py --depth-ref=0.3 --speed=8 --cam=camera1
     python3 servo_grasp.py --no-tactile --contact-load=0.6     (fallback: contact = servo load)

Options:
  --depth-ref=MM     contact depth to hold (default 0.30 mm)
  --depth-max=MM     never squeeze beyond this depth: open a bit (default 0.8 mm)
  --speed=DEG/S      approach speed of the fastest joint (default 8)
  --squeeze=DEG/S    max speed while regulating the contact (default 3)
  --cam=NAME         only use this sensor (default: any sensor that sends)
  --udp=PORT         listen port (default 9870)
  --load-stop=NM     safety: stop sending if |load| > this for 0.3 s (default 3.0, 0 = off)
  --no-tactile       contact from servo load instead of the sensor
  --contact-load=NM  load that counts as contact with --no-tactile (default 0.6)
  --port=/dev/ttyUSB0 --baud=460800
(sg dialout -c "python3 servo_grasp.py" if you are not in the dialout group)

Safety (same as servo_gui.py): stops sending on ERR / lost feedback (0.5 s), sustained load
above --load-stop, a joint lagging its command by > 25 deg, or tactile data stopping for 0.5 s
while squeezing (then it just holds). Closed pose reached with no contact -> holds there
("nothing grasped"). Commands are clamped 2 deg inside the sketch limits.
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
LIM = [l - 2 for l in fw.LIMIT_DEG]
TICK = 0.05
POSES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "servo_grasp.json")
WIN = "Grasp (g grasp, o open, h hold, q quit)"


def opt(args, name, default, cast=float):
    for a in args:
        if a.startswith(f"--{name}="):
            return cast(a.split("=", 1)[1])
    return default


class Tactile(threading.Thread):
    """Latest JSON message per camera from the sensor (run_sensor_* --udp)."""

    def __init__(self, port, cam=None):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.2)
        self.cam, self.lock, self.last, self.events = cam, threading.Lock(), {}, []

    def run(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
                m = json.loads(data)
            except (socket.timeout, ValueError):
                continue
            if self.cam and m.get("cam") != self.cam:
                continue
            m["rx"] = time.monotonic()
            with self.lock:
                self.last[m["cam"]] = m
                self.events += [(m["cam"], e) for e in m.get("events", [])]

    def get(self, max_age=0.5):
        """-> (fresh, contact, depth_mm, cams, events since last call)."""
        now = time.monotonic()
        with self.lock:
            fresh = [m for m in self.last.values() if now - m["rx"] < max_age]
            ev, self.events = self.events, []
        if not fresh:
            return False, False, 0.0, [], ev
        contact = any(m.get("contact", 0) > 0.5 for m in fresh)
        depth = max(m.get("depth_max_mm", 0.0) for m in fresh)
        return True, contact, depth, sorted(m["cam"] for m in fresh), ev


def load_poses():
    try:
        with open(POSES) as f:
            p = json.load(f)
        return p.get("open"), p.get("closed")
    except (OSError, ValueError):
        return None, None


def teach(which, link, rd):
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


def draw(st):
    W, H = 640, 300
    img = np.full((H, W, 3), (34, 28, 26), np.uint8)
    put = lambda s, y, c=(220, 220, 220), sc=0.55: cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                                                            sc, c, 1, cv2.LINE_AA)
    col = {"IDLE": (160, 160, 160), "APPROACH": (240, 180, 60), "SQUEEZE": (90, 200, 90),
           "OPEN": (240, 160, 80), "HOLD": (200, 200, 200), "STOPPED": (60, 60, 230)}[st["mode"]]
    cv2.putText(img, st["mode"], (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
    put(st["msg"], 64, (60, 190, 240))
    # closing fraction u
    x0, x1, y = 14, W - 14, 96
    cv2.rectangle(img, (x0, y), (x1, y + 16), (70, 70, 70), 1)
    cv2.rectangle(img, (x0, y), (x0 + int((x1 - x0) * st["u_cmd"]), y + 16), (230, 160, 60), -1)
    put(f"closing  {100 * st['u_cmd']:.0f} %   (0 = open pose, 100 = closed pose)", y + 36)
    # depth vs. reference
    y = 160
    dmax = max(st["depth_max"], 1e-6)
    cv2.rectangle(img, (x0, y), (x1, y + 16), (70, 70, 70), 1)
    cv2.rectangle(img, (x0, y), (x0 + int((x1 - x0) * min(1, st["depth"] / dmax)), y + 16),
                  (90, 200, 90) if st["contact"] else (120, 120, 120), -1)
    xr = x0 + int((x1 - x0) * min(1, st["depth_ref"] / dmax))
    cv2.line(img, (xr, y - 4), (xr, y + 20), (60, 190, 240), 2)
    src = "servo load" if st["no_tactile"] else ("sensor " + ",".join(st["cams"]) if st["cams"] else "NO SENSOR DATA")
    put(f"depth {st['depth']:.2f} mm  ref {st['depth_ref']:.2f}  max {st['depth_max']:.2f}   "
        f"contact {'YES' if st['contact'] else 'no'}   [{src}]", y + 36)
    put(f"max |load| {st['load']:.2f} Nm" + (f"  (stop at {st['load_stop']:.1f})" if st["load_stop"] > 0 else ""),
        y + 62, (200, 200, 200), 0.5)
    put("g grasp   o open   h hold   q quit", H - 14, (150, 150, 150), 0.5)
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
        teach(args[1], link, rd)
        return

    P_open, P_closed = load_poses()
    if P_open is None or P_closed is None:
        sys.exit(f"[ERROR] teach both poses first (missing in {POSES}):\n"
                 f"  python3 servo_grasp.py teach open\n  python3 servo_grasp.py teach closed")
    P_open, P_closed = np.array(P_open, float), np.array(P_closed, float)
    span = float(np.max(np.abs(P_closed - P_open)))
    if span < 2:
        sys.exit("[ERROR] open and closed poses are (almost) the same; teach them again")

    no_tact = "--no-tactile" in args
    st = dict(mode="IDLE", msg="g = grasp", u_cmd=0.0, depth=0.0, contact=False, cams=[], load=0.0,
              depth_ref=opt(args, "depth-ref", 0.30), depth_max=opt(args, "depth-max", 0.80),
              load_stop=opt(args, "load-stop", 3.0), no_tactile=no_tact)
    speed, squeeze = opt(args, "speed", 8.0), opt(args, "squeeze", 3.0)
    contact_load = opt(args, "contact-load", 0.6)
    ref0 = st["depth_ref"]
    tact = None if no_tact else Tactile(opt(args, "udp", 9870, int), opt(args, "cam", None, str))
    if tact:
        tact.start()
        print(f"[INFO] listening for tactile data on UDP {tact.sock.getsockname()[1]} "
              f"(start the sensor with --udp)")
    moving = np.abs(P_closed - P_open) > 0.5            # joints that take part in the grasp

    cmd = None                 # joint command actually sent (deg), ramps toward the target
    u = 0.0                    # target position along open -> closed
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
        nonlocal sending, cmd, t_start, heavy_n, u
        f, _ = rd.get()
        if f is None or any(x is None for x in f):
            st["msg"] = "cannot start: a servo reads ERR or no feedback"
            return
        act = np.array([math.degrees(x["pos"]) for x in f])
        if not sending:
            cmd, t_start, heavy_n = act.copy(), time.monotonic(), 0
            # where are we along the line? (projection, so a grasp can start from anywhere)
            d = P_closed - P_open
            u = float(np.clip(np.dot(act - P_open, d) / np.dot(d, d), 0, 1))
        sending = True
        st["mode"], st["msg"] = mode, msg
        print(f"[{mode}] {msg}")

    while True:
        now = time.monotonic()
        f, t_fb = rd.get()
        loads = [abs(x["load"]) for x in f if x is not None] if f else [0.0]
        st["load"] = max(loads)
        if tact:
            fresh, contact, depth, cams, events = tact.get()
        else:
            fresh, events, cams = True, [], []
            depth = 0.0
            contact = f is not None and all(x is not None for x in f) and \
                max(abs(f[i]["load"]) for i in range(N) if moving[i]) > contact_load
        st.update(contact=contact, depth=depth, cams=cams)
        for cam, e in events:
            print(f"[TACTILE] {cam}: {e}")
            if e == "slip_start" and st["mode"] == "SQUEEZE":
                st["depth_ref"] = min(st["depth_max"] * 0.9, st["depth_ref"] * 1.2)
                st["msg"] = f"slip on {cam} -> squeeze harder (ref {st['depth_ref']:.2f} mm)"

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
                    m = st["mode"]
                    if m == "APPROACH":
                        if contact:
                            st["mode"], st["msg"] = "SQUEEZE", "contact -> regulating the depth"
                            print("[SQUEEZE] contact")
                        elif u >= 1.0:
                            st["mode"], st["msg"] = "HOLD", "closed pose reached, no contact: nothing grasped"
                        else:
                            u = min(1.0, u + speed / span * dt)
                    elif m == "SQUEEZE":
                        if tact and not fresh:
                            st["msg"] = "no tactile data -> holding"
                        elif no_tact:
                            pass                                   # load fallback: hold at first contact
                        else:
                            ref = st["depth_ref"]
                            if depth > st["depth_max"]:
                                e = -1.0
                            elif not contact:
                                e = 1.0                            # lost the contact: close again
                            else:
                                e = float(np.clip((ref - depth) / max(ref, 1e-3), -1, 1))
                                if abs(e) < 0.1:
                                    e = 0.0                        # dead band +-10 %
                            u = float(np.clip(u + e * squeeze / span * dt, 0, 1))
                            if u >= 1.0 and e > 0:
                                st["msg"] = "closed pose reached while squeezing (object too small / soft?)"
                    elif m == "OPEN":
                        u = max(0.0, u - speed / span * dt)
                        if u <= 0.0 and np.all(np.abs(cmd - P_open) < 0.5):
                            st["mode"], st["msg"] = "IDLE", "open. g = grasp"
                            st["depth_ref"] = ref0
                    # target pose, then ramp the command (fastest joint <= speed deg/s)
                    tgt = np.clip(P_open + u * (P_closed - P_open), -np.array(LIM), np.array(LIM))
                    step = max(speed, squeeze) * dt
                    cmd = cmd + np.clip(tgt - cmd, -step, step)
                    link.send([math.radians(c) for c in cmd])
                    t_next = now + TICK
        st["u_cmd"] = float(np.clip(np.dot(cmd - P_open, P_closed - P_open)
                                    / np.dot(P_closed - P_open, P_closed - P_open), 0, 1)) if cmd is not None else 0.0

        cv2.imshow(WIN, draw(st))
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            stop("quit")
            break
        if k == ord("g"):
            if tact and not tact.get()[0]:
                st["msg"] = "no tactile data: start the sensor with --udp (or use --no-tactile)"
            else:
                start("APPROACH", f"closing at {speed:.0f} deg/s until contact")
        elif k == ord("o"):
            start("OPEN", "opening")
        elif k == ord("h") and sending:
            st["mode"], st["msg"] = "HOLD", "holding here (g = grasp again, o = open)"
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
