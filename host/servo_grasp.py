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

   ONE terminal for everything (starts the sensors itself, stops them on q):
     ./run.sh --script=servo_grasp.py --sensor=triple
     ./run.sh --script=servo_grasp.py --sensor=triple --depth-ref=0.2 -- --usb=camera1 --usb2=camera2 --usb3=camera3 gpu
     (options after a lone -- go to the sensor program; sensor lines are prefixed [SENSOR])

2) Or the sensors separately, with UDP output (another terminal; no ROS needed):
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
  --kp=12 --ki=2 --kd=0.3
                     PID per finger on ITS depth: closing speed [deg/s] = kp*e + ki*int(e) - kd*d(depth)/dt,
                     e = depth_ref - depth [mm]; output limited to --vmax (default 3 deg/s), anti-windup,
                     depth = median of the last 3 frames
  --search=DEG/S     closing speed of the other fingers once one finger touches (default 3)
  --overclose=DEG    a finger may close this far PAST the taught closed pose (default 15; the
                     sketch limits still apply). Lets a finger reach a small/soft object.
  --together         one controller for all fingers (contact = any sensor, depth = deepest)
  --sensor=MODE      start the sensor program too (triple | dual | usb), same terminal; it ends
                     with this program. Its own options go after a lone "--".
  --udp=PORT         listen port (default 9870)
  --log-hz=H         terminal status lines per second (default 5): mode, then per finger
                     phase, act/sent/tgt deg, load L, depth d (* = contact), camera@fps;
                     phase / mode / message changes are printed immediately
  --load-stop=NM     safety: stop sending if any |load| > this for 0.3 s (default 3.0, 0 = off)
  --no-tactile       contact from each finger's servo load instead of the sensors
  --contact-load=NM  load that counts as contact with --no-tactile (default 0.6)
  --no-log           do not write host/grasp_logs/grasp_<time>.csv (default: log every loop pass)
  --object=NAME      object name written in every row and in episodes.csv (for ML datasets)
  --deadband=0.03    mm: depth error below this -> the finger does not move
  --excursion=6      deg: after contact the PID may move a finger at most this far from the touch point
  --stale=0.3        s: a finger stops when its sensor sends no new frame for this long
Keys for datasets:  1 = success, 0 = fail, 2 = slip/drop  (labels the current episode = last g / a)
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
                           area=float(m.get("area_frac", 0.0)), fps=float(m.get("fps", 0.0)), rx=m["rx"], msg=m)
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
        sx = lambda d: int(x0 + np.clip((d - o) / (c - o), -0.05, 1.3) / 1.3 * (x1 - x0))
        cv2.line(img, (x0, yb), (x1, yb), (80, 80, 80), 2)
        xc = sx(c)
        for xx in (x0, xc):
            cv2.line(img, (xx, yb - 7), (xx, yb + 7), (120, 120, 120), 1)
        put("open", x0 - 2, yb - 11, (120, 120, 120), 0.35)
        put("closed", xc - 20, yb - 11, (120, 120, 120), 0.35)
        put(f"PID v {fg['pid_v']:+.1f} deg/s", x0 + 260, y + 52, (230, 120, 220), 0.45)
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
    put("g grasp  a active  o open  h hold  t tare  1/0/2 label  q quit    servo: green = actual, blue = sent, ring = target"
        "   depth: green above tare, line = ref", 14, H - 14, (150, 150, 150), 0.4)
    return img


class GraspLog:
    """CSV for analysis / machine learning, one row per control-loop pass (~40 Hz):
    host/grasp_logs/grasp_YYYYmmdd_HHMMSS.csv  +  host/grasp_logs/episodes.csv (one row per grasp)

    Row = time, episode, label, object, mode, every servo (actual / load / sent), and per finger:
    controller state (phase, target, closed fraction, contact angle, PID e/P/I/D/v, at_limit)
    + the COMPLETE tactile feature vector of its sensor (s_contact, s_area_frac, s_cx, s_cy,
    s_depth_max_mm, s_volume, s_force, s_shear_x, s_shear_y, s_twist, s_slip_score, s_slip,
    s_E, s_dE, s_fps) with its age and a new_frame flag (1 = this row carries a new camera frame;
    use only those rows if you want one sample per sensor frame).
    Episode = one press of g / a (until the next one or quit); label it with keys 1 (success) /
    0 (fail) / 2 (slip/drop) while it runs or right after."""

    def __init__(self, fingers, obj=""):
        import csv
        from tactile_ros import LABELS
        self.labels, self.obj = LABELS, obj
        self.dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_logs")
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, time.strftime("grasp_%Y%m%d_%H%M%S.csv"))
        self.fh = open(self.path, "w", newline="")
        self.w = csv.writer(self.fh)
        cols = ["t_s", "unix_time", "episode", "label", "object", "mode", "sending", "depth_ref_mm", "max_load_Nm"]
        for i in range(1, N + 1):
            cols += [f"id{i}_act_deg", f"id{i}_load_Nm", f"id{i}_sent_deg"]
        for fg in fingers:
            p = f"f{fg['id']}_"
            cols += [p + c for c in ("cam", "phase", "target_deg", "closed_frac", "contact_deg",
                                     "depth_mm", "depth_filt_mm", "tare_depth_mm", "contact",
                                     "pid_e_mm", "pid_p", "pid_i", "pid_d", "pid_v_degps", "at_limit",
                                     "fresh", "new_frame", "sensor_age_s")]
            cols += [p + "s_" + k for k in LABELS]
        cols += ["message", "events"]
        self.w.writerow(cols)
        self.t0, self.n, self.t_flush, self.last_msg = time.monotonic(), 0, 0.0, None
        self.ep = None                     # current episode dict
        self.n_ep = 0
        self.ep_path = os.path.join(self.dir, "episodes.csv")
        if not os.path.exists(self.ep_path):
            with open(self.ep_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["file", "episode", "mode", "object", "label", "t_start_s", "t_end_s",
                                         "duration_s", "depth_ref_mm", "rows", "fingers_in_contact",
                                         "max_load_Nm", "events"])

    def start_episode(self, now, st):
        self.end_episode(now)
        self.n_ep += 1
        self.ep = dict(id=self.n_ep, mode=st["mode"], t0=now - self.t0, label="", rows=0, max_load=0.0,
                       contact=set(), events=0, ref=st["depth_ref"])

    def set_label(self, lab):
        if self.ep is not None:
            self.ep["label"] = lab

    def end_episode(self, now):
        import csv
        e, self.ep = self.ep, None
        if e is None:
            return
        t1 = now - self.t0
        with open(self.ep_path, "a", newline="") as fh:
            csv.writer(fh).writerow([os.path.basename(self.path), e["id"], e["mode"], self.obj, e["label"],
                                     f"{e['t0']:.3f}", f"{t1:.3f}", f"{t1 - e['t0']:.3f}", f"{e['ref']:.3f}",
                                     e["rows"], " ".join(str(i) for i in sorted(e["contact"])),
                                     f"{e['max_load']:.3f}", e["events"]])

    def row(self, now, st, fingers, cmd, sending, notes):
        ep = self.ep
        r = [f"{now - self.t0:.3f}", f"{time.time():.3f}", ep["id"] if ep else 0, ep["label"] if ep else "",
             self.obj, st["mode"], int(sending), f"{st['depth_ref']:.3f}", f"{st['load']:.3f}"]
        for i, (a, l) in enumerate(st["servo"]):
            r += ["" if a is None else f"{a:.2f}", f"{l:.3f}", "" if cmd is None else f"{cmd[i]:.2f}"]
        for fg in fingers:
            raw = fg.get("raw")
            new = raw is not None and raw["rx"] != fg["last_rx"]
            fg["last_rx"] = raw["rx"] if raw is not None else fg["last_rx"]
            r += [fg["cam"] or "", fg["phase"], "" if fg["tgt_deg"] is None else f"{fg['tgt_deg']:.2f}",
                  f"{fg['u_act']:.3f}", "" if fg["contact_deg"] is None else f"{fg['contact_deg']:.2f}",
                  f"{fg['depth']:.4f}", f"{fg['d_filt']:.4f}", f"{fg['d0']:.4f}", int(fg["contact"]),
                  f"{fg['pid_e']:.4f}", f"{fg['pid_p']:.3f}", f"{fg['pid_i']:.4f}", f"{fg['pid_dterm']:.3f}",
                  f"{fg['pid_v']:.3f}", int(fg["at_limit"]), int(fg["fresh"]), int(new),
                  "" if raw is None else f"{now - raw['rx']:.3f}"]
            m = raw["msg"] if raw is not None else {}
            r += ["" if k not in m else f"{float(m[k]):.5g}" for k in self.labels]
            if ep is not None and fg["contact"]:
                ep["contact"].add(fg["id"])
        msg = st["msg"] if st["msg"] != self.last_msg else ""
        self.last_msg = st["msg"]
        r += [msg, " | ".join(notes)]
        self.w.writerow(r)
        self.n += 1
        if ep is not None:
            ep["rows"] += 1
            ep["max_load"] = max(ep["max_load"], st["load"])
            ep["events"] += len(notes)
        if now - self.t_flush > 1.0:
            self.fh.flush()
            self.t_flush = now

    def close(self):
        self.end_episode(time.monotonic())
        self.fh.close()


class SensorProc:
    """Starts the tactile sensor program as a child process (same terminal): its output is
    printed with a [SENSOR] prefix, and it is stopped when this program ends."""

    def __init__(self, sensor_args, port):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        cmd = [sys.executable, "-u", os.path.join(here, "run_sensor_fast.py")] + sensor_args + [f"--udp=127.0.0.1:{port}"]
        print("[SENSOR] starting: " + " ".join(cmd[2:]))
        self.p = subprocess.Popen(cmd, cwd=here, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for ln in self.p.stdout:
            if not getattr(self, "stopping", False):            # no traceback spam while stopping
                print("[SENSOR] " + ln.rstrip(), flush=True)

    def alive(self):
        return self.p.poll() is None

    def stop(self):
        import signal
        if not self.alive():
            return
        self.stopping = True
        for sig, wait in ((signal.SIGTERM, 4.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(self.p.pid, sig)          # the whole group: per-camera worker processes too
            except ProcessLookupError:
                return
            try:
                self.p.wait(wait)
                break
            except Exception:
                continue
        print(f"[SENSOR] stopped (exit {self.p.returncode})")


def main():
    args = sys.argv[1:]
    # everything after a lone "--" goes to the sensor program
    sensor_args = []
    if "--" in args:
        i = args.index("--")
        args, sensor_args = args[:i], args[i + 1:]
    sensor_mode = opt(args, "sensor", None, str)       # triple | dual | usb | none
    notes = []

    def note(msg):
        """print + keep for the 'events' column of the log."""
        print(msg)
        notes.append(msg)
    port = opt(args, "port", None, str) or fw.find_port()
    baud = opt(args, "baud", 460800, int)
    link = fw.Link(port, baud)
    note(f"[INFO] {port} @ {baud}")
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
    kp, ki, kd = opt(args, "kp", 12.0), opt(args, "ki", 2.0), opt(args, "kd", 0.3)
    v_max = opt(args, "vmax", 3.0)                       # deg/s, PID output limit
    dead = opt(args, "deadband", 0.03)                   # mm: |error| below this -> no motion, no integration
    excursion = opt(args, "excursion", 6.0)              # deg: PID may move a finger this far from where it touched
    stale = opt(args, "stale", 0.3)                      # s: no new sensor frame for this long -> that finger stops
    obj_name = opt(args, "object", "", str)
    search = opt(args, "search", 3.0)                    # deg/s, closing speed after the first contact
    over = opt(args, "overclose", 15.0)                  # deg a finger may close past the taught closed pose
    st = dict(mode="IDLE", msg="g = grasp", load=0.0, depth_ref=opt(args, "depth-ref", 0.30),
              depth_max=opt(args, "depth-max", 0.80), load_stop=opt(args, "load-stop", 3.0), t_slip=0.0,
              min_fps=opt(args, "min-fps", 10.0), servo=[(None, 0.0)] * N)
    ref0 = st["depth_ref"]
    tact = None if no_tact else Tactile(opt(args, "udp", 9870, int))
    sensor = None
    if tact:
        tact.start()
        if sensor_mode and sensor_mode != "none":
            sensor = SensorProc([sensor_mode] + sensor_args, tact.sock.getsockname()[1])
    if sensor:
        cv2.namedWindow(WIN)
        cv2.moveWindow(WIN, 980, 40)                       # next to the sensor window
        note(f"[INFO] listening for tactile data on UDP {tact.sock.getsockname()[1]} (sensor with --udp)")

    # finger = one moving joint; u = 0 (open) .. 1 (closed) along its own taught range
    user_map = {}
    for item in filter(None, opt(args, "map", "", str).split(",")):
        i, c = item.split(":")
        user_map[int(i)] = c
    fingers = [dict(id=j + 1, j=j, cam=user_map.get(j + 1), u=0.0, u_cmd=0.0, phase="open",
                    contact=False, depth=0.0, d0=0.0, a0=0.0, n_on=0, fps=0.0, raw=None,
                    open_deg=float(P_open[j]), closed_deg=float(P_closed[j]), act=None, load=0.0,
                    u_act=0.0, cmd_deg=None, tgt_deg=None, fresh=False,
                    pid_i=0.0, pid_d=0.0, pid_v=0.0, pid_t=None, pid_prev=None,
                    pid_e=0.0, pid_p=0.0, pid_dterm=0.0, dhist=[], d_filt=0.0, contact_deg=None,
                    at_limit=False, new_frame=False, last_rx=None) for j in moving]
    for fg in fingers:
        j = fg["j"]
        far = float(np.clip(P_closed[j] + np.sign(span[j]) * over, -LIM[j], LIM[j]))
        fg["u_max"] = max(1.0, (far - P_open[j]) / span[j])     # u > 1 = past the closed pose

    def pid_reset(fg):
        fg["pid_i"], fg["pid_d"], fg["pid_v"], fg["pid_t"], fg["pid_prev"] = 0.0, 0.0, 0.0, None, None
        fg["pid_e"] = fg["pid_p"] = fg["pid_dterm"] = 0.0
        fg["dhist"], fg["at_limit"] = [], False

    def pid_step(fg):
        """PID on this finger's own (median-filtered) depth -> closing speed (deg/s, + = close).
        Runs once per NEW sensor frame; the speed is held until the next one, and drops to 0
        when the sensor goes quiet (--stale). Dead band, small integral, limited excursion."""
        r = fg["raw"]
        if r is None or time.monotonic() - r["rx"] > stale:
            fg["pid_v"] = 0.0
            return 0.0
        if r["rx"] == fg["pid_t"]:
            return fg["pid_v"]
        dt_s = (r["rx"] - fg["pid_t"]) if fg["pid_t"] is not None else 1.0 / 15
        dt_s = min(max(dt_s, 1e-3), 0.5)
        fg["pid_t"] = r["rx"]
        fg["dhist"] = (fg["dhist"] + [fg["depth"]])[-3:]
        d = float(np.median(fg["dhist"]))                      # median of 3 frames: no single-frame spikes
        fg["d_filt"] = d
        ref = st["depth_ref"]
        e = (st["depth_max"] - d) * 3 if d > st["depth_max"] else ref - d      # mm
        if abs(e) < dead:
            e = 0.0
        if fg["pid_prev"] is not None:                        # derivative on the measurement, filtered
            fg["pid_d"] = 0.6 * fg["pid_d"] + 0.4 * (d - fg["pid_prev"]) / dt_s
        fg["pid_prev"] = d
        p_t, d_t = kp * e, -kd * fg["pid_d"]
        v_unsat = p_t + ki * fg["pid_i"] + d_t
        v = float(np.clip(v_unsat, -v_max, v_max))
        # limited excursion from the touch point: the depth does not always follow the finger
        # (object moves / gel creep) -> never wander off; hold at the limit instead
        fg["at_limit"] = False
        if fg["contact_deg"] is not None and fg["act"] is not None:
            moved = (fg["act"] - fg["contact_deg"]) * np.sign(span[fg["j"]])   # + = closed further
            if (moved >= excursion and v > 0) or (moved <= -excursion and v < 0):
                v, fg["at_limit"] = 0.0, True
        if e != 0.0 and v == v_unsat and not fg["at_limit"]:    # anti-windup
            fg["pid_i"] = float(np.clip(fg["pid_i"] + e * dt_s, -0.1, 0.1))
        if abs(v) < 0.2:
            v = 0.0
        fg["pid_e"], fg["pid_p"], fg["pid_dterm"], fg["pid_v"] = e, p_t, d_t, v
        return v


    def tare():
        """Current sensor reading of every finger = zero (removes a static phantom offset).
        Only call it while the fingers touch nothing."""
        for fg in fingers:
            r = fg["raw"]
            fg["d0"], fg["a0"], fg["n_on"] = (r["depth"], r["area"], 0) if r else (0.0, 0.0, 0)
        note("[TARE] " + "  ".join(f"ID{fg['id']} d0={fg['d0']:.2f}mm a0={100 * fg['a0']:.1f}%" for fg in fingers))

    def assign_cams():
        """Default map: n-th moving ID -> n-th camera heard (sorted names)."""
        if tact is None or together:
            return
        names = tact.names()
        for k, fg in enumerate(fingers):
            if fg["id"] not in user_map:
                fg["cam"] = names[k] if k < len(names) else None
        note("[MAP] " + "  ".join(f"ID{fg['id']}->{fg['cam']}" for fg in fingers))

    log = None if "--no-log" in args else GraspLog(fingers, obj_name)
    if log:
        note(f"[LOG] {log.path}")
    keyname = {ord(c): c for c in "gaohtq102"}

    cmd = None
    log_dt = 1.0 / max(opt(args, "log-hz", 5.0), 0.1)
    t_log, last_phase, last_mode, last_msg = 0.0, {}, None, None
    t0 = time.monotonic()
    sending, t_start, heavy_n = False, 0.0, 0
    t_next = time.monotonic()
    cv2.namedWindow(WIN)

    def stop(why):
        nonlocal sending
        if sending:
            note(f"[STOP] {why}")
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
        note(f"[{mode}] {msg}")
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
                note(f"[TACTILE] {cam}: {e}")
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
                                if fg["phase"] != "squeeze":
                                    pid_reset(fg)
                                    fg["contact_deg"] = fg["act"]
                                fg["phase"] = "squeeze"
                                v = pid_step(fg)
                                fg["u"] = float(np.clip(fg["u"] + v * rate * dt, 0, fg["u_max"]))
                        elif st["mode"] == "GRASP":
                            if not fg["fresh"]:
                                fg["phase"] = "nodata"         # its sensor is silent: hold
                            elif fg["phase"] in ("open", "close", "miss", "hold", "nodata") and not fg["contact"]:
                                if fg["u"] >= fg["u_max"]:
                                    fg["phase"] = "miss"
                                else:
                                    fg["phase"] = "close"
                                    # once any finger touches, the others keep closing, but slower
                                    sp = search if any(g["phase"] == "squeeze" for g in fingers) else speed
                                    fg["u"] = min(fg["u_max"], fg["u"] + sp * rate * dt)
                            else:
                                if fg["phase"] != "squeeze":
                                    note(f"[CONTACT] ID {fg['id']} ({fg['cam'] or 'load'}) at {fg['act']:+.1f} deg")
                                    pid_reset(fg)
                                    fg["contact_deg"] = fg["act"]
                                fg["phase"] = "squeeze"
                                if no_tact:
                                    continue                     # load fallback: hold at first contact
                                if not fg["contact"]:            # lost the contact: search again, slowly
                                    v = search
                                    pid_reset(fg)
                                    moved = ((fg["act"] - fg["contact_deg"]) * np.sign(span[fg["j"]])
                                             if fg["contact_deg"] is not None and fg["act"] is not None else 0)
                                    if moved >= excursion:
                                        v, fg["at_limit"] = 0.0, True
                                else:
                                    v = pid_step(fg)             # PID on its own depth
                                fg["u"] = float(np.clip(fg["u"] + v * rate * dt, 0, fg["u_max"]))
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
                    step = max(speed, squeeze, v_max) * dt
                    cmd = cmd + np.clip(tgt - cmd, -step, step)
                    link.send([math.radians(c) for c in cmd])
                    t_next = now + TICK
        for fg in fingers:
            fg["cmd_deg"] = float(cmd[fg["j"]]) if (cmd is not None and sending) else None
            fg["tgt_deg"] = float(P_open[fg["j"]] + fg["u"] * span[fg["j"]]) if sending else None
            fg["u_cmd"] = (float(np.clip((cmd[fg["j"]] - P_open[fg["j"]]) / span[fg["j"]], 0, 1))
                           if cmd is not None else 0.0)

        if log:
            log.row(now, st, fingers, cmd if sending else None, sending, notes)
            notes.clear()
        if sensor and not sensor.alive() and not st.get("sensor_dead"):
            st["sensor_dead"] = True
            stop(f"sensor program ended (exit {sensor.p.returncode}) -> not sending")
        # ---- terminal log: phase / mode / message changes at once, full status every log_dt ----
        tt = now - t0
        if st["mode"] != last_mode:
            print(f"[{tt:7.2f}s] MODE {last_mode} -> {st['mode']}")
            last_mode = st["mode"]
        if st["msg"] != last_msg and st["msg"]:
            print(f"[{tt:7.2f}s] MSG  {st['msg']}")
            last_msg = st["msg"]
        for fg in fingers:
            if last_phase.get(fg["id"]) != fg["phase"]:
                print(f"[{tt:7.2f}s] ID{fg['id']} {last_phase.get(fg['id'], '-')} -> {fg['phase']}"
                      f"  act {fg['act'] if fg['act'] is None else round(fg['act'], 1)}"
                      f"  depth {fg['depth']:.2f} mm  ({fg['cam']})")
                last_phase[fg["id"]] = fg["phase"]
        if now - t_log >= log_dt:
            t_log = now
            parts = []
            for fg in fingers:
                a = "  ERR " if fg["act"] is None else f"{fg['act']:+6.1f}"
                sc = "   -  " if fg["cmd_deg"] is None else f"{fg['cmd_deg']:+6.1f}"
                tg = "   -  " if fg["tgt_deg"] is None else f"{fg['tgt_deg']:+6.1f}"
                cam = f"{fg['cam'] or '-'}" + (f"@{fg['fps']:.0f}" if fg["fresh"] else "@--")
                parts.append(f"ID{fg['id']} {fg['phase'][:4]:4s} act{a} sent{sc} tgt{tg} "
                             f"L{fg['load']:+.2f} d{fg['depth']:.2f}{'*' if fg['contact'] else ' '} "
                             f"v{fg['pid_v']:+.1f} I{fg['pid_i']:+.3f} {cam}")
            others = " ".join(f"ID{i + 1}{a:+.1f}" if a is not None else f"ID{i + 1}ERR"
                              for i, (a, _) in enumerate(st["servo"]) if i not in [fg["j"] for fg in fingers])
            print(f"[{tt:7.2f}s] {st['mode']:7s} ref{st['depth_ref']:.2f} | " + " | ".join(parts)
                  + (f" | {others}" if others else ""), flush=True)

        cv2.imshow(WIN, draw(st, fingers))
        k = cv2.waitKey(20) & 0xFF
        if k in keyname:
            notes.append(f"key {keyname[k]}")
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
                if log:
                    log.start_episode(time.monotonic(), dict(st, mode="GRASP" if k == ord("g") else "ACTIVE"))
                    note(f"[EPISODE] {log.n_ep} started" + (f" (object {obj_name})" if obj_name else "")
                         + " - label it with 1 success / 0 fail / 2 slip-drop")
                if k == ord("g"):
                    start("GRASP", f"each finger closes at {speed:.0f} deg/s until its sensor touches")
                else:
                    start("ACTIVE", "fingers stay; touch a sensor: that finger keeps the reference depth")
                if slow:
                    st["msg"] = "WARNING low fps: " + ", ".join(slow) + " -> USB bandwidth (own port per Zero)"
                    note("[WARN] " + st["msg"])
        elif k in (ord("1"), ord("0"), ord("2")) and log and log.ep is not None:
            lab = {ord("1"): "success", ord("0"): "fail", ord("2"): "slip"}[k]
            log.set_label(lab)
            note(f"[EPISODE] {log.ep['id']} labelled {lab}")
            st["msg"] = f"episode {log.ep['id']} = {lab}"
        elif k == ord("t"):
            tare(); st["msg"] = "tare: current sensor readings = zero"
        elif k == ord("o"):
            start("OPEN", "opening")
        elif k == ord("h") and sending:
            st["mode"], st["msg"] = "HOLD", "holding here (g = grasp again, o = open)"
    if log:
        log.row(time.monotonic(), st, fingers, None, False, notes)
        log.close()
        print(f"[LOG] saved {log.path} ({log.n} rows)")
    cv2.destroyAllWindows()
    if sensor:
        sensor.stop()


if __name__ == "__main__":
    main()
