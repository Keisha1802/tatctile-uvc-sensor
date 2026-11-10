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
   Mechanical max range per finger (no finger is ever commanded past it, 1 deg margin; also caps --advance):
     ./run.sh --script=servo_grasp.py limit closed -94.5 -78.5 -87.7      (ID1 ID2 ID3, measured with servo_gui)
     ./run.sh --script=servo_grasp.py limit open   -2.9 1.7 0.6
     or move there with servo_gui and:  teach limit  /  teach limit_open

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
  BASE (default on; --no-base = off): the joints that are not fingers (or --base-ids=4,5,6) realign
                     the hand. Once >= 2 fingers hold steadily, each base joint is probed +--base-probe=1.5
                     deg and back to measure how it shifts the depth between the fingers; then the base
                     moves (<= --base-speed=1 deg/s, <= --base-max=6 deg from its pose) so the fingers
                     press evenly, while the fingers only keep the grip. Base load > --base-load=0.6 Nm
                     -> base goes back to its pose. 'o' also brings the base back.
                     OPTICAL FLOW: the base also drives each finger's gel shear (mean flow in the
                     contact, px) to zero = the object no longer pushes the fingers sideways.
                     --base-shear-w=0.01 mm per px (weight vs depth), --shear-dead=0.4 px,
                     --no-base-shear = depth only.
                     BASE TACTILE SERVO (default): continuous, linear in the tactile signals. After a
                     probe (+-1.5 deg per base joint) it knows d(tactile)/d(base); then every 50 ms the
                     base moves with a PID on dq = -(J'J+lam)^-1 J' e (--base-kp=1.5 --base-ki=0.5 --base-kd=0.15) with e = contact centre of touching fingers
                     (-> middle of the gel, --base-centre-w=0.05), their shear (-> 0), their depth
                     differences (-> equal) and, for fingers not touching yet, depth - ref (-> they
                     pull the object to them). J is refined online from every base move (Broyden).
                     --base-sweep = old trial-and-error search instead.
                     (old) SEARCH: when fewer than --need fingers (default: all) touch with their sensor and
                     the others are done closing, the base sweeps each joint both ways in
                     --search-step=1 deg steps (--search-speed=2 deg/s, --search-settle=0.6 s) up to
                     --base-max; more fingers touching -> kept at once; otherwise the position where
                     the sensors saw the most depth (getting closer) is kept if it clearly helped.
  --kp=12 --ki=2 --kd=0.3
                     PID per finger on ITS depth: closing speed [deg/s] = kp*e + ki*int(e) - kd*d(depth)/dt,
                     e = depth_ref - depth [mm]; output limited to --vmax (default 3 deg/s), anti-windup,
                     depth = median of the last 3 frames
  (default) COORDINATED grip once >= 2 fingers touch: the PID acts on the MEAN depth of the
                     touching fingers (same speed for all), plus a zero-sum balance term
                     --kbal=6 deg/s per mm (deeper finger opens a bit, shallower closes a bit).
                     A single touching finger holds still until another one touches.
  --independent      one PID per finger instead (opposing fingers can fight through the object)
  --slip-boost       raise the reference on a slip that lasts --slip-hold=0.6 s (off by default:
                     the slip detector still flickers); at most 2x the start reference
  --search=DEG/S     closing speed of the other fingers once one finger touches (default 3)
  --advance=10       deg: a finger whose sensor has felt nothing at its closed pose moves on FORWARD
                     this far (--advance-speed=3 deg/s) while its load < --advance-load=0.5 Nm
                     it PAUSES at that load and goes on once the load is --advance-hyst=0.12 lower
  CENTRE (default; --no-centre = off): ADAPTIVE - works with the fingers that touch (>= 2), and its
                     gain adapts x0.5..x3 to how far the object really moves per commanded degree.
                     Once the fingers touch, a zero-sum PID on the finger ANGLES
                     (--centre-kp=0.4 --centre-ki=0.08 --centre-kd=0.05, <= --centre-vmax=1.5 deg/s) moves the
                     object to the middle of the hand until ID1-3 have (almost) the same degree: starts at a
                     spread > 3x --centre-dead=1 deg, stops < 1.5x. A finger never closes deeper than
                     --depth-max or opens when about to lose contact; if the object does not move (fixed /
                     on the table) the depth balance takes over again. In contact, fingers may use --advance.
  OPTIMAL GRIP (--optimal [--object=name]): once all fingers hold steadily, the grip (depth-ref) goes down
                     by --opt-step=6 % every --opt-dt=1 s until the FIRST slip sign (tactile slip lasting
                     0.2 s, ML slip / contact-loss prediction with --ml, or a lost contact). Optimal grip =
                     the depth the fingers had then x --opt-margin=1.3. The grip settles there and then
                     everything STOPS adjusting: mode HOLD (fingers, base, centring, ML all still; the hard
                     load stop stays on). --opt-keep = keep adapting instead (slip while holding -> +1 step).
                     While searching / settling the base and the centring hold still. Saved per object in grasp_logs/optimal_grip.json;
                     the next grasp of that object starts there and refines it. Status line 'OPT ...'.
  TACTILE GRIP (default; --no-tactile-grip = off): the grip force follows what each gel FEELS:
                     shear growing beyond its value when the grasp settled (tangential load, e.g. lifting
                     or pulling) -> grip +--tg-shear-k=15 % per px above --tg-shear-dead=0.8 px; a finger
                     slipping -> +--tg-slip=25 % at once; <= --tg-max=1.6 x the base ref; back down slowly
                     when the load is gone. Contact near the pad EDGE is reported; the base weighs contact
                     centre (--base-centre-w=0.15) and shear (--base-shear-w=0.01), not only depth. Shear the hand makes
                     itself (closing, base moving) is ignored; a steady load slowly becomes normal (--tg-tau=10 s).
  SENSING HOLD (default; --hold-frozen = old frozen HOLD): HOLD stops the SEARCH (angles, base, centring)
                     but keeps SENSING: the tactile grip may tighten the held fingers (shear / slip / edge
                     contact, <= --hold-speed=2 deg/s, load guard on) and relax them back to the held pose;
                     a held finger without contact for --hold-lost=0.5 s -> GRASP again (it closes again,
                     the others keep the grip). Edge contact adds --tg-edge-grip=10 % grip; all fingers on the
                     same pad edge -> a hint to place the object / hand differently.
  SENSOR REBASE (default; --no-rebase = off): when all fingers are back at the open pose the gels touch
                     nothing, so whatever they still show is the slowly recovering imprint of the last grasp
                     -> the sensor baseline is reset (UDP 'reset' to 127.0.0.1:9871 = key r in the sensor
                     window), and once more 3 s later.
  SYMMETRIC STOP (default; --no-stop = off): all fingers touch AND ID1-3 within --stop-spread=3 deg of
                     each other AND ID4-6 within --stop-spread-rot=6 deg of each other -> STOP searching,
                     HOLD. No time limit. Status line 'SPREAD ID1-3 x/3 ID4-6 y/6'.
  LIVE ML (--ml=ml_data/models, models of grasp_train.py --save, run by --ml-python=~/mlenv/bin/python):
                     predicts every 0.1 s from the last 0.5 s (same features as grasp_to_ml.py):
                     contact soon (closing finger)  -> closes x--ml-slow=0.35 slower (gentle touch)
                     slip / contact loss soon       -> depth-ref x--ml-boost=1.15 (<= --ml-boost-max=1.5 x start)
                     overload soon (> --ml-thr-overload=0.7) -> depth-ref x0.9 (>= 0.7 x start)
                     3 s without a prediction -> back to the start ref. --ml-thr=0.5. --ml-show = no action.
                     Status line 'ML contact.. lose.. slip.. over..'; CSV columns ml_p_* and ml_action.
  YIELD (default; --no-yield = off): while a finger still seeks the object, the touching fingers keep
                     only a light depth (--yield-depth=0.6 x ref) on their own sensor, never push alone,
                     open when their load > --yield-load=0.35 Nm, and when the seeker is paused by its
                     load they push less than it so the object moves onto its pad (--excursion caps it)
  --overclose=DEG    a finger may close this far PAST the taught closed pose (default 0 = never:
                     the taught closed pose is the mechanical end; going past it can break a link)
  --together         one controller for all fingers (contact = any sensor, depth = deepest)
  --sensor=MODE      start the sensor program too (triple | dual | usb), same terminal; it ends
                     with this program. Its own options go after a lone "--".
  --udp=PORT         listen port (default 9870)
  --log-hz=H         terminal status lines per second (default 5): mode, then per finger
                     phase, act/sent/tgt deg, load L, depth d (* = contact), camera@fps;
                     phase / mode / message changes are printed immediately
  --load-stop=NM     HARD safety: any |load| > this for 0.15 s -> all fingers open --release-deg=5 and
                     sending stops (default 1.2 Nm, 0 = off)
  --grip-load=NM     a finger above this load never closes further and backs off at --backoff=2 deg/s
                     (default 0.8 Nm; normal sponge grasps were 0.2-0.6 Nm). Works without the sensors.
  --no-tactile       contact from each finger's servo load instead of the sensors
  --contact-load=NM  load that counts as contact with --no-tactile (default 0.6)
  --no-log           do not write host/grasp_logs/grasp_<time>.csv (default: log every loop pass)
  --object=NAME      object name written in every row and in episodes.csv (for ML datasets)
  --deadband=0.02    mm: grip (mean depth) error below this -> no squeeze correction
  --deadband-bal=0.008  mm: finger-to-finger depth difference below this -> counted as balanced
  --kbal-i=4         deg/s per mm*s: slow integral on the balance, removes the last small offsets
  --excursion=4      deg: after contact the PID may move a finger at most this far from the touch point
  --stale=1.0        s: a finger stops when its sensor sends no new frame for this long
  (default) FRAME-DROP COMPENSATION: between / without camera frames the depth of a touching
                     finger is predicted from its own motion: d = last depth + k * (closing since that
                     frame); k [mm/deg] is learned online (--stiff0=0.02 first guess). The terminal marks
                     predicted values with ~. --no-predict = use the last frame as is
Keys for datasets:  1 = success, 0 = fail, 2 = slip/drop  (labels the current episode = last g / a)
  --port=/dev/ttyUSB0 --baud=460800

Safety (same as servo_gui.py): stops sending on ERR / lost feedback (0.5 s), sustained load
above --load-stop, a joint lagging its command by > 25 deg. A finger whose sensor stops
sending (0.5 s) holds where it is. Commands are clamped 2 deg inside the sketch limits.
"""
import glob
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


def set_limit(which, vals):
    """servo_grasp.py limit closed|open v1 v2 v3 [v4 v5 v6]: store measured max angles (deg) by hand."""
    key = {"closed": "limit", "open": "limit_open"}[which]
    p = json.load(open(POSES)) if os.path.exists(POSES) else {}
    base = list(p.get(key) or p.get("closed" if which == "closed" else "open") or [0.0] * N)
    for i, v in enumerate(vals[:N]):
        base[i] = round(float(v), 2)
    p[key] = base
    tmp = POSES + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(p, fh, indent=1)
    os.replace(tmp, POSES)
    print(f"[LIMIT] {which} = " + "  ".join(f"ID{i + 1} {d:+.1f}" for i, d in enumerate(base)) + f"  -> {POSES}")


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
                                     "fresh", "new_frame", "sensor_age_s", "bal_degps", "grip_mean_depth_mm", "depth_hat_mm", "stiff_mm_per_deg", "predicted")]
            cols += [p + "s_" + k for k in LABELS]
        cols += ["ml_p_contact", "ml_p_lose", "ml_p_slip", "ml_p_overload", "ml_action"]
        cols += ["message", "events"]
        self.cols = cols
        self.ml = {}                       # set by the live ML: probabilities + action of this row
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
                  "" if raw is None else f"{now - raw['rx']:.3f}", f"{fg.get('bal', 0.0):.3f}",
                  f"{fg.get('grip_mean', 0.0):.4f}", f"{fg['d_hat']:.4f}", f"{fg['k_est']:.4f}", int(fg["predicted"])]
            m = raw["msg"] if raw is not None else {}
            r += ["" if k not in m else f"{float(m[k]):.5g}" for k in self.labels]
            if ep is not None and fg["contact"]:
                ep["contact"].add(fg["id"])
        P = self.ml.get("p", {})
        r += ["" if k not in P else f"{P[k]:.3f}" for k in
              ("y_contact_soon", "y_lose_contact_soon", "y_slip_soon", "y_overload_soon")]
        r += [self.ml.get("action", "")]
        msg = st["msg"] if st["msg"] != self.last_msg else ""
        self.last_msg = st["msg"]
        r += [msg, " | ".join(notes)]
        self.w.writerow(r)
        self.last_row = dict(zip(self.cols, r))
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


class MLProc:
    """grasp_ml_live.py in the ML Python (own numpy / scikit-learn), fed with the log rows."""

    def __init__(self, py, models):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        import shutil
        py = py if os.path.exists(py) else (shutil.which(py) or py)
        if not os.path.exists(py):
            raise RuntimeError(f"ML python {py} not found (python3 -m venv ~/mlenv; ~/mlenv/bin/pip install "
                               f"numpy pandas scikit-learn joblib) or set --ml-python=")
        self.p = subprocess.Popen([py, "-u", os.path.join(here, "grasp_ml_live.py"), f"--models={models}"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.p_latest, self.t_latest, self.ready, self.err = {}, 0.0, None, None
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if "ready" in m:
                self.ready = m["ready"]
            elif "error" in m:
                self.err = m["error"]
            elif "p" in m:
                self.p_latest, self.t_latest = m["p"], time.monotonic()

    def send(self, t, row):
        if self.p.poll() is not None:
            return
        try:
            self.p.stdin.write(json.dumps(dict(t=t, row=row)) + "\n")
        except (BrokenPipeError, OSError):
            pass

    def latest(self, max_age=0.5):
        return self.p_latest if time.monotonic() - self.t_latest <= max_age else {}

    def stop(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except Exception:
            pass


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
    if args and args[0] == "limit":                      # no serial port needed
        if len(args) < 3 or args[1] not in ("closed", "open"):
            sys.exit("usage: servo_grasp.py limit closed|open DEG_ID1 DEG_ID2 DEG_ID3 ...")
        set_limit(args[1], args[2:])
        return
    # everything after a lone "--" goes to the sensor program
    sensor_args = []
    if "--" in args:
        i = args.index("--")
        args, sensor_args = args[:i], args[i + 1:]
    sensor_mode = opt(args, "sensor", None, str)       # triple | dual | usb | none
    notes = []

    t_prog = time.monotonic()

    def note(msg):
        """print (with the time since start) + keep for the 'events' column of the log."""
        print(f"[{time.monotonic() - t_prog:7.2f}s] {msg}")
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
        if len(args) < 2 or args[1] not in ("open", "closed", "limit", "limit_open"):
            sys.exit("usage: servo_grasp.py teach open|closed|limit|limit_open")
        teach(args[1], rd)
        return

    P_open, P_closed = load_poses()
    if P_open is None or P_closed is None:
        sys.exit(f"[ERROR] teach both poses first (missing in {POSES}):\n"
                 f"  ./run.sh --script=servo_grasp.py teach open\n  ./run.sh --script=servo_grasp.py teach closed")
    P_open, P_closed = np.array(P_open, float), np.array(P_closed, float)
    try:
        P_limit = json.load(open(POSES)).get("limit")
    except (OSError, ValueError):
        P_limit = None
    P_limit = np.array(P_limit, float) if P_limit else None
    lim_margin = opt(args, "limit-margin", 1.0)
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
    dead = opt(args, "deadband", 0.02)                   # mm: grip error below this -> no grip motion
    dead_bal = opt(args, "deadband-bal", 0.008)          # mm: finger-to-finger difference below this -> balanced
    kbi = opt(args, "kbal-i", 4.0)                       # deg/s per (mm*s): slow integral that removes what is left
    # CENTRE PID: once ALL fingers touch, a zero-sum PID on the finger ANGLES (ID1-3) moves the held
    # object to the middle of the hand until the finger degrees are (almost) equal: the finger that is
    # closed further opens, the others close by the same amount -> the grip itself does not change
    use_ctr = "--no-centre" not in args
    ctr_kp = opt(args, "centre-kp", 0.4)                 # deg/s per deg of angle difference
    ctr_ki = opt(args, "centre-ki", 0.08)                # deg/s per deg*s
    ctr_kd = opt(args, "centre-kd", 0.05)                # s
    ctr_dead = opt(args, "centre-dead", 1.0)             # deg: angles closer than this to the mean -> centred
    ctr_vmax = opt(args, "centre-vmax", 1.5)             # deg/s: centring speed per finger
    excursion = opt(args, "excursion", 10.0)             # deg: PID may move a finger this far from where it touched
                                                         # (the force is limited by --grip-load / --load-stop)
    grip_load = opt(args, "grip-load", 0.8)              # Nm: a finger above this never closes further, it backs off
    backoff = opt(args, "backoff", 2.0)                  # deg/s: back-off speed of a finger above --grip-load
    release_deg = opt(args, "release-deg", 5.0)          # deg: all fingers open this much on a hard load stop
    # BASE (the joints that are not fingers, e.g. ID4-6) realigns the hand so the fingers press evenly
    use_base = "--no-base" not in args
    base_probe_max = opt(args, "base-probe-max", 3.0)    # deg: largest probe (grown when a probe shows nothing)
    base_probe = opt(args, "base-probe", 0.8)            # deg: probe step per base joint (identification);
                                                         # halved automatically if a probe disturbs the grasp
    base_max = opt(args, "base-max", 6.0)                # deg: base never moves further than this from its pose
    base_speed = opt(args, "base-speed", 2.5)            # deg/s (servo)
    base_gain = opt(args, "base-kp", opt(args, "base-gain", 1.5))   # PID on the base: P (1/s) ...
    base_ki = opt(args, "base-ki", 0.5)                  # ... I (1/s^2): pushes on until the tactile error is 0
    base_kd = opt(args, "base-kd", 0.15)                 # ... D (-): damps the push (filtered)
    probe_speed = opt(args, "probe-speed", 3.0)          # deg/s while probing
    probe_wait = opt(args, "probe-wait", 0.45)           # s at each probe position (sensor + filter lag ~0.3 s)
    start_wait = opt(args, "base-start", 0.6)            # s the fingers must stand still before the base starts
    base_patience = opt(args, "base-patience", 8.0)      # s without progress (new finger / rising depth) -> base stops
    base_load = opt(args, "base-load", 0.6)              # Nm: base joint above this -> base backs off to its pose
    base_min_eff = opt(args, "base-min-eff", 0.008)      # mm/deg: weaker base joints are not used
    need = None                                          # set below: fingers that must touch (default: all)
    search_step = opt(args, "search-step", 1.0)          # deg per base search step
    search_speed = opt(args, "search-speed", 2.0)        # deg/s of the base while searching
    search_settle = opt(args, "search-settle", 0.6)      # s to wait after each step
    # optical flow in the base alignment: tangential gel shear (px, from the flow inside the contact)
    # means the object pushes the finger sideways -> the base also drives the shear to zero
    shear_w = 0.0 if "--no-base-shear" in args else opt(args, "base-shear-w", 0.01)   # mm-equivalent per px
    shear_dead = opt(args, "shear-dead", 0.4)            # px: shear below this counts as zero
    cen_w = 0.0 if "--no-base-centre" in args else opt(args, "base-centre-w", 0.15)  # mm-eq per unit of cx/cy
    sweep_mode = "--base-sweep" in args                  # old trial-and-error search instead of the servo
    stale = opt(args, "stale", 1.0)                      # s: frames missing longer than this -> that finger stops
    k0 = opt(args, "stiff0", 0.02)                       # mm per deg: first guess of how fast depth grows when closing
    no_predict = "--no-predict" in args
    obj_name = opt(args, "object", "", str)
    slip_boost = "--slip-boost" in args
    slip_hold = opt(args, "slip-hold", 0.6)
    independent = "--independent" in args               # old: one PID per finger (fingers can fight)
    kb = opt(args, "kbal", 6.0)                          # deg/s per mm: balance between the fingers
    search = opt(args, "search", 3.0)                    # deg/s, closing speed after the first contact
    over = opt(args, "overclose", 0.0)                   # deg a finger may close past the taught closed pose (0 = never)
    advance = opt(args, "advance", 10.0)                 # deg a finger may go on PAST the closed pose while its
                                                         # sensor still feels nothing (forward, toward the object)
    adv_speed = opt(args, "advance-speed", 3.0)          # deg/s while advancing
    adv_load = opt(args, "advance-load", 0.5)            # Nm: an advancing finger pauses at this load ...
    adv_hyst = opt(args, "advance-hyst", 0.12)           # ... and goes on once it is this much lower again
    yield_on = "--no-yield" not in args                  # touching fingers give way while another one still seeks
    yield_frac = opt(args, "yield-depth", 0.6)           # x depth-ref: depth a touching finger keeps while yielding
    yield_load = opt(args, "yield-load", 0.35)           # Nm: a yielding finger opens if its load is above this
    # ---- live ML (models from grasp_train.py --save, run in the ML Python as a child process) ----
    ml_dir = opt(args, "ml", None, str)
    ml_py = os.path.expanduser(opt(args, "ml-python", "~/mlenv/bin/python", str))
    ml_act = "--ml-show" not in args                     # --ml-show: only show / log the predictions
    ml_thr = opt(args, "ml-thr", 0.5)                    # probability that triggers an action
    ml_thr_over = opt(args, "ml-thr-overload", 0.7)
    ml_slow = opt(args, "ml-slow", 0.35)                 # closing speed factor when contact is predicted
    ml_boost = opt(args, "ml-boost", 1.15)               # depth-ref factor per predicted slip / contact loss
    ml_boost_max = opt(args, "ml-boost-max", 1.5)        # x the start depth-ref at most
    ml = None
    # ---- OPTIMAL GRIP search (--optimal): once all fingers hold the object, lower the grip step by step
    # until the first sign of slip (tactile slip lasting --opt-slip-hold s, ML slip / contact-loss
    # prediction, or a lost contact); optimal grip = that depth x --opt-margin, held from then on and
    # remembered per --object in grasp_logs/optimal_grip.json (next grasp of that object starts there)
    use_opt = "--optimal" in args
    opt_step = opt(args, "opt-step", 0.06)               # fraction of the start depth-ref per step
    opt_dt = opt(args, "opt-dt", 1.0)                    # s between steps (time for the grip PID to settle)
    opt_min = opt(args, "opt-min", 0.08)                 # mm: never lower than this
    opt_margin = opt(args, "opt-margin", 1.3)            # safety factor on the slip onset
    opt_settle = opt(args, "opt-settle", 1.5)            # s of stable 3/3 contact before the search starts
    opt_slip_hold = opt(args, "opt-slip-hold", 0.2)      # s a tactile slip must last to count
    opt_keep = "--opt-keep" in args
    # SYMMETRIC STOP: all fingers touch AND the finger joints ID1-3 are (almost) the same angle AND the
    # rotation joints ID4-6 are (almost) the same angle -> the grasp is done: HOLD, no more searching
    # TACTILE GRIP (default; --no-tactile-grip = off): the grip force follows what the gel FEELS, not a
    # fixed depth: tangential load (shear grows vs. the shear when the grasp settled) -> grip harder
    # (friction cone); slip -> grip harder at once; contact at the pad EDGE -> reported, and the base
    # weighs the contact centre; nothing felt for a while -> back to the base ref
    use_tg = "--no-tactile-grip" not in args
    tg_k = opt(args, "tg-shear-k", 0.15)                 # grip + 15 % per px of extra shear ...
    tg_dead = opt(args, "tg-shear-dead", 0.8)            # ... beyond this many px (noise)
    tg_slip = opt(args, "tg-slip", 0.25)                 # grip + 25 % while a finger slips
    tg_max = opt(args, "tg-max", 1.6)                    # x the base ref at most (load guard still on top)
    tg_up = opt(args, "tg-up", 0.4)                      # mm/s grip up (fast: slip is fast)
    tg_down = opt(args, "tg-down", 0.03)                 # mm/s grip down (slow)
    tg_edge = opt(args, "tg-edge", 0.6)                  # |contact centre| (-1..1) = near the pad edge
    tg_tau = opt(args, "tg-tau", 10.0)
    tg_edge_need = opt(args, "tg-edge-grip", 0.10)       # + grip for a contact at the pad edge                    # s: the shear reference follows slow / steady shear
    hold_sense = "--hold-frozen" not in args             # HOLD keeps sensing (tactile grip, slip, contact)
    hold_v = opt(args, "hold-speed", 2.0)                # deg/s a holding finger may tighten
    hold_lost = opt(args, "hold-lost", 0.5)              # s without contact on a held finger -> grasp again
    steady_on = "--no-stop" not in args
    stop_spread = opt(args, "stop-spread", 3.0)          # deg: ID1-3 this close to each other ...
    stop_spread_rot = opt(args, "stop-spread-rot", 6.0)  # deg: ... and ID4-6 this close -> STOP (HOLD)                      # keep adapting after the optimum (default: HOLD there)
    opt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_logs", "optimal_grip.json")
    st = dict(slip_since={}, c_i=0.0, c_d=0.0, c_prev=None, c_t=None, c_v=0.0, c_e=0.0, mode="IDLE", msg="g = grasp", load=0.0, depth_ref=opt(args, "depth-ref", 0.30),
              depth_max=opt(args, "depth-max", 0.80), load_stop=opt(args, "load-stop", 1.2), t_slip=0.0,
              min_fps=opt(args, "min-fps", 10.0), servo=[(None, 0.0)] * N)
    ref0 = st["depth_ref"]
    if ml_dir:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            cands = [os.path.expanduser(ml_dir)]
            if not os.path.isabs(cands[0]):                 # run.sh starts us inside host/: accept a path
                cands += [os.path.join(here, cands[0]),     # relative to host/ or to the repo root
                          os.path.join(os.path.dirname(here), cands[0]),
                          os.path.join(here, cands[0].split("host/", 1)[-1])]
            hit = [c for c in cands if glob.glob(os.path.join(c, "*.joblib"))]
            if not hit:
                sys.exit(f"[ML] no models (*.joblib) in any of: {', '.join(os.path.abspath(c) for c in cands)}\n"
                         f"     make them: cd host && ~/mlenv/bin/python grasp_train.py --data=ml_data --save")
            ml = MLProc(ml_py, os.path.abspath(hit[0]))
            note(f"[ML] models from {os.path.abspath(hit[0])}")
        except RuntimeError as e:
            sys.exit(f"[ML] {e}")
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
                    d_hat=0.0, k_est=k0, x_frame=None, d_frame=0.0, x_prev=None, d_prev=None, predicted=False,
                    at_limit=False, new_frame=False, last_rx=None) for j in moving]
    for fg in fingers:
        j = fg["j"]
        far = float(np.clip(P_closed[j] + np.sign(span[j]) * over, -LIM[j], LIM[j]))
        fg["u_max"] = max(1.0, (far - P_open[j]) / span[j])     # u > 1 = past the closed pose
        # a finger whose SENSOR has felt nothing yet may advance further (slowly, light load only)
        adv = float(np.clip(P_closed[j] + np.sign(span[j]) * advance, -LIM[j], LIM[j]))
        fg["u_adv"] = max(fg["u_max"], (adv - P_open[j]) / span[j])
        # taught mechanical LIMIT (teach limit): no finger is ever commanded past it (-1 deg margin)
        if P_limit is not None:
            lim_deg = float(P_limit[j]) - np.sign(span[j]) * lim_margin
            u_l = (lim_deg - P_open[j]) / span[j]
            fg["u_max"], fg["u_adv"] = min(fg["u_max"], u_l), min(fg["u_adv"], u_l)
            fg["limit_deg"] = lim_deg
        fg["u_lim"] = fg["u_max"]
        fg["adv_block"], fg["adv_noted"] = False, False
    try:
        P_lopen = json.load(open(POSES)).get("limit_open")
    except (OSError, ValueError):
        P_lopen = None
    for fg in fingers:
        j = fg["j"]
        if P_limit is not None and (P_closed[j] - P_limit[j]) * np.sign(span[j]) > 0:
            note(f"[LIMIT] WARNING ID {fg['id']}: taught closed pose {P_closed[j]:+.1f} is past its max "
                 f"{P_limit[j]:+.1f} -> clamped; teach closed again")
        if P_lopen is not None and (P_open[j] - P_lopen[j]) * np.sign(span[j]) < 0:
            note(f"[LIMIT] WARNING ID {fg['id']}: taught open pose {P_open[j]:+.1f} is past its max open "
                 f"{P_lopen[j]:+.1f}; teach open again")
    if P_limit is not None:
        note("[LIMIT] max finger angles (taught, -%.0f deg margin): " % lim_margin
             + "  ".join(f"ID{fg['id']} {fg['limit_deg']:+.1f}" for fg in fingers)
             + "  -> advance up to " + "  ".join(f"ID{fg['id']} {P_open[fg['j']] + fg['u_adv'] * span[fg['j']]:+.1f}"
                                               for fg in fingers))

    def seeking(fg):
        """Still trying to reach the object: closing, advancing, or advancing paused by its load."""
        return fg["phase"] in ("close", "advance") or (fg["phase"] == "miss" and fg["u"] < fg["u_adv"] - 1e-6)

    def pid_reset(fg):
        fg["pid_i"], fg["pid_d"], fg["pid_v"], fg["pid_t"], fg["pid_prev"] = 0.0, 0.0, 0.0, None, None
        fg["pid_e"] = fg["pid_p"] = fg["pid_dterm"] = 0.0
        fg["dhist"], fg["at_limit"], fg["bal_i"] = [], False, 0.0

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
        d = fg["d_hat"]                                        # filtered + drop-compensated depth
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


    def coordinate(now):
        """Grip controller for the fingers that touch (not one PID per finger, which made the
        opposing fingers push the object into each other):
          grip    = PID on the MEAN depth of the touching fingers -> same speed for all of them
          balance = kbal * (mean - own depth), sums to zero -> evens the fingers out without
                    changing the total squeeze
        One touching finger alone holds still (pushing alone only moves the object) while the
        others keep closing to it."""
        touch = [fg for fg in fingers if fg["phase"] == "squeeze" and fg["contact"] and fg["raw"] is not None
                 and now - fg["raw"]["rx"] <= stale]
        for fg in fingers:
            fg["bal"], fg["grip_mean"] = 0.0, 0.0
        if len(touch) < 2:
            st["c_v"], st["c_i"], st["c_prev"], st["c_t"], st["c_d"] = 0.0, 0.0, None, None, 0.0
            return
        dm = float(np.mean([fg["d_hat"] for fg in touch]))
        if True:                                                 # every tick: d_hat moves with the fingers
            dt_s = TICK
            st["c_t"] = now
            ref = st["depth_ref"]
            e = (st["depth_max"] - dm) * 3 if dm > st["depth_max"] else ref - dm
            if abs(e) < dead:
                e = 0.0
            if st["c_prev"] is not None:
                st["c_d"] = 0.6 * st["c_d"] + 0.4 * (dm - st["c_prev"]) / dt_s
            st["c_prev"] = dm
            v_unsat = kp * e + ki * st["c_i"] - kd * st["c_d"]
            v = float(np.clip(v_unsat, -v_max, v_max))
            if e != 0.0 and v == v_unsat:
                st["c_i"] = float(np.clip(st["c_i"] + e * dt_s, -0.1, 0.1))
            st["c_v"], st["c_e"] = (v if abs(v) >= 0.05 else 0.0), e
        for fg in touch:
            fg["grip_mean"] = dm
            diff = dm - fg["d_hat"]                              # > 0: shallower than the others -> close a bit
            if abs(diff) < dead_bal:
                diff = 0.0
            fg["bal_i"] = float(np.clip(fg.get("bal_i", 0.0) + diff * TICK, -0.05, 0.05))
            fg["bal"] = kb * diff + kbi * fg["bal_i"]
        _zero_sum(touch)
        centre(touch)

    def now_ctr():
        return time.monotonic()

    def centre(touch):
        """ADAPTIVE zero-sum PID on the angles (closing degrees) of the TOUCHING fingers (>= 2).
        Adaptive: every 0.5 s it compares how much the angle spread really shrank with how much the
        commanded centring motion should have shrunk it, and scales its own gain (x0.5 .. x3):
        the object moves less than commanded (soft gel, friction) -> push harder; it overshoots ->
        softer. The set of fingers it centres adapts to which ones touch."""
        for g in fingers:
            g["ctr"], g["ctr_e"], g["ctr_last"] = 0.0, 0.0, 0.0
        if (st.get("opt") or {}).get("phase") in ("search", "settle"):
            return                                       # optimal-grip search: nothing else moves the object
        key = tuple(sorted(g["id"] for g in touch))
        if (not use_ctr or len(touch) < 2 or any(g["act"] is None for g in touch)
                or any(seeking(g) for g in fingers if g not in touch)    # a finger still on its way: wait
                or B["phase"] == "probe" or key != st.get("ctr_set")):
            for g in fingers:
                g["ctr_i"], g["ctr_prev"], g["ctr_d"] = 0.0, None, 0.0
            st["ctr_ok"] = None
            st["ctr_best"], st["ctr_stuck"], st["ctr_blk"] = None, False, False
            st["ctr_act"], st["ctr_win"] = False, None
            st["ctr_set"] = key if (use_ctr and len(touch) >= 2) else None
            st.setdefault("ctr_g", 1.0)
            return
        c = {g["id"]: g["act"] * np.sign(span[g["j"]]) for g in touch}     # + = closed further
        cm = float(np.mean(list(c.values())))
        spread = max(c.values()) - min(c.values())
        # hysteresis: centre when the spread is > 3x --centre-dead, stop once < 1.5x; in between the
        # depth balance evens the pressure (both use the same differential finger motion)
        if spread > 3 * ctr_dead:
            st["ctr_act"] = True
        elif spread < 1.5 * ctr_dead:
            st["ctr_act"] = False
        if not st.get("ctr_act"):
            for g in fingers:
                g["ctr_i"], g["ctr_prev"], g["ctr_d"], g["ctr_e"] = 0.0, None, 0.0, 0.0
            st["ctr_best"], st["ctr_tblk"], st["ctr_tfree"], st["ctr_win"] = None, None, None, None
            if st.get("ctr_ok") is False:
                note("[CENTRE] finger angles " + " ".join(f"ID{g['id']} {g['act']:+.1f}" for g in touch)
                     + f" -> centred (spread {spread:.1f} deg)")
            st["ctr_ok"] = True
            return
        # ---- adaptation of the gain from the real effect of the last 0.5 s
        t = now_ctr()
        W = st.get("ctr_win")
        if W is None:
            st["ctr_win"] = dict(t=t, s=spread, cmd=0.0)
        else:
            W["cmd"] += sum(abs(g.get("ctr_last", 0.0)) for g in touch) * TICK
            if t - W["t"] >= 0.5:
                if W["cmd"] > 0.2:                               # commanded >= 0.2 deg of centring motion
                    eff = (W["s"] - spread) / W["cmd"]           # 1 = the spread shrank as commanded
                    g0 = st.get("ctr_g", 1.0)
                    if eff < 0.4:
                        st["ctr_g"] = min(3.0, g0 * 1.2)         # object lags (gel / friction): push harder
                    elif eff > 1.5 or spread > W["s"] + 0.3:
                        st["ctr_g"] = max(0.5, g0 * 0.8)         # overshoot / oscillation: softer
                    if abs(st["ctr_g"] - g0) > 1e-6 and t - st.get("t_ctrg", 0.0) > 3.0:
                        st["t_ctrg"] = t
                        note(f"[CENTRE] adaptive gain x{st['ctr_g']:.2f} (effect {eff:+.2f} of the commanded motion)")
                st["ctr_win"] = dict(t=t, s=spread, cmd=0.0)
        G = st.get("ctr_g", 1.0)
        for g in touch:
            e = cm - c[g["id"]]                                  # > 0: this finger is less closed -> close
            g["ctr_e"] = e
            if abs(e) < ctr_dead:
                e = 0.0
            if g.get("ctr_prev") is not None:
                g["ctr_d"] = 0.7 * g.get("ctr_d", 0.0) + 0.3 * (e - g["ctr_prev"]) / TICK
            g["ctr_prev"] = e
            if not g["at_limit"] and not g.get("guard"):
                g["ctr_i"] = float(np.clip(g.get("ctr_i", 0.0) + e * TICK, -10.0, 10.0))
            g["ctr"] = float(np.clip(G * (ctr_kp * e + ctr_ki * g["ctr_i"] + ctr_kd * g.get("ctr_d", 0.0)),
                                     -ctr_vmax, ctr_vmax))
        m = float(np.mean([g["ctr"] for g in touch]))
        for g in touch:
            g["ctr"] -= m                                        # zero-sum: moves the object, not the grip
        # SAFETY: a finger never closes while pressing deeper than --depth-max, never opens while about
        # to lose its contact. PROGRESS: the object must actually MOVE - if the angle spread does not get
        # smaller for 3 s (object fixed / resting on the table: the centring would only shift the
        # pressure between the fingers) the centring pauses until the contacts change
        lo, hi = max(0.11, 0.3 * st["depth_ref"]), st["depth_max"]
        blocked = [g["id"] for g in touch if (g["ctr"] > 0.05 and g["d_hat"] > hi) or (g["ctr"] < -0.05 and g["d_hat"] < lo)]
        if st.get("ctr_best") is None or spread < st["ctr_best"] - 0.3 or blocked:   # safety hold: not "stuck"
            st["ctr_best"], st["ctr_tbest"] = spread, t
        if blocked:                                              # blocked (also on and off) for > 4 s
            st["ctr_tblk"], st["ctr_tfree"] = st.get("ctr_tblk") or t, None
        elif st.get("ctr_tblk") is not None:
            st["ctr_tfree"] = st.get("ctr_tfree") or t
            if t - st["ctr_tfree"] > 1.5:
                st["ctr_tblk"] = None
        if (spread > 3 * ctr_dead and t - st["ctr_tbest"] > 3.0) or (st.get("ctr_tblk") is not None and t - st["ctr_tblk"] > 4.0):
            blocked = blocked or ["no progress"]
            st["ctr_stuck"] = True
        if blocked and not st.get("ctr_stuck"):
            # ADAPTIVE: only the blocked finger holds; the others go on in their (safe) direction, e.g. the
            # finger opposite closes a bit and pushes the object onto the shallow one, which then can move
            for g in touch:
                if g["id"] in blocked:
                    g["ctr"], g["ctr_i"] = 0.0, 0.95 * g.get("ctr_i", 0.0)
        if st.get("ctr_stuck"):
            for g in fingers:
                g["ctr"], g["ctr_i"] = 0.0, 0.95 * g.get("ctr_i", 0.0)
            if not st.get("ctr_blk") and t - st.get("t_ctrblk", 0.0) > 5.0:
                st["t_ctrblk"] = t
                note(f"[CENTRE] paused ({blocked}): spread {spread:.1f} deg; " + (
                     "the object does not move (fixed / on the table?)" if st.get("ctr_stuck")
                     else "a finger would press too deep / lose its contact") + (" -> depth balance instead" if st.get("ctr_stuck") else " -> that finger holds, the others go on"))
            st["ctr_blk"] = True
        for g in touch:
            g["ctr_last"] = g["ctr"]
        if not st.get("ctr_stuck"):                              # centring (or only paused for safety):
            if not blocked:
                st["ctr_blk"] = False
            for g in touch:
                g["bal"], g["bal_i"] = 0.0, 0.0                  # replaces the depth balance (both are the
                                                                 # same differential motion and would fight)
        if st.get("ctr_ok") is not False:
            note(f"[CENTRE] finger angles " + " ".join(f"ID{g['id']} {g['act']:+.1f}" for g in touch)
                 + f" (spread {spread:.1f} deg) -> moving the object to the middle")
        st["ctr_ok"] = False

    def _zero_sum(touch):
        """Balance terms of the touching fingers sum to zero (they only even out, never squeeze)."""
        if touch:
            m = float(np.mean([g["bal"] for g in touch]))
            for g in touch:
                g["bal"] -= m

    def ml_step(now):
        """Use the live predictions (only in GRASP; --ml-show = log / show only, no action):
           contact predicted for a closing finger -> close slower (gentle touch)
           slip / contact loss predicted while holding -> depth-ref x --ml-boost (<= --ml-boost-max x start)
           overload predicted -> depth-ref x 0.9 (>= 0.7 x start); 3 s without a prediction -> back to start"""
        if ml.err and not st.get("ml_err_noted"):
            st["ml_err_noted"] = True; note(f"[ML] {ml.err} -> ML off")
        if ml.ready and not st.get("ml_ready_noted"):
            st["ml_ready_noted"] = True
            note(f"[ML] models ready: {', '.join(ml.ready)}" + ("" if ml_act else "  (--ml-show: no action)"))
        P = ml.latest() if st["mode"] == "GRASP" else {}
        st["ml_p"], st["ml_slow"] = P, False
        act = ""
        if P and ml_act:
            touching = any(g["phase"] == "squeeze" and g["contact"] for g in fingers)
            if P.get("y_contact_soon", 0.0) > ml_thr and any(g["phase"] == "close" for g in fingers):
                st["ml_slow"], act = True, "slow"
            ps = max(P.get("y_slip_soon", 0.0), P.get("y_lose_contact_soon", 0.0))
            if (ps > ml_thr and touching and now - st.get("t_mlb", 0.0) > 1.0
                    and st["depth_ref"] < ml_boost_max * ref0 - 1e-6):
                st["t_mlb"] = st["t_mlact"] = now
                st["depth_ref"] = min(ml_boost_max * ref0, st["depth_max"] * 0.9, st["depth_ref"] * ml_boost)
                act = "grip+"
                note(f"[ML] slip {P.get('y_slip_soon', 0):.2f} / contact loss {P.get('y_lose_contact_soon', 0):.2f} "
                     f"predicted -> grip harder, depth-ref {st['depth_ref']:.2f} mm")
            elif (touching and P.get("y_overload_soon", 0.0) > ml_thr_over and now - st.get("t_mlo", 0.0) > 1.0
                    and st["depth_ref"] > 0.7 * ref0 + 1e-6):
                st["t_mlo"] = st["t_mlact"] = now
                st["depth_ref"] = max(0.7 * ref0, st["depth_ref"] * 0.9)
                act = "grip-"
                note(f"[ML] overload {P['y_overload_soon']:.2f} predicted -> lighter, depth-ref {st['depth_ref']:.2f} mm")
        if st["mode"] != "GRASP" and st.get("t_mlact"):
            st["depth_ref"], st["t_mlact"] = st.get("ref_base", ref0), 0.0   # ML changes never carry over
        if (st["mode"] == "GRASP" and now - st.get("t_mlact", 0.0) > 3.0 and st.get("t_mlact")
                and abs(st["depth_ref"] - st.get("ref_base", ref0)) > 1e-4):
            base = st.get("ref_base", ref0)
            st["depth_ref"] += (base - st["depth_ref"]) * min(1.0, 0.5 * TICK)  # slowly back to the base ref
        if log:
            log.ml = dict(p=P, action=act)

    rs_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    no_rebase = "--no-rebase" in args

    def sensor_reset(why):
        """Baseline reset of the tactile sensors (= key r in the sensor window) via UDP 9871."""
        if no_rebase:
            return
        try:
            rs_sock.sendto(b"reset", ("127.0.0.1", 9871))
        except OSError:
            return
        for g in fingers:
            g["dhist"], g["d_filt"] = [], 0.0
        note(f"[TACTILE] {why} -> sensor baseline reset (the gel's leftover imprint is not a touch)")

    def rebase_check(now):
        """With the fingers fully OPEN the gels touch nothing: whatever they still show is the imprint
        of the last grasp (the gel recovers slowly) -> reset the baseline, and once more 3 s later."""
        R = st.setdefault("rebase", {})
        if (st["mode"] not in ("OPEN", "IDLE") or any(g["u"] > 1e-3 for g in fingers)
                or any(g["act"] is None or abs(g["act"] - P_open[g["j"]]) > 3.0 for g in fingers)):
            R.clear()
            return
        if "t1" not in R:
            R["t1"] = now
            sensor_reset("fingers open")
        elif "t2" not in R and now - R["t1"] > 3.0:
            R["t2"] = now
            sensor_reset("fingers open 3 s (gel recovered further)")

    def tactile_grip(now):
        """Grip force from the tactile SENSING of each finger (shear, slip, contact position)."""
        T = st.setdefault("tg", dict(sh0={}, t_in={}, need=0.0, why=""))
        touch = [g for g in fingers if g["phase"] == "squeeze" and g["contact"]]
        holding = st["mode"] == "HOLD" and hold_sense
        if holding:
            touch = [g for g in fingers if g["contact"]]
        if (not use_tg or (st["mode"] != "GRASP" and not holding) or len(touch) < 2
                or (st.get("opt") or {}).get("phase") in ("search", "settle")):
            if st["mode"] not in ("GRASP", "HOLD"):
                T.update(sh0={}, t_in={}, need=0.0, why="", edge=set(), same_edge=set())
            return
        # shear the HAND makes itself (a finger still closing, base probing / moving, centring) is not a
        # load from outside: while the hand moves, the shear reference simply follows the shear
        # (the fingers' own grip motion does not count: that is the reaction to the load)
        q_now = np.array([B["off"][b] for b in base_ids]) if base_ids else np.zeros(1)
        moving = (any(seeking(g) for g in fingers) or B["phase"] == "probe"
                  or (T.get("q") is not None and np.abs(q_now - T["q"]).max() > 0.01))
        T["q"] = q_now
        dt_ = min(0.2, max(0.0, now - T.get("t_prev", now)))     # real time since the last call
        T["t_prev"] = now
        a_ref = 1.0 if moving else min(1.0, dt_ / tg_tau)       # reference: snap / slow follow
        base = st.get("ref_base", ref0)
        need, why = 0.0, ""
        for g in fingers:
            if g not in touch:
                T["t_in"].pop(g["id"], None); T["sh0"].pop(g["id"], None)
                continue
            T["t_in"].setdefault(g["id"], now)
            sx, sy = g.get("sx_f", 0.0), g.get("sy_f", 0.0)
            if g["id"] not in T["sh0"]:
                if now - T["t_in"][g["id"]] >= 0.8:      # shear of the settled grasp = its reference
                    T["sh0"][g["id"]] = (sx, sy)
                continue
            x0, y0 = T["sh0"][g["id"]]
            dsh = math.hypot(sx - x0, sy - y0)
            slipping = st["slip_since"].get(g["cam"]) is not None
            if not slipping:                             # a steady load is held -> it becomes the reference
                T["sh0"][g["id"]] = (x0 + a_ref * (sx - x0), y0 + a_ref * (sy - y0))
            n_i = tg_k * max(0.0, dsh - tg_dead)
            w_i = f"ID{g['id']} shear +{dsh:.1f} px" if n_i > 0 else ""
            t_sl = st["slip_since"].get(g["cam"])
            if t_sl is not None and now - t_sl >= 0.1 and tg_slip > n_i:
                n_i, w_i = tg_slip, f"ID{g['id']} slips"
            if n_i > need:
                need, why = n_i, w_i
            r = math.hypot(g.get("cx_f", 0.0), g.get("cy_f", 0.0))
            if r > tg_edge and tg_edge_need > n_i:       # edge contact = small area, less friction:
                n_i, w_i = tg_edge_need, f"ID{g['id']} edge contact"   # a bit more grip as a margin
                if n_i > need:
                    need, why = n_i, w_i
            if r > tg_edge and g["id"] not in T.setdefault("edge", set()):
                T["edge"].add(g["id"])                   # once per grasp
                note(f"[TACTILE] ID {g['id']} touches near its pad EDGE (centre {g.get('cx_f', 0):+.2f}, "
                     f"{g.get('cy_f', 0):+.2f}) -> the base works to move the contact to the pad centre")
        # all contacts on the SAME edge of their pads: the object sits off the pads' centre line -
        # the base cannot move it there (it barely changes the contact position): tell the user once
        cys = [g.get("cy_f", 0.0) for g in touch]
        cxs = [g.get("cx_f", 0.0) for g in touch]
        for ax, vals in (("cy", cys), ("cx", cxs)):
            if len(vals) == len(fingers) and (all(v < -tg_edge for v in vals) or all(v > tg_edge for v in vals)) \
                    and ax not in T.setdefault("same_edge", set()):
                T["same_edge"].add(ax)
                side = "top" if ax == "cy" and vals[0] < 0 else "bottom" if ax == "cy" else ("left" if vals[0] < 0 else "right")
                note(f"[TACTILE] ALL fingers touch the {side.upper()} edge of their pads (image {ax} mean "
                     f"{np.mean(vals):+.2f}): the object sits off the pads' centre -> place it / the hand so "
                     f"the contact moves toward the pad centre (the base cannot shift it that far)")
        need = min(need, tg_max - 1.0)
        target = min(st["depth_max"] * 0.9, base * (1.0 + need))
        d = target - st["depth_ref"]
        st["depth_ref"] += float(np.clip(d, -tg_down * dt_, tg_up * dt_)) if abs(d) > 1e-4 else 0.0
        if need > 0.05 and need > T.get("n_note", 0.0) + 0.08 and now - T.get("t_note", 0.0) > 0.5:
            T["t_note"], T["n_note"] = now, need
            note(f"[TACTILE-GRIP] {why} -> grip up to {target:.2f} mm (+{100 * need:.0f} %)")
        if need < 0.02:
            T["n_note"] = 0.0
        T["need"], T["why"] = need, why if need > 0.05 else ""

    def hold_check(now):
        """Entering HOLD: remember pose + depth per finger. While holding: a finger that loses its
        contact for --hold-lost s -> back to GRASP (it closes again, the others keep the grip)."""
        if st["mode"] != "HOLD" or not hold_sense:
            st["hold"] = None
            return
        Hd = st.get("hold")
        if Hd is None:
            st["hold"] = Hd = dict(u={g["id"]: g["u"] for g in fingers if g["contact"]},
                                   d={g["id"]: max(0.05, g["d_hat"]) for g in fingers if g["contact"]}, lost={})
            return
        for g in fingers:
            if g["id"] not in Hd["u"]:
                continue
            if g["contact"]:
                Hd["lost"].pop(g["id"], None)
                continue
            Hd["lost"].setdefault(g["id"], now)
            if now - Hd["lost"][g["id"]] >= hold_lost:
                st["mode"], st["msg"] = "GRASP", "contact lost while holding -> grasping again"
                for f2 in fingers:
                    f2["phase"] = "squeeze" if f2["contact"] else "close"
                note(f"[HOLD] ID {g['id']} lost its contact for {hold_lost:.1f} s -> GRASP again "
                     f"(the other fingers keep the grip, ID {g['id']} closes again)")
                st["hold"] = None
                return

    def steady_check(now):
        """STOP when the grasp is symmetric: all fingers touch, the finger joints ID1-3 are within
        --stop-spread deg of each other and the rotation joints ID4-6 within --stop-spread-rot deg
        (a few frames in a row, so one noisy reading does not stop it). No time limit."""
        if not steady_on or st["mode"] != "GRASP" or any(a is None for a, _ in st["servo"]):
            st["sym_n"] = 0
            return
        all_touch = all(g["phase"] == "squeeze" and g["contact"] for g in fingers)
        searching = (st.get("opt") or {}).get("phase") in ("wait", "search", "settle") or B["phase"] == "probe"
        ang = [a for a, _ in st["servo"]]
        flex = [ang[g["j"]] for g in fingers]
        rot = [ang[b] for b in base_ids] if base_ids else []
        sp_f = max(flex) - min(flex)
        sp_r = (max(rot) - min(rot)) if rot else 0.0
        st["sym"] = (sp_f, sp_r)
        ok = all_touch and not searching and sp_f <= stop_spread and sp_r <= stop_spread_rot
        st["sym_n"] = st.get("sym_n", 0) + 1 if ok else 0
        if st["sym_n"] >= 5:
            st["sym_n"] = 0
            st["mode"], st["msg"] = "HOLD", "symmetric grasp reached -> holding (g = grasp again, o = open)"
            note(f"[STOP] ID1-3 within {sp_f:.1f} deg (<= {stop_spread:.1f}), ID4-6 within {sp_r:.1f} deg "
                 f"(<= {stop_spread_rot:.1f}), all fingers touch -> STOP searching: HOLD "
                 + " ".join(f"ID{i + 1} {a:+.1f}" for i, a in enumerate(ang))
                 + "  (depth " + " ".join(f"{g['d_hat']:.2f}" for g in fingers) + " mm)")

    def opt_load():
        try:
            return json.load(open(opt_file))
        except (OSError, ValueError):
            return {}

    def opt_save(val, load):
        d = opt_load()
        key = obj_name or "_default"
        d[key] = dict(depth_ref_mm=round(val, 4), load_Nm=round(load, 3), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        os.makedirs(os.path.dirname(opt_file), exist_ok=True)
        tmp = opt_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, indent=1)
        os.replace(tmp, opt_file)

    def opt_signs(now, O):
        """First signs of slip, or '' if none."""
        for cam, t1 in st["slip_since"].items():
            if t1 >= O.get("t0", 0.0) and now - t1 >= opt_slip_hold:     # only slips that began in this phase
                return f"tactile slip on {cam}"
        P = st.get("ml_p") or {}
        if P.get("y_slip_soon", 0.0) > ml_thr:
            return f"ML slip {P['y_slip_soon']:.2f}"
        if P.get("y_lose_contact_soon", 0.0) > ml_thr:
            return f"ML contact loss {P['y_lose_contact_soon']:.2f}"
        lost = [g["id"] for g in fingers if g["id"] in O["held"] and not g["contact"]]
        if lost:
            return f"contact lost on ID {lost}"
        return ""

    def opt_step_fn(now):
        O = st.get("opt")
        if not use_opt or O is None or st["mode"] != "GRASP":
            return
        touching = [g["id"] for g in fingers if g["phase"] == "squeeze" and g["contact"]]
        if O["phase"] == "wait":
            d_all = [g["d_hat"] for g in fingers]
            steady = all(abs(d - min(st["depth_ref"], max(d_all))) < 0.3 * st["depth_ref"] + 0.03 for d in d_all)
            if (len(touching) == len(fingers) and not any(seeking(g) for g in fingers) and steady
                    and not st["slip_since"]):
                O["t_ok"] = O.get("t_ok") or now
                if now - O["t_ok"] >= opt_settle:
                    # start from the depth the fingers REALLY have (the load guard may keep it below the ref)
                    d_now = float(np.mean([g["d_hat"] for g in fingers]))
                    st["depth_ref"] = st["ref_base"] = max(opt_min, min(st["depth_ref"], d_now))
                    O.update(phase="search", t_step=now, held=set(touching), start=st["depth_ref"], t0=now)
                    note(f"[OPTIMAL] all fingers hold -> lowering the grip from {st['depth_ref']:.2f} mm "
                         f"in steps of {opt_step * ref0:.3f} mm every {opt_dt:.1f} s until the first slip sign")
            else:
                O["t_ok"] = None
            return
        sign = opt_signs(now, O)
        if O["phase"] == "search":
            if sign:
                # the grip the fingers REALLY had when the slip began (the ref may be ahead of them)
                onset = max(opt_min, float(np.mean([g["d_hat"] for g in fingers if g["contact"]] or [st["depth_ref"]])))
                best = min(st["depth_max"] * 0.9, max(onset * opt_margin, onset + 0.02))
                st["depth_ref"] = st["ref_base"] = best
                load = max(abs(g["load"]) for g in fingers)
                O.update(phase="hold" if opt_keep else "settle", onset=onset, best=best, t_sign=now, t0=now)
                opt_save(best, load)
                note(f"[OPTIMAL] slip sign ({sign}) at {onset:.3f} mm -> OPTIMAL GRIP {best:.3f} mm "
                     f"(x{opt_margin:.1f}), load ~{load:.2f} Nm; saved for '{obj_name or '_default'}'")
            elif now - O["t_step"] >= opt_dt:
                O["t_step"] = now
                if st["depth_ref"] - opt_step * ref0 < opt_min:
                    best = max(opt_min, st["depth_ref"]) * opt_margin
                    st["depth_ref"] = st["ref_base"] = best
                    O.update(phase="hold" if opt_keep else "settle", onset=None, best=best, t_sign=now, t0=now)
                    opt_save(best, max(abs(g["load"]) for g in fingers))
                    note(f"[OPTIMAL] no slip down to {opt_min:.2f} mm -> grip {best:.3f} mm (x{opt_margin:.1f}); saved")
                else:
                    st["depth_ref"] = st["ref_base"] = st["depth_ref"] - opt_step * ref0
        elif O["phase"] == "settle":
            # let the grip PID reach the optimal depth, then STOP adjusting: HOLD the fingers there
            d_m = float(np.mean([g["d_hat"] for g in fingers if g["contact"]] or [0.0]))
            ok = abs(d_m - O["best"]) < 0.1 * O["best"] + 0.01
            O["t_set"] = (O.get("t_set") or now) if ok else None
            if (O["t_set"] and now - O["t_set"] >= 0.5) or now - O["t0"] > 4.0:
                O["phase"] = "done"
                st["mode"], st["msg"] = "HOLD", f"optimal grip {O['best']:.3f} mm reached -> holding (g = grasp again, o = open)"
                note(f"[OPTIMAL] grip {d_m:.3f} mm (target {O['best']:.3f}) reached -> STOP adjusting, HOLD "
                     + " ".join(f"ID{g['id']} {g['act']:+.1f}" for g in fingers)
                     + f"  load {max(abs(g['load']) for g in fingers):.2f} Nm")
        elif O["phase"] == "hold" and sign and now - O["t_sign"] > 1.5:
            # slips again while held (object / load changed): one margin step up, remember it
            O["t_sign"] = now
            best = min(st["depth_max"] * 0.9, st["depth_ref"] * opt_margin)
            st["depth_ref"] = st["ref_base"] = O["best"] = best
            opt_save(best, max(abs(g["load"]) for g in fingers))
            note(f"[OPTIMAL] {sign} while holding -> grip up to {best:.3f} mm; saved")

    def coord_v(fg, now):
        """Speed of one touching finger = grip + balance, limited, and within --excursion.
        While another finger is still seeking the object (closing / advancing) the touching fingers
        YIELD: they keep only a light depth (--yield-depth x ref) on their OWN sensor, never push
        harder, and open when their load goes above --yield-load. So the object is not clamped
        rigidly between them while the seeking finger pushes it, and it can move onto that finger."""
        others = [g for g in fingers if g is not fg]
        fg["yield"] = False
        if fg["raw"] is None or now - fg["raw"]["rx"] > stale:
            v = 0.0                                              # stale -> hold
        elif yield_on and (any(seeking(g) for g in others) or fg.get("grip_mean", 0.0) == 0.0):
            fg["yield"] = True
            yref = max(0.15, yield_frac * st["depth_ref"])
            e = yref - fg["d_hat"]
            if abs(e) < dead:
                e = 0.0
            n_touch = sum(1 for g in fingers if g["phase"] == "squeeze" and g["contact"])
            v = float(np.clip(kp * e, -v_max, 0.5 if n_touch >= 2 else 0.0))   # alone: never pushes
            v += fg.get("ctr", 0.0)                              # adaptive centring of the touching fingers
            if abs(fg["load"]) > yield_load:
                v = min(v, -1.0)                                 # loaded by the others' push: give way
            blk = [abs(g["load"]) for g in others if g.get("adv_block") and g["phase"] == "miss"]
            if blk and abs(fg["load"]) > 0.8 * max(blk) / max(1, n_touch):
                v = min(v, -1.0)                                 # a seeker is blocked by the load: the touching
                                                                 # fingers together push less than it, so the
                                                                 # object can move onto its pad (--excursion caps it)
        elif fg.get("grip_mean", 0.0) == 0.0:
            v = 0.0                                              # alone (--no-yield) -> hold
        else:
            v = float(np.clip(st["c_v"] + fg["bal"] + fg.get("ctr", 0.0), -v_max, v_max))
        fg["at_limit"] = False
        if fg["contact_deg"] is not None and fg["act"] is not None:
            moved = (fg["act"] - fg["contact_deg"]) * np.sign(span[fg["j"]])
            if (moved >= excursion and v > 0) or (moved <= -excursion and v < 0):
                v, fg["at_limit"] = 0.0, True
        fg["pid_e"], fg["pid_p"], fg["pid_i"] = st["c_e"], kp * st["c_e"], st["c_i"]
        fg["pid_dterm"], fg["pid_v"] = -kd * st["c_d"], v
        return v


    base_ids = ([int(x) - 1 for x in opt(args, "base-ids", "", str).split(",") if x]
                or [j for j in range(N) if j not in moving])
    B = dict(phase="idle", off=np.zeros(N), J=None, cols=[], t_touch=None, probe=None, t_note=0.0, S=None)
    need = int(opt(args, "need", len(fingers)))

    def base_reset():
        B.update(phase="idle", J=None, cols=[], t_touch=None, probe=None, S=None, amp=base_probe, n_dist=0)

    NFEAT = 5                                            # per finger: depth, cx, cy, shear x, shear y

    def feat(fg, touching):
        """Tactile feature vector of one finger (mm-equivalent units).
        depth: filtered/compensated if touching, raw depth (also below the contact threshold) if not;
        cx, cy: contact centre on the gel (-1..1, or mm with --gel-mm), shear: mean flow in the contact."""
        if touching:
            return np.array([fg["d_hat"], cen_w * fg.get("cx_f", 0.0), cen_w * fg.get("cy_f", 0.0),
                             shear_w * fg.get("sx_f", 0.0), shear_w * fg.get("sy_f", 0.0)])
        return np.array([max(fg["d_filt"], 0.0) if fg["fresh"] else np.nan, np.nan, np.nan, np.nan, np.nan])

    def base_step(now, dt):
        if (st.get("opt") or {}).get("phase") in ("search", "settle"):
            return                                       # optimal-grip search: the base holds still
        """Base = continuous, LINEAR tactile servo (no trial and error):
          identify -> each base joint +-base_probe, measure J = d(tactile features)/d(q_base)
          servo    -> dq/dt = -gain * (J'J + lam I)^-1 J' e(tactile), every 50 ms, where e =
                      * touching fingers: contact centre -> middle of the gel, shear -> 0,
                        depth -> same as the other touching fingers
                      * fingers NOT touching yet: depth -> --depth-ref (they pull the object to them)
                      J is refined online (Broyden) from every base move, so a missing finger whose
                      sensor starts to feel the object immediately pulls harder."""
        if not use_base or not base_ids:
            return
        loads = {b: abs(st["servo"][b][1]) for b in base_ids}
        if st["mode"] == "OPEN" or any(l > base_load for l in loads.values()):
            if any(l > base_load for l in loads.values()) and now - B["t_note"] > 2.0:
                B["t_note"] = now
                note("[BASE] load " + ", ".join(f"ID{b + 1} {l:.2f}" for b, l in loads.items() if l > base_load)
                     + f" Nm > {base_load:.2f}: base back to its pose")
            for b in base_ids:                                   # back to the taught pose, slowly
                B["off"][b] -= float(np.clip(B["off"][b], -base_speed * dt, base_speed * dt))
            if st["mode"] != "OPEN":
                B["phase"] = "off"
            return
        if st["mode"] != "GRASP":
            return
        touch = [fg for fg in fingers if fg["phase"] == "squeeze" and fg["contact"]]
        done_closing = all(fg["phase"] in ("squeeze", "miss") for fg in fingers)
        if sweep_mode and B["phase"] in ("idle", "search") and done_closing and len(touch) < need:
            base_search(now, dt, touch)                          # old trial-and-error search (--base-sweep)
            return
        if not touch:
            B["t_touch"] = None
            if B["phase"] in ("probe", "track"):
                back = B["probe"]["base0"] if (B["phase"] == "probe" and B["probe"]) else B["off"]
                B["off"][:] = back
                what = "identification" if B["phase"] == "probe" else "servo"
                B["phase"], B["probe"] = "idle", None
                note(f"[BASE] contact lost during {what} -> base holds, fingers close again")
            return
        F = np.array([feat(fg, fg in touch) for fg in fingers])  # fingers x NFEAT
        if B["phase"] == "idle":
            # start once every finger is done closing (touching, at its closed pose, or held) and
            # nothing has moved for 1.5 s
            moving_now = any(fg["act"] is None or abs(fg["act"] - fg.get("act_settle", 1e9)) > 0.6 for fg in fingers)
            if moving_now or B["t_touch"] is None:
                B["t_touch"] = now
                for fg in fingers:
                    fg["act_settle"] = fg["act"]
            if now - B["t_touch"] > start_wait and done_closing:
                amp = B.get("amp", base_probe)
                nb = len(base_ids)
                if nb <= 3:                              # 4 runs identify up to 3 joints at once
                    H = [[+1, +1, +1], [+1, -1, -1], [-1, +1, -1], [-1, -1, +1]]
                    pats = [h[:nb] for h in H]
                else:                                    # more joints: one joint at a time, +-
                    pats = [[(1 if i == c else 0) * sg for i in range(nb)] for c in range(nb) for sg in (+1, -1)]
                B["phase"], B["probe"] = "probe", dict(s=0, t=None, J={}, Fs=[], pats=pats, base0=B["off"].copy(),
                                                      amp=amp, held={fg["id"]: fg["d_hat"] for fg in touch},
                                                      t0=now)
                note(f"[BASE] {len(touch)}/{len(fingers)} touching -> identifying all base joints together "
                     f"(+-{amp:.1f} deg, {len(pats)} positions)")
            return
        if B["phase"] == "probe":
            # each base joint: +probe, wait, -probe, wait, back. J = (F(+) - F(-)) / (2 probe):
            # symmetric, so sensor lag / slow drift cancel out
            P = B["probe"]
            # only what the PROBE caused counts: a slip that was already going on before it started does not
            moved = any(abs(B["off"][b] - P["base0"][b]) > 0.05 for b in base_ids)
            bad = [fg["id"] for fg in fingers if fg["id"] in P["held"] and moved and
                   (not fg["contact"] or fg["d_hat"] < 0.5 * P["held"][fg["id"]] or
                    (st["slip_since"].get(fg["cam"], -1e9) >= P["t0"]
                     and now - st["slip_since"][fg["cam"]] >= slip_hold))]
            if bad:                                      # the probe disturbs the grasp: undo, smaller next time
                B["off"][:] = P["base0"]
                B["amp"] = max(0.3, 0.5 * P["amp"])
                B["n_dist"] = B.get("n_dist", 0) + 1
                if B["n_dist"] >= 3:                     # it keeps disturbing: no base identification this grasp
                    B["phase"], B["probe"] = "off", None
                    note(f"[BASE] probe disturbed finger(s) {bad} {B['n_dist']}x -> base stays where it is for this grasp")
                    return
                B["phase"], B["probe"], B["t_touch"] = "idle", None, None
                note(f"[BASE] probe disturbed finger(s) {bad} -> base back, next probe +-{B['amp']:.1f} deg")
                return
            # all base joints at once: Hadamard sign patterns (orthogonal columns, each sums to 0)
            pat, wait = P["pats"][P["s"]], probe_wait
            goal = P["base0"].copy()
            for c, b in enumerate(base_ids):
                goal[b] += pat[c] * P["amp"]
            reached = True
            for b in base_ids:
                o = B["off"][b]
                B["off"][b] = o + float(np.clip(goal[b] - o, -probe_speed * dt, probe_speed * dt))
                reached &= abs(B["off"][b] - goal[b]) < 1e-6
            if not reached:
                P["t"] = None
                return
            if P["t"] is None:
                P["t"] = now
            if now - P["t"] < wait:
                return
            P["Fs"].append(F.copy())
            P["s"], P["t"] = P["s"] + 1, None
            if P["s"] >= len(P["pats"]):
                B["off"][:] = P["base0"]                 # back (the command ramp keeps it smooth)
                Fs = np.array(P["Fs"])                   # runs x fingers x NFEAT
                S = np.array(P["pats"], dtype=float)     # runs x base joints
                seen = (np.nan_to_num(Fs[:, :, 0]) > 0.03).sum(axis=0) >= 2      # per finger
                for c, b in enumerate(base_ids):
                    # J_b = sum_k s_kb F_k / (amp * sum_k s_kb^2); NaN (not seen) stays NaN
                    Jb = np.tensordot(S[:, c], Fs, axes=(0, 0)) / (P["amp"] * float(S[:, c] @ S[:, c]))
                    for i, fg in enumerate(fingers):
                        if fg["id"] not in P["held"] and not seen[i]:
                            Jb[i, :] = 0.0                 # never really felt the object during the probe
                    P["J"][b] = Jb
                finish_probe(P)
            return
        if B["phase"] != "track" or B["J"] is None:
            return
        J = B["J"]                                               # (fingers*NFEAT) x base joints
        nf = len(fingers)
        # ---- online refinement (Broyden): every >= 0.4 deg of base motion, correct J along it
        q = np.array([B["off"][b] for b in base_ids])
        if B.get("q_prev") is None:
            B["q_prev"], B["F_prev"] = q.copy(), F.copy()
        dqm = q - B["q_prev"]
        if np.linalg.norm(dqm) >= 0.4:
            dF = (F - B["F_prev"]).reshape(-1)
            ok = np.isfinite(dF)
            if ok.any():
                pred = J @ dqm
                corr = np.zeros_like(dF)
                corr[ok] = dF[ok] - pred[ok]
                J += 0.5 * np.outer(corr, dqm) / float(dqm @ dqm)
                B["J"] = J
            B["q_prev"], B["F_prev"] = q.copy(), F.copy()
        # ---- tactile error (linear): rows and targets
        rows, err = [], []
        idx_t = [i for i, fg in enumerate(fingers) if fg in touch]
        if len(idx_t) >= 2:                                      # touching fingers: equal depth
            dm = float(np.mean(F[idx_t, 0]))
            Jt = np.array([J[i * NFEAT + 0] for i in idx_t])
            Jt = Jt - Jt.mean(axis=0, keepdims=True)
            for k_, i in enumerate(idx_t):
                e_ = F[i, 0] - dm
                rows.append(Jt[k_]); err.append(0.0 if abs(e_) < dead_bal else e_)
        for i in idx_t:                                          # contact centre -> middle, shear -> 0
            for c, dz in ((1, cen_w * 0.1), (2, cen_w * 0.1), (3, shear_w * shear_dead), (4, shear_w * shear_dead)):
                if (c <= 2 and cen_w > 0) or (c >= 3 and shear_w > 0):
                    e_ = F[i, c]
                    rows.append(J[i * NFEAT + c]); err.append(0.0 if abs(e_) < dz else e_)
        for i, fg in enumerate(fingers):                         # fingers not touching: bring the object to them
            # only once their sensor has shown an effect of the base (else this row is a constant
            # that the base cannot change and would only hide real progress)
            if fg not in touch and np.isfinite(F[i, 0]) and np.any(np.abs(J[i * NFEAT + 0]) > 1e-4):
                rows.append(J[i * NFEAT + 0]); err.append(F[i, 0] - st["depth_ref"])
        A = np.array(rows) if rows else np.zeros((0, len(base_ids)))
        e = np.array(err) if err else np.zeros(0)
        if not np.any(e):
            if B.get("pid_i") is not None:
                B["pid_i"][:] = 0.0                              # at the optimum: no I push left over
            # nothing the base could correct: fine when every finger touches; otherwise tell the
            # user after --base-patience (the missing fingers' sensors feel nothing at all)
            if len(touch) < len(fingers):
                if B.get("t_idle") is None:
                    B["t_idle"] = now
                elif now - B["t_idle"] > base_patience:
                    B["phase"] = "off"
                    miss = [fg["id"] for fg in fingers if fg not in touch]
                    note(f"[BASE] the tactile signals give the base no direction (ID {miss} feel nothing, "
                         f"the touching contact is centred) -> base stays. Re-place the object and grasp again")
            return
        B["t_idle"] = None
        # damped least squares; damping = 5 % of the strongest direction (not a fixed number:
        # tactile effects are only ~0.005-0.03 mm per deg, a fixed damping ate most of the step)
        AtA = A.T @ A
        lam = 0.05 * float(np.max(np.diag(AtA))) + 1e-9
        dq = -np.linalg.solve(AtA + lam * np.eye(A.shape[1]), A.T @ e)
        # PID that PUSHES the object to the optimal base position: dq = the base correction (deg) that
        # the tactile error asks for (contact centred, shear 0, depths even, missing fingers reached)
        #   rate = kp*dq + ki*integral(dq) + kd*d(dq)/dt
        # I keeps pushing while a small error remains (P alone stalls in the gel friction);
        # anti-windup: no integration while saturated / at --base-max; reset when the error rows change
        nb = len(base_ids)
        if B.get("pid_i") is None or len(B["pid_i"]) != nb:
            B["pid_i"], B["pid_prev"], B["pid_d"], B["pid_rows"] = np.zeros(nb), None, np.zeros(nb), None
        if B["pid_rows"] != len(rows):
            B["pid_i"][:], B["pid_prev"], B["pid_d"][:], B["pid_rows"] = 0.0, None, 0.0, len(rows)
        if B["pid_prev"] is not None and dt > 0:
            B["pid_d"] = 0.7 * B["pid_d"] + 0.3 * (dq - B["pid_prev"]) / dt
        B["pid_prev"] = dq.copy()
        rates = base_gain * dq + base_ki * B["pid_i"] + base_kd * B["pid_d"]
        for c, b in enumerate(base_ids):
            rate = float(np.clip(rates[c], -base_speed, base_speed))
            sat = rate != rates[c] or (abs(B["off"][b]) >= base_max - 0.05 and np.sign(dq[c]) == np.sign(B["off"][b]))
            if not sat:
                B["pid_i"][c] = float(np.clip(B["pid_i"][c] + dq[c] * dt, -base_max, base_max))
            B["off"][b] = float(np.clip(B["off"][b] + rate * dt, -base_max, base_max))
        B["pid_e"], B["pid_rate"] = float(np.linalg.norm(e)), rates
        n_t = len(touch)
        # progress = more fingers touching, or the tactile error getting smaller (contact centre
        # moving to the gel middle, shear / depth differences shrinking, missing fingers feeling more)
        prog = n_t * 10.0 - float(np.linalg.norm(e)) / max(1e-6, B.setdefault("e0", float(np.linalg.norm(e))))
        if B.get("t_prog") is None or prog > B.get("best_prog", -1e9) + 0.05:
            B["t_prog"], B["best_prog"] = now, prog
        at_lim = sum(abs(B["off"][b]) >= base_max - 0.05 for b in base_ids) >= max(1, len(base_ids) - 1)
        if n_t < len(fingers) and (now - B["t_prog"] > base_patience or at_lim):
            B["off"][:] = B["probe0"]
            B["phase"] = "off"
            miss = [fg["id"] for fg in fingers if fg not in touch]
            note(f"[BASE] no progress for {now - B['t_prog']:.0f} s" + (" (at the limits)" if at_lim else "")
                 + f": the sensors of ID {miss} feel nothing to follow -> base back, stopped. "
                 f"Re-place the object (it touches those fingers off the gel pad) and grasp again")
            return
        if n_t != B.get("n_last"):
            if B.get("n_last") is not None:
                note(f"[BASE] servo: {n_t}/{len(fingers)} fingers touching now")
            B["n_last"] = n_t


    def base_search(now, dt, touch):
        """Sweep search: each base joint, each direction, in --search-step steps up to --base-max.
        Score = 10 x (fingers touching with their sensor) + depth seen by every sensor (also the
        ones not yet 'in contact': a rising depth on a missing finger means 'getting closer').
        More fingers touching -> keep at once. At the end of a sweep the best position is kept if
        it clearly improved the score, otherwise the base goes back and the next sweep starts."""
        def n_touch():
            return sum(1 for fg in fingers if fg["phase"] == "squeeze" and fg["contact"])

        def score():
            return 10.0 * n_touch() + sum(min(max(fg["depth"], 0.0), st["depth_ref"]) for fg in fingers
                                          if fg["fresh"])
        S = B["S"]
        if S is None:                                    # wait until the fingers stand still 1.5 s
            moving = any(fg["act"] is None or abs(fg["act"] - fg.get("act_s", 1e9)) > 0.6 for fg in fingers)
            if moving or B["t_touch"] is None:
                B["t_touch"] = now
                for fg in fingers:
                    fg["act_s"] = fg["act"]
            if now - B["t_touch"] < 1.5:
                return
            cands = [(b, sgn) for b in base_ids for sgn in (+1, -1)]
            sc = score()
            B["S"] = S = dict(base=B["off"].copy(), n0=n_touch(), s0=sc, cands=cands, k=0, n_step=1,
                              stage="go", t=None, best=(sc, B["off"].copy()))
            B["phase"] = "search"
            miss = [fg["id"] for fg in fingers if not (fg["phase"] == "squeeze" and fg["contact"])]
            note(f"[BASE] only {S['n0']}/{need} fingers touch (missing ID {miss}) -> sweeping the base")
            return
        if S["k"] >= len(S["cands"]):
            if B["phase"] != "search_fail":
                note(f"[BASE] search: no base position within +-{base_max:.0f} deg brings more fingers into "
                     f"contact -> re-place the object or open (o)")
            B["phase"] = "search_fail"
            return
        b, sgn = S["cands"][S["k"]]
        if S["stage"] == "go":
            goal = S["base"].copy()
            goal[b] = float(np.clip(goal[b] + sgn * S["n_step"] * search_step, -base_max, base_max))
        elif S["stage"] == "best":
            goal = S["best"][1]
        else:
            goal = S["base"]
        reached = True
        for j in base_ids:                               # move toward the goal
            o = B["off"][j]
            B["off"][j] = o + float(np.clip(goal[j] - o, -search_speed * dt, search_speed * dt))
            reached &= abs(B["off"][j] - goal[j]) < 1e-6
        n = n_touch()
        if S["stage"] in ("go", "best") and n > S["n0"]:  # more fingers touch: keep this position
            S["base"], S["n0"], S["k"], S["n_step"], S["t"] = B["off"].copy(), n, 0, 1, None
            S["stage"], S["s0"] = "go", score()
            S["best"] = (S["s0"], B["off"].copy())
            note(f"[BASE] search: ID{b + 1} at {B['off'][b]:+.1f} deg -> {n}/{need} fingers touch, kept")
            return
        if not reached:
            S["t"] = None
            return
        if S["t"] is None:
            S["t"] = now
        if now - S["t"] < search_settle:                 # let the fingers / sensors settle
            return
        S["t"] = None
        if S["stage"] == "go":
            sc = score()
            if sc > S["best"][0]:
                S["best"] = (sc, B["off"].copy())
            at_limit = abs(goal[b]) >= base_max - 1e-6
            if n < S["n0"] or at_limit:                  # end of this sweep
                if S["best"][0] > S["s0"] + 0.04:        # clearly closer somewhere: go there and keep it
                    S["stage"] = "best"
                else:
                    S["stage"] = "back"
            else:
                S["n_step"] += 1                         # one step further
        elif S["stage"] == "best":                       # at the best position of the sweep: keep it
            S["base"], S["s0"], S["k"], S["n_step"] = B["off"].copy(), score(), 0, 1
            S["best"] = (S["s0"], B["off"].copy())
            S["stage"] = "go"
            note(f"[BASE] search: closer at " + " ".join(f"ID{j + 1}{B['off'][j]:+.1f}" for j in base_ids)
                 + f" (score {S['s0']:.2f}) -> kept, searching on from there")
        else:                                            # back at the start of this sweep: next direction
            S["stage"], S["k"], S["n_step"] = "go", S["k"] + 1, 1


    def finish_probe(P):
        names = ["depth", "cx", "cy", "shx", "shy"]
        Jm = np.zeros((len(fingers) * NFEAT, len(base_ids)))
        parts = []
        for c, b in enumerate(base_ids):
            v = np.nan_to_num(np.array(P["J"].get(b), dtype=float))       # fingers x NFEAT
            Jm[:, c] = v.reshape(-1)
            big = [(i, k) for i in range(len(fingers)) for k in range(NFEAT) if abs(v[i, k]) > 0.002]
            parts.append(f"ID{b + 1}: " + (", ".join(f"F{fingers[i]['id']}.{names[k]} {v[i, k]:+.3f}" for i, k in big)
                                             or "no effect"))
        if np.max(np.abs(Jm)) * P["amp"] < 0.01 and P["amp"] < base_probe_max:
            # the probe did not change any tactile signal noticeably (< 0.01 mm-eq): too small for
            # this object / servo backlash -> probe again, twice as far
            B["amp"] = min(base_probe_max, 2.0 * P["amp"])
            B["phase"], B["probe"], B["t_touch"] = "idle", None, None
            note(f"[BASE] probe +-{P['amp']:.1f} deg changed no tactile signal -> next probe +-{B['amp']:.1f} deg")
            return
        B["J"], B["cols"], B["phase"], B["q_prev"], B["n_last"] = Jm, list(base_ids), "track", None, None
        B["pid_i"], B["pid_prev"], B["pid_d"], B["pid_rows"] = np.zeros(len(base_ids)), None, np.zeros(len(base_ids)), None
        B["t_track0"] = time.monotonic()
        B["probe0"], B["t_prog"], B["best_prog"] = P["base0"].copy(), None, -1e9
        B.pop("e0", None)
        note("[BASE] effect per deg (mm-eq): " + "  |  ".join(parts) + " -> tactile servo running")


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
                fg["u_lim"] = fg["u_max"]
                fg["adv_block"], fg["adv_noted"] = False, False
        sending = True
        st["mode"], st["msg"] = mode, msg
        if mode == "GRASP":
            st["ref_base"] = ref0
            st["opt"] = dict(phase="wait") if use_opt else None
            if use_opt:
                saved = opt_load().get(obj_name or "_default")
                if saved:
                    st["ref_base"] = float(saved["depth_ref_mm"])
                    note(f"[OPTIMAL] '{obj_name or '_default'}': last optimal grip {st['ref_base']:.3f} mm "
                         f"({saved.get('time', '')}) -> start there, search again from it")
            st["depth_ref"] = st["ref_base"]
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
            cur, events = tact.snapshot(max_age=stale + 0.2)
            for cam, e in events:
                note(f"[TACTILE] {cam}: {e}")
                if e == "slip_start":
                    st["slip_since"][cam] = now
                elif e == "slip_end":
                    st["slip_since"].pop(cam, None)
            # slip boost (off by default: the slip detector still flickers): only a slip that
            # LASTS --slip-hold seconds raises the reference, at most once per 2 s, up to 2x
            if slip_boost and st["mode"] == "GRASP":
                for cam, t1 in list(st["slip_since"].items()):
                    if now - t1 >= slip_hold and now - st["t_slip"] > 2.0 and st["depth_ref"] < 2 * ref0:
                        st["t_slip"] = now
                        st["depth_ref"] = min(st["depth_max"] * 0.9, 2 * ref0, st["depth_ref"] * 1.2)
                        st["msg"] = f"sustained slip on {cam} -> squeeze harder (ref {st['depth_ref']:.2f} mm)"
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
                if heavy_n >= 3:                                   # 0.15 s above the hard limit
                    # RELEASE first (holding a crushing pose keeps the force on the links), then stop
                    for fg in fingers:
                        jj = fg["j"]
                        cmd[jj] = float(np.clip(cmd[jj] - np.sign(span[jj]) * release_deg, -LIM[jj], LIM[jj]))
                        fg["u"] = max(0.0, fg["u"] - release_deg / abs(span[jj]))
                    link.send([math.radians(c) for c in cmd])
                    hv = [f"ID{i + 1} {abs(x['load']):.2f}" for i, x in enumerate(f) if x and abs(x["load"]) > st["load_stop"]]
                    stop(f"HARD LOAD {', '.join(hv)} Nm > {st['load_stop']:.1f} -> fingers opened {release_deg:.0f} deg, stopped")
                elif lag:
                    stop(f"ID {lag} not following (> 25 deg behind) -> stopped")
                elif now >= t_next:
                    dt = TICK
                    # filtered depth per finger (median of its last 3 NEW frames) + DROP COMPENSATION:
                    # depth between / without frames is predicted from the finger's own motion,
                    # d_hat = last measured depth + k * (closing since that frame), k learned online
                    for fg in fingers:
                        r = fg["raw"]
                        x = fg["act"] * np.sign(span[fg["j"]]) if fg["act"] is not None else None   # + = closing
                        if r is not None and r["rx"] != fg.get("f_rx"):
                            fg["f_rx"] = r["rx"]
                            fg["dhist"] = (fg["dhist"] + [fg["depth"]])[-3:]
                            fg["d_filt"] = float(np.median(fg["dhist"]))
                            m_ = r.get("msg", {})
                            for k_, kk in (("sx_f", "shear_x"), ("sy_f", "shear_y"), ("cx_f", "cx"), ("cy_f", "cy")):
                                fg[k_] = 0.6 * fg.get(k_, 0.0) + 0.4 * float(m_.get(kk, 0.0))
                            if x is not None:
                                if fg["contact"] and fg["x_prev"] is not None and abs(x - fg["x_prev"]) > 0.3:
                                    k_inst = (fg["d_filt"] - fg["d_prev"]) / (x - fg["x_prev"])
                                    fg["k_est"] = float(np.clip(0.8 * fg["k_est"] + 0.2 * k_inst, 0.002, 0.2))
                                if abs(x - (fg["x_prev"] if fg["x_prev"] is not None else 1e9)) > 0.3 or fg["x_prev"] is None:
                                    fg["x_prev"], fg["d_prev"] = x, fg["d_filt"]
                                fg["x_frame"], fg["d_frame"] = x, fg["d_filt"]
                        age = (now - r["rx"]) if r is not None else 1e9
                        if no_predict or x is None or fg["x_frame"] is None or not fg["contact"]:
                            fg["d_hat"], fg["predicted"] = fg["d_filt"], False
                        else:
                            fg["d_hat"] = max(0.0, fg["d_frame"] + fg["k_est"] * (x - fg["x_frame"]))
                            fg["predicted"] = age > 0.1                # no new frame for > 1.5 frame periods
                    if st["mode"] == "GRASP" and not independent:
                        coordinate(now)
                        if B["phase"] in ("probe", "track"):
                            for g in fingers:
                                g["bal"] = 0.0                     # the base does the balancing
                            pass                                   # the grip keeps working during the probe
                    base_step(now, dt)
                    for fg in fingers:
                        rate = 1.0 / abs(span[fg["j"]])       # u per degree
                        if st["mode"] == "OPEN":
                            fg["u"] = max(0.0, fg["u"] - speed * rate * dt)
                            fg["phase"] = "open"
                        elif st["mode"] == "HOLD":
                            # SENSING HOLD: the pose stays, but the tactile grip may tighten it (shear /
                            # slip) and relax it back to the held pose - never further open than that
                            fg["phase"] = "hold"
                            Hd = st.get("hold") or {}
                            u_h = Hd.get("u", {}).get(fg["id"])
                            if u_h is not None and fg["contact"] and hold_sense:
                                tgt_d = Hd["d"].get(fg["id"], fg["d_hat"]) * (1.0 + (st.get("tg") or {}).get("need", 0.0))
                                if fg["d_hat"] < tgt_d - 0.01 and abs(fg["load"]) < grip_load:
                                    fg["u"] = min(fg["u_adv"], fg["u"] + hold_v * rate * dt)       # grip harder
                                elif fg["u"] > u_h and fg["d_hat"] > tgt_d + 0.02:
                                    fg["u"] = max(u_h, fg["u"] - 0.5 * hold_v * rate * dt)  # back to the hold
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
                                fg["u"] = float(np.clip(fg["u"] + v * rate * dt, 0, fg["u_lim"]))
                        elif st["mode"] == "GRASP":
                            if not fg["fresh"]:
                                fg["phase"] = "nodata"         # its sensor is silent: hold
                            elif fg["phase"] in ("open", "close", "miss", "hold", "nodata", "advance") and not fg["contact"]:
                                if fg["u"] >= fg["u_max"]:
                                    # taught closed pose reached and the sensor still feels nothing:
                                    # go on forward (slowly) while the load stays light
                                    L = abs(fg["load"])
                                    if fg.get("adv_block") and L < adv_load - adv_hyst:
                                        fg["adv_block"] = False          # load came down (others yielded): go on
                                    elif not fg.get("adv_block") and L >= adv_load:
                                        fg["adv_block"] = True
                                        if now - fg.get("t_advblk", 0.0) > 3.0:
                                            fg["t_advblk"] = now
                                            note(f"[ADVANCE] ID {fg['id']} paused at {fg['act']:+.1f} deg, load {L:.2f} Nm "
                                                 f"-> touching fingers yield, goes on below {adv_load - adv_hyst:.2f} Nm")
                                    if fg["u"] < fg["u_adv"] - 1e-6 and not fg.get("adv_block"):
                                        if not fg.get("adv_noted"):
                                            fg["adv_noted"] = True
                                            note(f"[ADVANCE] ID {fg['id']} at its closed pose, sensor feels nothing "
                                                 f"-> moving on forward (max {advance:.0f} deg)")
                                        fg["phase"] = "advance"
                                        fg["u"] = min(fg["u_adv"], fg["u"] + adv_speed * rate * dt)
                                        fg["u_lim"] = max(fg["u_lim"], fg["u"])
                                    else:
                                        fg["phase"] = "miss"
                                else:
                                    fg["phase"] = "close"
                                    # once any finger touches, the others keep closing, but slower
                                    sp = search if any(g["phase"] == "squeeze" for g in fingers) else speed
                                    if st.get("ml_slow"):
                                        sp *= ml_slow            # ML predicts the touch: arrive gently
                                    fg["u"] = min(fg["u_max"], fg["u"] + sp * rate * dt)
                            else:
                                if fg["phase"] != "squeeze":
                                    note(f"[CONTACT] ID {fg['id']} ({fg['cam'] or 'load'}) at {fg['act']:+.1f} deg")
                                    pid_reset(fg)
                                    fg["contact_deg"] = fg["act"]
                                fg["phase"] = "squeeze"
                                if no_tact:
                                    continue                     # load fallback: hold at first contact
                                if not fg["contact"]:            # lost the contact: close again (to the
                                    v = search                   # closed pose at most, force guard active)
                                    if yield_on and any(g.get("adv_block") and g["phase"] == "miss" for g in fingers):
                                        v = 0.0                  # it made room for a blocked finger: wait, the
                                                                 # object is pushed back onto it
                                    pid_reset(fg)
                                    fg["contact_deg"] = None
                                    if fg["u"] >= fg["u_max"]:
                                        fg["phase"] = "miss"
                                else:
                                    v = pid_step(fg) if independent else coord_v(fg, now)
                                # a finger its SENSOR confirms on the object may use the --advance range
                                # (grip + centring); force guard / load stop still apply
                                fg["u"] = float(np.clip(fg["u"] + v * rate * dt, 0,
                                                        fg["u_adv"] if fg["contact"] else fg["u_lim"]))
                    # FORCE GUARD (independent of the tactile sensors): a finger whose servo load is
                    # above --grip-load never closes further; it backs off at --backoff deg/s and its
                    # target never runs ahead of where it actually is
                    if st["mode"] in ("GRASP", "ACTIVE", "HOLD"):
                        for fg in fingers:
                            if fg["act"] is None:
                                continue
                            jj, rate = fg["j"], 1.0 / abs(span[fg["j"]])
                            u_now = (fg["act"] - P_open[jj]) / span[jj]
                            if abs(fg["load"]) > grip_load:
                                fg["u"] = max(0.0, min(fg["u"], u_now) - backoff * rate * dt)
                                if not fg.get("guard") and now - fg.get("t_guard", 0.0) > 2.0:
                                    fg["t_guard"] = now
                                    note(f"[GUARD] ID {fg['id']} load {abs(fg['load']):.2f} Nm > {grip_load:.2f}: backing off")
                                fg["guard"] = True
                                if fg["phase"] == "advance":          # loaded while advancing: pause there
                                    fg["phase"], fg["adv_block"] = "miss", True
                                if fg["phase"] == "close":            # pressing on something the sensor did not see
                                    fg["phase"] = "squeeze"
                                    fg["contact_deg"] = fg["act"]

                            else:
                                fg["guard"] = False
                            if (not fg["contact"] and fg["phase"] in ("miss", "squeeze") and abs(fg["load"]) > 0.3
                                    and now - fg.get("t_offpad", 0.0) > 5.0):
                                fg["t_offpad"] = now
                                note(f"[WARN] ID {fg['id']} presses with {abs(fg['load']):.2f} Nm but its sensor feels "
                                     f"nothing: the object touches the finger OFF the gel pad")
                            # never command more than 3 deg ahead of the actual position in the closing direction
                            fg["u"] = min(fg["u"], u_now + 3.0 * rate)
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
                    tgt = P_open.copy() + (B["off"] if use_base else 0.0)
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
            if ml is not None:
                ml.send(now - log.t0, log.last_row)
        opt_step_fn(now)                                 # before the ML: a predicted slip is the onset
        tactile_grip(now)
        hold_check(now)
        rebase_check(now)
        steady_check(now)
        if ml is not None:
            ml_step(now)
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
                parts.append(f"ID{fg['id']} {('yld' if fg.get('yield') and fg['phase'] == 'squeeze' else fg['phase'][:4]):4s} act{a} sent{sc} tgt{tg} "
                             f"L{fg['load']:+.2f} d{fg['d_hat']:.2f}{'*' if fg['contact'] else ' '}{'~' if fg['predicted'] else ' '}"
                             f"v{fg['pid_v']:+.1f} I{fg['pid_i']:+.3f} sh{fg.get('sx_f', 0.0):+.1f},{fg.get('sy_f', 0.0):+.1f} {cam}")
            others = " ".join(f"ID{i + 1}{a:+.1f}" if a is not None else f"ID{i + 1}ERR"
                              for i, (a, _) in enumerate(st["servo"]) if i not in [fg["j"] for fg in fingers])
            mlp = st.get("ml_p") or {}
            O = st.get("opt") or {}
            sym = st.get("sym")
            tg_ = st.get("tg") or {}
            sym_txt = (f"TGRIP +{100 * tg_.get('need', 0.0):.0f}% | " if st["mode"] in ("GRASP", "HOLD") and use_tg else "")
            sym_txt += (f"SPREAD ID1-3 {sym[0]:.1f}/{stop_spread:.0f} ID4-6 {sym[1]:.1f}/{stop_spread_rot:.0f} | "
                       if sym and st["mode"] == "GRASP" and steady_on else "")
            ml_txt = sym_txt + ((f"OPT {O['phase']}" + (f" best{O['best']:.3f}" if O.get("best") else "") + " | ") if O else "")
            ml_txt += ("ML contact{:.2f} lose{:.2f} slip{:.2f} over{:.2f} | ".format(*(mlp.get(k, 0.0) for k in
                      ("y_contact_soon", "y_lose_contact_soon", "y_slip_soon", "y_overload_soon"))) if mlp else "")
            if use_base and base_ids:
                others = ((f"CTR {'ok' if st['ctr_ok'] else 'move'} " + " ".join(f"ID{g['id']}{g.get('ctr', 0.0):+.1f}" for g in fingers) + " | ")
                          if st.get("ctr_ok") is not None else "") + (f"BASE {B['phase']} " + (f"|e|{B.get('pid_e', 0.0):.3f} " if B['phase'] == "track" else "") + " ".join(f"ID{b + 1}{st['servo'][b][0]:+.1f}({B['off'][b]:+.1f})"
                                                          if st['servo'][b][0] is not None else f"ID{b + 1}ERR"
                                                          for b in base_ids))
            print(f"[{tt:7.2f}s] {st['mode']:7s} ref{st['depth_ref']:.2f} | " + ml_txt + " | ".join(parts)
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
                base_reset()
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
    if ml is not None:
        ml.stop()
    if log:
        log.row(time.monotonic(), st, fingers, None, False, notes)
        log.close()
        print(f"[LOG] saved {log.path} ({log.n} rows)")
    cv2.destroyAllWindows()
    if sensor:
        sensor.stop()


if __name__ == "__main__":
    main()
