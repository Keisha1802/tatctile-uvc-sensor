#!/usr/bin/env python3
"""
servo_gui.py  —  light manual control of the 6 servos, SLOW, through the lab's ESP32 sketch
===========================================================================================
Same link as servo_fw.py (460800 baud, one line 'q1,..,q6' in rad -> all six servos).
The sketch itself moves fast (speed 3400), so this program ramps: every 50 ms it sends the
commanded angles moved at most SPEED deg/s toward the slider targets.

  ./run.sh --script=servo_gui.py            (sg dialout -c "..." if not in the dialout group)

Window: one slider per servo (target, deg), 'speed deg/s', 'load stop x0.1Nm'.
Keys (click the window first):
  SPACE  start / stop sending. Start = targets and commands set to the CURRENT angles, so
         nothing moves until you drag a slider. Stop = nothing more is sent (servos hold).
  h      hold here: targets = current angles (stops any motion)
  0      targets = 0 deg for all (slow move to the calibrated zero pose)
  q/ESC  quit (stops sending first)
Safety: sending stops by itself when a servo reads ERR or the feedback stops (0.5 s), when
|load| > the load limit (e.g. a gripper squeezing the tactile sensor), or when a servo lags
its command by > 25 deg (blocked). Targets are clamped to the sketch limits minus 2 deg.
"""
import math
import sys
import threading
import time

import cv2
import numpy as np

import servo_fw as fw

N = fw.N
LIM = [l - 2 for l in fw.LIMIT_DEG]          # stay 2 deg inside the sketch clamp
TICK = 0.05                                    # 20 Hz command rate
WIN = "Servo control (SPACE send on/off, h hold, 0 zero pose, q quit)"
OFF = 130                                      # trackbar value = deg + OFF


class Reader(threading.Thread):
    def __init__(self, link):
        super().__init__(daemon=True)
        self.link, self.latest, self.t, self.n_err = link, None, 0.0, 0
        self.lock = threading.Lock()

    def run(self):
        for ln in self.link.lines(1e9):
            f = fw.parse(ln)
            if f is None:
                continue
            with self.lock:
                self.latest, self.t = f, time.monotonic()
                if any(x is None for x in f):
                    self.n_err += 1

    def get(self):
        with self.lock:
            return self.latest, self.t


def draw(state):
    W, H = 760, 74 + 46 * N + 22
    img = np.full((H, W, 3), (34, 28, 26), np.uint8)
    on = state["sending"]
    cv2.putText(img, "SENDING" if on else "NOT SENDING (SPACE to start)", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 200, 90) if on else (80, 160, 240), 2, cv2.LINE_AA)
    cv2.putText(img, f"speed {state['speed']:.0f} deg/s   load stop {state['load_lim']:.1f} Nm",
                (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if state["msg"]:
        cv2.putText(img, state["msg"], (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 190, 240), 1, cv2.LINE_AA)
    f = state["fb"]
    for i in range(N):
        y = 74 + 46 * i
        act = None if f is None or f[i] is None else math.degrees(f[i]["pos"])
        load = None if f is None or f[i] is None else f[i]["load"]
        cv2.putText(img, f"ID {i + 1}", (14, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        # angle bar: -limit .. +limit
        x0, x1, yb = 80, 520, y + 22
        cv2.line(img, (x0, yb), (x1, yb), (80, 80, 80), 2)
        sx = lambda d: int(x0 + (d + fw.LIMIT_DEG[i]) / (2 * fw.LIMIT_DEG[i]) * (x1 - x0))
        cv2.line(img, (sx(0), yb - 8), (sx(0), yb + 8), (120, 120, 120), 1)
        cv2.circle(img, (sx(state["target"][i]), yb), 7, (60, 160, 240), 2)          # target
        if on:
            cv2.circle(img, (sx(state["cmd"][i]), yb), 4, (230, 160, 60), -1)        # command
        if act is not None:
            cv2.circle(img, (sx(max(-fw.LIMIT_DEG[i], min(fw.LIMIT_DEG[i], act))), yb), 5, (120, 220, 120), -1)
        txt = (f"act {act:+6.1f}  tgt {state['target'][i]:+6.1f}" if act is not None else "ERR")
        cv2.putText(img, txt, (535, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        if load is not None:
            frac = min(1.0, abs(load) / max(state["load_lim"], 1e-3))
            col = (90, 200, 90) if frac < 0.6 else (60, 190, 240) if frac < 0.9 else (60, 60, 230)
            cv2.rectangle(img, (535, y + 26), (535 + int(150 * frac), y + 34), col, -1)
            cv2.putText(img, f"{load:+.2f}Nm", (690, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "green = actual   orange ring = target   blue dot = sent now", (14, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    return img


def main():
    args = sys.argv[1:]
    port, baud = None, 460800
    for a in args:
        if a.startswith("--port="):
            port = a.split("=", 1)[1]
        elif a.startswith("--baud="):
            baud = int(a.split("=", 1)[1])
    port = port or fw.find_port()
    print(f"[INFO] {port} @ {baud}")
    link = fw.Link(port, baud)
    if not fw.wait_ready(link):
        sys.exit("[ERROR] no feedback within 6 s (DC power on? sketch at 460800?)")
    rd = Reader(link); rd.start()
    time.sleep(0.3)

    f, _ = rd.get()
    cur = [math.degrees(x["pos"]) if x else 0.0 for x in (f or [None] * N)]
    st = dict(sending=False, speed=10.0, load_lim=1.5, msg="", fb=f,
              target=[max(-LIM[i], min(LIM[i], cur[i])) for i in range(N)], cmd=list(cur))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    for i in range(N):
        cv2.createTrackbar(f"ID{i + 1} deg+{OFF}", WIN, int(round(st["target"][i])) + OFF, 2 * OFF, lambda _: None)
    cv2.createTrackbar("speed deg/s", WIN, 10, 60, lambda _: None)
    cv2.createTrackbar("load stop x0.1Nm", WIN, 15, 40, lambda _: None)

    def set_targets(vals):
        for i in range(N):
            v = max(-LIM[i], min(LIM[i], vals[i]))
            st["target"][i] = v
            cv2.setTrackbarPos(f"ID{i + 1} deg+{OFF}", WIN, int(round(v)) + OFF)

    def stop(why):
        if st["sending"]:
            print(f"[STOP] {why}")
        st["sending"], st["msg"] = False, why

    t_next = time.monotonic()
    while True:
        f, t_fb = rd.get()
        st["fb"] = f
        now = time.monotonic()
        for i in range(N):
            st["target"][i] = max(-LIM[i], min(LIM[i], cv2.getTrackbarPos(f"ID{i + 1} deg+{OFF}", WIN) - OFF))
        st["speed"] = max(1, cv2.getTrackbarPos("speed deg/s", WIN))
        st["load_lim"] = max(1, cv2.getTrackbarPos("load stop x0.1Nm", WIN)) / 10.0

        if st["sending"]:
            if f is None or now - t_fb > 0.5:
                stop("feedback stopped -> not sending")
            elif any(x is None for x in f):
                stop("a servo reads ERR -> not sending (a command would send it to 0 rad)")
            else:
                act = [math.degrees(x["pos"]) for x in f]
                heavy = [i + 1 for i in range(N) if abs(f[i]["load"]) > st["load_lim"]]
                lag = [i + 1 for i in range(N) if abs(act[i] - st["cmd"][i]) > 25]
                if heavy:
                    set_targets(act); stop(f"load > {st['load_lim']:.1f} Nm on ID {heavy}: held where it is")
                elif lag:
                    set_targets(act); stop(f"ID {lag} not following (> 25 deg behind): held")
                elif now >= t_next:
                    step = st["speed"] * TICK
                    for i in range(N):
                        d = st["target"][i] - st["cmd"][i]
                        st["cmd"][i] += max(-step, min(step, d))
                    link.send([math.radians(c) for c in st["cmd"]])
                    t_next = now + TICK

        cv2.imshow(WIN, draw(st))
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            stop("quit"); break
        if k == ord(" "):
            if st["sending"]:
                stop("stopped by user (servos hold the last command)")
            elif f is None or any(x is None for x in f):
                st["msg"] = "cannot start: a servo reads ERR or no feedback"
            else:
                act = [math.degrees(x["pos"]) for x in f]
                out = [i + 1 for i in range(N) if abs(act[i]) > LIM[i]]
                if out:
                    st["msg"] = f"cannot start: ID {out} outside the sketch limits (move by hand / re-zero)"
                else:
                    st["cmd"] = list(act); set_targets(act)
                    st["sending"], st["msg"] = True, "sending: drag a slider; SPACE = stop"
                    print("[START] sending (targets = current angles)")
        elif k == ord("h") and f is not None and all(x is not None for x in f):
            set_targets([math.degrees(x["pos"]) for x in f]); st["msg"] = "hold: targets = current angles"
        elif k == ord("0"):
            set_targets([0.0] * N); st["msg"] = f"moving to the zero pose at {st['speed']:.0f} deg/s"
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
