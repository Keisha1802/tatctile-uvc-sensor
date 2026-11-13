#!/usr/bin/env python3
"""
run_sensor_dual.py  —  TWO CAMERAS AT ONCE: two USB Pi Zero cameras, or CSI + USB
=======================================================================================
Usually launched through run_sensor_fast.py:
  python3 run_sensor_fast.py triple --usb=camera1 --usb2=camera2 --usb3=camera3   # THREE Zeros
  python3 run_sensor_fast.py triple                          # the first three Tactile cameras found
  python3 run_sensor_fast.py dual usb2                       # camera1 + camera2 (two Pi Zero USB cameras)
  python3 run_sensor_fast.py dual --usb=camera1 --usb2=camera2   # pick both Zeros by hostname
  (with no CSI camera connected, 'dual' switches to two USB cameras automatically)
  python3 run_sensor_fast.py dual                  # CSI turbo + USB
  python3 run_sensor_fast.py dual full             # CSI full FOV + USB
  python3 run_sensor_fast.py dual nodepth          # no depth (lighter)
  python3 run_sensor_fast.py dual cam1 --usb=/dev/video8 --sensor-w=1280 --out=/mnt/ssd
Slip log and raw recording name each camera by its Zero hostname (camera1, camera2).
(or directly: python3 run_sensor_dual.py ... with the same arguments)

One window: status header, one row per camera (Speckle | Flow | Depth), and a
key footer. Each camera has its own TactileSensor in its own THREAD, so the
USB camera (~15 fps) does not hold back the CSI camera (40-100 fps).

Two kinds of recording, both using the same clock (time.monotonic) for both cameras:
  g  = slip-feature log of both cameras -> slip_log_dual.csv (camera column csi/usb)
  v  = RAW frames of both cameras       -> rec_YYYYmmdd_HHMMSS/{csi,usb}/, frames.csv, meta.json

Keys:
  r reset baseline (no contact)     d re-detect ROI        a flow arrows
  e lock CSI AE/AWB                 g record slip          k label slip/static
  v record raw frames               s snapshot             q quit (NOT Ctrl+C)
  x texture DETAIL panel: off -> fine (2nd full-res flow) -> residual -> depth high-pass (arg: detail)
  t tracking single / adaptive (paper flow chaining)   track=adaptive to start with it
  w adaptive referencing on/off (contact-aware auto re-baseline, adaptive_ref.py)
  u clean controls (deadzone/EMA 0, tavg 1, no rebaseline)   c save config
  --display-hz=N = screen refresh (default 10; processing stays at the camera rate)
  nogui = no windows, measurement only (--seconds=N)
  --dis=full = deltact_plus DIS settings (default in dual: lighter grad 12 / var 2)
  --depth-hz=N = depth rate per camera (default 7.5 = every 2nd frame; 15 or 0 = every
               frame). Flow, slip, contact, auto-reference always run at the camera rate.
  gpu = depth of both cameras on the NVIDIA GPU (CuPy, Jetson), checked against the CPU
  nopardepth = depth in the camera process itself (default: own process per camera)
  threads = old single-process mode (both cameras as threads; slower on a Pi 5)
  prof = print ms per stage every 3 s (find what limits proc fps)
  f depth refinement on/off (default OFF = calibrated raw depth, shown in mm)
ROS 2 ('ros'): contact / wrench / contact_point / slip / event / state / depth topics per
camera under /tactile/<camera>/ -> see tactile_ros.py (--gel-mm, --force-gain, --slip-thr).
Depth -> mm per camera from calib/depth_calib_<camera>.json (e.g. depth_calib_camera1.json),
otherwise the built-in synthetic values. 'clean' = start with neutral settings.
Controls window: roi_thresh per camera; arrow_scale/step, deadzone, ema, release, tavg,
depth_min, depth_open, mask margin, rebase hold/thr (shared by both cameras).
"""
import copy
import os
import sys
import threading
import time

import cv2
import numpy as np

import run_sensor_fast as rsf          # reuse cameras, profiles, display helpers
import tactile_ros
import adaptive_ref
from run_sensor_fast import dp, SlipLogger, make_slip_grid, slip_features

FOCAL_AT_1080 = 1426.0
DIST = (-0.60, -0.21, 0.0, 0.0, 0.0)
ROI_THRESH = 104

# ─── display theme (BGR) ───
BG      = (34, 28, 26)      # dark background
CARD    = (52, 45, 42)      # camera row background
TXT     = (235, 235, 235)
DIM     = (150, 150, 150)
C_CSI   = (230, 160, 60)    # blue
C_USB   = (60, 160, 240)    # orange
C_CAM3  = (170, 110, 220)   # pink/purple (third camera)
C_REC   = (60, 60, 230)     # red
C_OK    = (120, 200, 90)    # green
C_WARN  = (60, 190, 240)    # yellow
FONT    = cv2.FONT_HERSHEY_SIMPLEX
MAX_WIN_H = 1000            # initial window height cap (3 cameras on a laptop screen)
TW, TH  = 320, 240          # size of one on-screen panel
GAP     = 6


# ─────────────────────────────────────────────────────────────────────────────
class CamPipe(threading.Thread):
    """One camera = one thread: read latest frame -> optical flow + depth -> slip features."""

    def __init__(self, name, reader, cfg, cw, ch, window, slog, slog_lock,
                 slip_rows, slip_cols, dcal=None, refine_on=False, tact=None, ros=None,
                 aref=None, dworker=None, depth_hz=0.0):
        super().__init__(daemon=True, name=f"pipe-{name}")
        self.name_ = name
        self.reader = reader
        self.cfg = cfg
        self.window = window
        self.sensor = dp.TactileSensor(cfg, cw, ch)
        self.refine = rsf.DepthRefiner()       # same depth refinement as run_sensor_fast
        self.refine.on = refine_on             # default OFF = calibrated raw depth
        self.dcal = dcal                       # depth -> mm (calib/depth_calib_<camera>.json)
        self._n_reconnect = 0
        self.tact = tact                       # contact / force / shear / slip features
        self.ros = ros                         # ROS 2 publisher (None = off)
        self.aref = aref                       # contact-aware automatic re-baseline
        # depth in its own process (rsf.DepthWorker): flow of the next frame is computed
        # while the depth of this one runs on another core. Depth lags ~1 frame.
        self.dworker = dworker
        self.depth_on = bool(cfg.enable_depth)
        self._dw_depth, self._dw_tick = None, 0
        self.depth_dt = (1.0 / depth_hz) if depth_hz and depth_hz > 0 else 0.0
        self._dw_t = 0.0
        self.blocked = False                   # True while the slip log records
        self.slog, self.slog_lock = slog, slog_lock
        self.slip_rows, self.slip_cols = slip_rows, slip_cols
        self.lock = threading.Lock()
        self.running = True
        # requests from the main thread (sensor is not thread-safe -> executed here)
        self.req_reset = self.req_redetect = self.req_slip_reset = False
        self.req_thresh = None
        self.req_refine_reset = False
        self.req_margin = None
        # latest result (read by the main thread for display)
        self.latest = None          # dict(frame, roi, flow, depth, E, dE, flow_mag)
        self.proc_fps = 0.0
        self.cam_fps = 0.0
        self.prof = {}                          # 'prof': ms per stage (summed, reset by main)
        self.prof_n = 0
        # state slip
        self._grid = self._shape = None
        self._prev_D = self._prev_E = self._t_prev = None

    def _apply_requests(self):
        if self.req_margin is not None:
            self.cfg.valid_margin = self.req_margin; self.req_margin = None
            self.sensor.rebuild_mask()
        if self.req_refine_reset:
            self.req_refine_reset = False; self.refine.reset()
            if self.dworker is not None:
                self.dworker.set_refine(self.refine.on)
        if self.req_thresh is not None:
            self.cfg.roi_thresh = self.req_thresh; self.req_thresh = None
            self.sensor.redetect_roi()
        if self.req_redetect:
            self.req_redetect = False; self.sensor.redetect_roi(); self.refine.reset()
            if self.dworker is not None:
                self.dworker.cmd("redetect_roi"); self._dw_depth = None
        if self.req_reset:
            self.req_reset = False; self.sensor.reset(); self.refine.reset()
            if self.dworker is not None:
                self.dworker.cmd("reset"); self._dw_depth = None
            if self.tact: self.tact.reset()
            if self.aref: self.aref.on_reference()
        if getattr(self, "req_aref", None) is not None:  # set from another thread: applied HERE
            on_, self.req_aref = self.req_aref, None
            if self.aref:
                self.aref.enabled = on_; self.aref.on_reference()
        if self.req_slip_reset:
            self.req_slip_reset = False
            self._prev_D = self._prev_E = self._t_prev = None

    def run(self):
        """Never let one bad frame kill this camera: an exception is printed and the loop goes
        on (a dead thread = 'no data from cameraN' for the rest of the run)."""
        import traceback
        n_err, t_err = 0, 0.0
        while self.running:
            try:
                self._run_loop()
                return
            except Exception:
                n_err += 1
                if time.time() - t_err > 5.0:
                    t_err = time.time()
                    print(f"[ERROR] {self.name_}: frame processing failed ({n_err}x) - camera keeps running")
                    traceback.print_exc()
                time.sleep(0.02)

    def _run_loop(self):
        last_seq, n, t0, seq0 = None, 0, time.time(), 0
        t_new = t_warn = time.monotonic()
        while self.running:
            self._apply_requests()
            ok, frame, seq = rsf.wait_frame(self.reader, last_seq)
            now_ = time.monotonic()
            if not ok or seq == last_seq:
                if now_ - t_new > 0.7 and now_ - t_warn > 2.0:   # STALL diagnosis: camera side
                    t_warn = now_
                    print(f"[STALL] {self.name_}: no new camera frame for {now_ - t_new:.1f} s "
                          f"(USB / camera, not the processing)")
                continue
            if now_ - t_new > 0.7 and now_ - t_warn > 2.0:
                t_warn = now_
                print(f"[STALL] {self.name_}: {now_ - t_new:.1f} s between processed frames")
            t_new = now_
            last_seq = seq
            nrc = getattr(self.reader, "n_reconnect", 0)
            if nrc != self._n_reconnect:       # USB camera came back -> new ROI + baseline
                self._n_reconnect = nrc
                self.sensor.redetect_roi(); self.sensor.reset(); self.refine.reset()
                if self.aref: self.aref.on_reference()
                print(f"[INFO] {self.name_}: reconnected -> ROI re-detected, baseline reset")
            _t0, _c0 = time.perf_counter(), time.thread_time()
            if self.dworker is not None:
                self.cfg.enable_depth = False          # depth is done by the depth process
            res = self.sensor.process(rsf.apply_window(frame, self.window))
            if self.dworker is not None:
                self.cfg.enable_depth = self.depth_on
                d_ = self.dworker.poll()               # collect the last result FIRST, so the
                if d_ is not None:                     # worker is free for this frame
                    self._dw_depth = d_
                _due = time.monotonic() - self._dw_t >= 0.8 * self.depth_dt   # 0.8: frame-time jitter
                if res is not None and self.depth_on and _due and not self.dworker.busy:
                    self._dw_t = time.monotonic()
                    wm, fm = self.sensor.mask, self.sensor.undist.fold_mask
                    if fm is not None:
                        wm = fm.astype(np.float32) if wm is None else wm * fm
                    self.dworker.submit(seq, res["flow"],
                                        None if wm is None else wm.astype(np.float16))
                    self._dw_tick += 1
                    if self._dw_tick % 15 == 0:
                        self.dworker.set_cfg(self.cfg)  # depth_min / depth_open follow
            _t1, _c1 = time.perf_counter(), time.thread_time()

            out = dict(frame=frame, roi=None, flow=None, depth=None,
                       E=float("nan"), dE=0.0, flow_mag=0.0)
            if res is not None:
                flow, roi = res["flow"], res["roi"]
                if self.dworker is not None:
                    depth = self._dw_depth if self.depth_on else None   # refined in the worker
                else:
                    depth = res.get("depth") if self.cfg.enable_depth else None
                    if depth is not None and self.refine.on:
                        depth = self.refine(depth, self.cfg.depth_min)
                depth_mm = (self.dcal.to_mm(depth, self.cfg.depth_min)
                            if (depth is not None and self.dcal is not None
                                and self.dcal.gain is not None) else None)
                if self._shape != flow.shape[:2]:
                    self._grid = make_slip_grid(flow.shape[0], flow.shape[1],
                                                self.slip_rows, self.slip_cols)
                    self._shape = flow.shape[:2]
                    self._prev_D = self._prev_E = self._t_prev = None
                t_now = time.monotonic()
                dt = (t_now - self._t_prev) if self._t_prev else 0.0
                E, dE, vx, vy, mg, p90, D = slip_features(
                    flow, self._grid, self.slog.edges, self._prev_D, self._prev_E, dt)
                self._prev_D, self._prev_E, self._t_prev = D, E, t_now
                _t2 = time.perf_counter()
                fmax = float(cv2.magnitude(flow[..., 0], flow[..., 1]).max())
                if self.slog.recording:
                    cf = (float((depth > self.cfg.depth_min).mean())
                          if depth is not None else float("nan"))
                    with self.slog_lock:
                        self.slog.write(t_now, self.name_, E, dE, vx, vy, mg, p90,
                                        fmax, cf, dt)
                if self.tact is not None:
                    f_, ev_ = self.tact.update(flow, depth_mm if depth_mm is not None else depth,
                                               E, dE, self.proc_fps, t_now)
                    for e_ in ev_:
                        print(f"[TACTILE] {self.name_}: {e_}")
                    if self.ros is not None:
                        self.ros.publish(self.name_, f_, ev_, depth_mm)
                    self._aref_n = getattr(self, "_aref_n", 0) + 1
                    if self.aref is not None and (self._aref_n % 2 == 0 or f_["contact"]):
                        why = self.aref.update(flow, bool(f_["contact"]),
                                               float(roi[::4, ::4].mean()) / 255.0, t_now,
                                               blocked=self.blocked, gray=res.get("gray"),
                                               depth_max_mm=(f_["depth_max_mm"]
                                                             if depth_mm is not None else None))
                        if why:
                            self.sensor.reset(); self.refine.reset(); self.tact.reset()
                            print(f"[AREF] {self.name_}: new reference: {why}")
                if DETAIL["mode"] != "off":
                    r_, rms_ = rsf.texture_detail(self.sensor, res, res.get("depth"), DETAIL["mode"])
                    out.update(detail=r_, detail_rms=rms_, detail_mode=DETAIL["mode"])
                out.update(roi=roi, flow=flow, depth=depth, depth_mm=depth_mm,
                           E=E, dE=dE, flow_mag=fmax)
                _t3 = time.perf_counter()
                pr = self.prof
                _k = "flow" if self.dworker is not None else "flow+depth"
                pr[_k] = pr.get(_k, 0.0) + (_t1 - _t0)
                pr["(cpu)"] = pr.get("(cpu)", 0.0) + (_c1 - _c0)   # CPU time of the same step
                pr["slip"] = pr.get("slip", 0.0) + (_t2 - _t1)
                pr["tactile+aref+detail"] = pr.get("tactile+aref+detail", 0.0) + (_t3 - _t2)
                self.prof_n += 1

            n += 1
            now = time.time()
            if now - t0 >= 1.0:
                self.proc_fps = n / (now - t0)
                self.cam_fps = (seq - seq0) / (now - t0) if seq0 else 0.0
                n, t0, seq0 = 0, now, seq
            with self.lock:
                self.latest = out

    def snapshot(self):
        with self.lock:
            return self.latest


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────
def put(img, text, org, scale=0.5, color=TXT, thick=1):
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def chip(img, x, y, text, bg, fg=(255, 255, 255), scale=0.45, dot=None):
    """Coloured pill label. -> next x."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    pad = 8 + (14 if dot is not None else 0)
    w, h = tw + pad + 8, th + 12
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1, cv2.LINE_AA)
    if dot is not None:
        cv2.circle(img, (x + 11, y + h // 2), 4, dot, -1, cv2.LINE_AA)
    put(img, text, (x + pad, y + h - 7), scale, fg)
    return x + w + 6


def caption(img, text):
    cv2.rectangle(img, (0, 0), (len(text) * 8 + 12, 20), (0, 0, 0), -1)
    put(img, text, (6, 14), 0.42, TXT)


def fit(img):
    if img is None:
        return np.full((TH, TW, 3), 20, np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (TW, TH)) if img.shape[:2] != (TH, TW) else img.copy()


def quiver_on_panel(panel, flow, step, scale, thresh2=0.04):
    """Flow arrows on a panel of another size than the flow (the ROI is resized to
    TW x TH for display): grid and vectors are mapped with the same x/y scale."""
    h, w = flow.shape[:2]
    sx, sy = panel.shape[1] / float(w), panel.shape[0] / float(h)
    st = max(2, int(round(step / max(sx, sy))))          # 'step' in panel pixels
    ys = np.arange(st // 2, h, st)
    xs = np.arange(st // 2, w, st)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    f = flow[gy, gx]
    sel = np.where(f[..., 0] ** 2 + f[..., 1] ** 2 > thresh2)
    for r, c in zip(*sel):
        x, y = xs[c] * sx, ys[r] * sy
        dx, dy = f[r, c]
        p0 = (int(x), int(y))
        p1 = (int(x + dx * sx * scale), int(y + dy * sy * scale))
        cv2.arrowedLine(panel, p0, p1, (0, 0, 0), 3, cv2.LINE_AA, tipLength=0.35)    # outline
        cv2.arrowedLine(panel, p0, p1, (0, 255, 0), 1, cv2.LINE_AA, tipLength=0.35)
    return panel


def placeholder(msg, color=DIM):
    p = np.full((TH, TW, 3), 20, np.uint8)
    (tw, _), _ = cv2.getTextSize(msg, FONT, 0.5, 1)
    put(p, msg, ((TW - tw) // 2, TH // 2), 0.5, color)
    return p


ARROW = dict(step=16, scale=5)           # set from the controls window
DETAIL = dict(mode="off")                # texture detail panel: off / fine / residual / depth


def render_camera(pipe, title, color, info, draw_arrow, show_depth, slip_max):
    """One camera card: subheader + Speckle | Flow | Depth panels."""
    ncol = (3 if show_depth else 2) + (1 if DETAIL["mode"] != "off" else 0)
    W = ncol * TW + (ncol - 1) * GAP + 2 * GAP
    card = np.full((28 + TH + GAP + 26, W, 3), CARD, np.uint8)   # + tactile line below
    cv2.rectangle(card, (0, 0), (5, card.shape[0]), color, -1)
    put(card, title, (14, 20), 0.6, color, 2)
    snap = pipe.snapshot()
    tact_txt = None

    # subheader: info + fps + bar |F|max
    x = 14 + cv2.getTextSize(title, FONT, 0.6, 2)[0][0] + 12
    put(card, info, (x, 19), 0.42, DIM)
    fps_txt = f"cam {pipe.cam_fps:4.1f}  proc {pipe.proc_fps:4.1f} fps"
    (fw, _), _ = cv2.getTextSize(fps_txt, FONT, 0.45, 1)
    fmag = 0.0 if snap is None else snap["flow_mag"]
    bx, bw = W - 132, 120                      # |F|max bar on the right
    fx = bx - 78 - fw - 14                     # fps text left of the px label
    put(card, fps_txt, (fx, 19), 0.45, C_OK if pipe.cam_fps > 1 else C_WARN)
    cv2.rectangle(card, (bx, 9), (bx + bw, 19), (25, 25, 25), -1)
    frac = min(1.0, fmag / max(slip_max, 1e-6))
    cv2.rectangle(card, (bx, 9), (bx + int(bw * frac), 19),
                  C_OK if frac < 0.5 else C_WARN if frac < 0.85 else C_REC, -1)
    put(card, f"|F| {fmag:.2f}px", (bx - 78, 19), 0.4, DIM)

    # panel
    if snap is None:
        panels = [placeholder("waiting for frames...")] * ncol
    elif snap["roi"] is None:
        panels = [fit(snap["frame"]), placeholder("ROI? adjust roi_thresh, press d", C_WARN)]
        while len(panels) < ncol:
            panels.append(placeholder("-"))
        caption(panels[0], "CAMERA (no ROI)")
    else:
        roi, flow = snap["roi"], snap["flow"]
        p1 = fit(roi)
        if draw_arrow:
            quiver_on_panel(p1, flow, ARROW["step"], ARROW["scale"])
        caption(p1, "SPECKLE")
        if not np.isnan(snap["E"]):
            put(p1, f"E {snap['E']:.2f}  dE/dt {snap['dE']:+.1f}", (6, TH - 8), 0.42, TXT)
        p2 = fit(rsf.flow_to_color_fixed(flow)); caption(p2, f"FLOW (0-{rsf.FLOW_VIS_MAX:g} px)")
        if pipe.tact is not None and pipe.tact.last is not None:
            txt, slipping = pipe.tact.summary()        # line UNDER the panels (below)
            tact_txt = (txt, C_REC if slipping else C_OK) if pipe.tact.last["contact"] \
                else ("no contact", DIM)
        panels = [p1, p2]
        if show_depth:
            d = snap["depth"]
            p3 = fit(dp.depth_to_color(d)) if d is not None else placeholder("depth -")
            dmm = snap.get("depth_mm")
            if dmm is not None:
                caption(p3, f"DEPTH  max {dmm.max():.3f} mm")
            else:
                caption(p3, "DEPTH" + (f"  max {d.max():.3f}" if d is not None else ""))
            panels.append(p3)
        if DETAIL["mode"] != "off":
            r = snap.get("detail")
            if r is not None and snap.get("detail_mode") == DETAIL["mode"]:
                p4 = fit(rsf.detail_to_color(r, DETAIL["mode"]))
                caption(p4, f"DETAIL {DETAIL['mode']}  rms {snap.get('detail_rms', 0):.3f}")
            else:
                p4 = placeholder("detail -")
            panels.append(p4)
    x = GAP + 6
    for p in panels:
        card[28:28 + TH, x:x + TW] = p
        x += TW + GAP
    if tact_txt is not None:
        put(card, "TACTILE", (GAP + 6, 28 + TH + GAP + 18), 0.45, DIM)
        put(card, tact_txt[0], (GAP + 80, 28 + TH + GAP + 18), 0.55, tact_txt[1])
    return card


def pipe_stats(p):
    """What the header needs from one camera (thread mode: read from the CamPipe)."""
    return dict(cam_fps=p.cam_fps, proc_fps=p.proc_fps,
                aref_n=(p.aref.n_resets if p.aref is not None else None),
                aref_on=(p.aref.enabled if p.aref is not None else False))


def render(pipes, metas, state, draw_arrow, show_depth, slip_max):
    cards = [render_camera(p, m["title"], m["color"], m["info"], draw_arrow, show_depth, slip_max)
             for p, m in zip(pipes, metas)]
    return compose(cards, [pipe_stats(p) for p in pipes], metas, state)


def compose(cards, stats, metas, state):
    """Header + one card per camera + key footer -> the window image."""
    W = max(c.shape[1] for c in cards) + 2 * GAP
    hdr = np.full((44, W, 3), BG, np.uint8)
    put(hdr, "TACTILE", (12, 29), 0.75, TXT, 2)
    put(hdr, "DUAL" if len(cards) <= 2 else "TRIPLE" if len(cards) == 3 else f"{len(cards)} CAMS",
        (128, 29), 0.55, DIM, 1)
    x = 190
    for st, m in zip(stats, metas):
        x = chip(hdr, x, 11, f"{m['title']} {st['cam_fps']:.0f} fps", m["color"])
    if state["ae"]:
        x = chip(hdr, x, 11, "AE CSI LOCK", (90, 90, 90))
    if stats and stats[0]["aref_n"] is not None:
        on = stats[0]["aref_on"]
        x = chip(hdr, x, 11, "AUTO-REF " + " ".join(str(st["aref_n"]) for st in stats)
                 if on else "AUTO-REF off", (70, 120, 70) if on else (90, 90, 90))
    # right: recording status
    rx = W - 12
    blink = int(time.time() * 2) % 2 == 0
    items = []
    if state["raw_t0"] is not None:
        items.append((f"RAW {time.monotonic() - state['raw_t0']:5.1f}s", C_REC))
    if state["slip_rec"]:
        items.append((f"SLIP LOG t{state['trial']} {'SLIP' if state['label'] else 'static'}",
                      C_REC if state["label"] else (60, 60, 160)))
    for text, col in reversed(items):
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.45, 1)
        rx -= tw + 36
        chip(hdr, rx, 11, text, col, dot=(255, 255, 255) if blink else col)
    if not items:
        (tw, _), _ = cv2.getTextSize("IDLE", FONT, 0.45, 1)
        chip(hdr, W - tw - 30, 11, "IDLE", (70, 70, 70))

    body = [hdr]
    for c in cards:
        row = np.full((c.shape[0] + GAP, W, 3), BG, np.uint8)
        row[:c.shape[0], GAP:GAP + c.shape[1]] = c
        body.append(row)
    ftr = np.full((30, W, 3), BG, np.uint8)
    keys = [("r", "reset"), ("d", "ROI"), ("a", "arrows"), ("u", "clean"), ("c", "save cfg"),
            ("f", "refine"), ("t", "track"), ("x", "detail"), ("w", "auto-ref"), ("g", "slip log"), ("k", "label"), ("v", "raw rec"),
            ("s", "snapshot"), ("q", "quit")]
    total = sum(24 + cv2.getTextSize(l, FONT, 0.42, 1)[0][0] + 16 for _, l in keys)
    if total > W - 12:                         # narrow window (no depth): short labels
        keys = [("r", "reset"), ("d", "ROI"), ("a", "arrow"), ("u", "clean"), ("c", "save"),
                ("f", "refine"), ("t", "track"), ("x", "detail"), ("w", "aref"), ("g", "slip"), ("k", "label"), ("v", "raw"), ("s", "snap"),
                ("q", "quit")]
    x = 12
    for k, lbl in keys:
        cv2.rectangle(ftr, (x, 7), (x + 18, 25), (80, 80, 80), -1, cv2.LINE_AA)
        put(ftr, k, (x + 5, 21), 0.45, TXT)
        put(ftr, lbl, (x + 24, 21), 0.42, DIM)
        x += 24 + cv2.getTextSize(lbl, FONT, 0.42, 1)[0][0] + 16
    if state["msg"] and time.time() - state["msg_t"] < 4:
        msg = state["msg"]
        if len(msg) > 70:
            msg = msg[:67] + "..."
        (tw, _), _ = cv2.getTextSize(msg, FONT, 0.45, 1)
        cv2.rectangle(ftr, (W - tw - 22, 3), (W - 4, 28), BG, -1)   # message on top of the keys
        put(ftr, msg, (W - tw - 12, 21), 0.45, C_WARN)
    body.append(ftr)
    return np.vstack(body)


# ─────────────────────────────────────────────────────────────────────────────
#  PROCESS MODE (default for two USB cameras): one PROCESS per camera
# ─────────────────────────────────────────────────────────────────────────────
# Python runs one thread at a time (GIL). With both cameras and the GUI as threads of
# one process they wait for each other: on a Pi 5 one camera needs ~25 ms per frame
# alone but ~110 ms in the threaded dual mode. Here each camera has its own process
# (reader, flow/depth, tactile features, adaptive referencing, ROS, raw recorder, and
# it even draws its own card); the main process only shows the cards and sends keys.
# 'threads' = old single-process mode.

class _NullLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _SlogProxy:
    """Stand-in for the shared SlipLogger inside a camera process: rows go to the main
    process, which writes the one CSV (same columns as before)."""
    def __init__(self, edges, out_q):
        self.edges, self.out_q, self.recording = edges, out_q, False

    def write(self, *row):
        try:
            self.out_q.put(("slip", row), timeout=0.5)
        except Exception:
            pass


class _TactView:
    """Tactile summary received from a camera process (what render_camera needs)."""
    def __init__(self):
        self.last, self._sum = None, ("", False)

    def summary(self):
        return self._sum


class _RemotePipe:
    """Main-process stand-in for a CamPipe running in another process: holds the latest
    data it sent, so the SAME render_camera() draws the card here, not in the camera
    process (drawing next to the computation slowed it down 5x on a Pi 5)."""
    def __init__(self, name):
        self.name_, self.cam_fps, self.proc_fps = name, 0.0, 0.0
        self.tact, self.latest, self.aref = _TactView(), None, None

    def snapshot(self):
        return self.latest


def cmd_socket(port=9871):
    """Commands from other programs (servo_grasp.py) as UDP text on 127.0.0.1:<port>:
    'reset' = key r (baseline reset), 'aref hold' = adaptive referencing off while grasping,
    'aref release' = back to the state it had before the hold.
    Returns None if the port is taken (then only the keyboard works)."""
    import socket as _s
    try:
        k = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
        k.bind(("127.0.0.1", port))
        k.setblocking(False)
        print(f"[INFO] listening for commands on UDP {port} ('reset' = baseline reset, like key r)")
        return k
    except OSError as e:
        print(f"[WARN] command port {port} not available ({e}) - baseline reset only by key r")
        return None


def cmd_poll(k):
    """Pending command words from the command socket."""
    out = []
    while k is not None:
        try:
            data, _ = k.recvfrom(256)
        except (BlockingIOError, OSError):
            break
        out.append(data.decode("utf-8", "replace").strip().lower())
    return out


def _cam_proc(spec, cmd_q, out_q):
    """One camera, run in its own process. spec: plain dict (picklable)."""
    import queue as _q
    parent = os.getppid()          # if the main program dies (even kill -9), stop too:
    try:                           # an orphan would keep the camera busy and use CPU
        sys.setswitchinterval(0.001)          # shorter GIL hand-over waits (default 5 ms)
        cv2.setNumThreads(spec["cv_threads"])
        rsf.install_flow_method(spec["flow_method"])
        nm, dev = spec["name"], spec["dev"]
        prof = dict(rsf.PROFILES["usb"])
        reader = rsf.UsbCameraReader(dev, prof["cw"], prof["ch"], prof["fr"])
        prof["cw"], prof["ch"] = reader.size
        cfg = spec["cfg"]
        cfg.focal = FOCAL_AT_1080 / (spec["sensor_w"] / float(prof["cw"]))
        reader.start()
        print(f"[INFO] {nm}: own process (pid {os.getpid()}) {dev} {prof['cw']}x{prof['ch']} "
              f"| focal={cfg.focal:.1f}", flush=True)
        dcal = None
        if not spec["nocalib"]:
            dcal = rsf.DepthCalib()
            f = os.path.join(spec["calib_dir"], f"depth_calib_{nm}.json")
            if os.path.exists(f):
                dcal.load(f)
            else:
                print(f"[CAL] {nm}: no {f} -> built-in SYNTHETIC calibration (approximate)")
                dcal.load_default()
        topt, aopt = spec["topt"], spec["aopt"]
        ros = (tactile_ros.TactileROS([nm], topt["ns"], topt["img_hz"], node_name=f"tactile_sensor_{nm}",
                                      udp=topt["udp"], ros=topt["ros"]) if topt["on"] else None)
        slog = _SlogProxy(spec["edges"], out_q)
        dworker = None
        if spec.get("par_depth", True) and cfg.enable_depth:
            dworker = rsf.DepthWorker(cfg, prof["cw"], prof["ch"], spec["flow_method"],
                                      refine_on=spec["refine"], n_threads=1,
                                      use_gpu=spec.get("gpu", False))
        pipe = CamPipe(nm, reader, cfg, prof["cw"], prof["ch"], None, slog, _NullLock(),
                       spec["slip_rows"], spec["slip_cols"], dcal=dcal,
                       refine_on=spec["refine"], tact=tactile_ros.make_features(topt),
                       ros=ros, aref=adaptive_ref.make(aopt), dworker=dworker,
                       depth_hz=spec.get("depth_hz", 7.5))
        rec = rsf.Recorder(nm, reader, False, spec["raw_fmt"], True, ram=spec["ram"],
                           pre_s=spec["pre_s"])
        rec.start()
        pipe.start()
    except BaseException as e:                       # report, never hang the GUI
        msg = repr(e)
        if "cannot open USB camera" in msg:
            msg += (" -> in use by another program? (an old run still alive?) check: "
                    "pgrep -af 'multiprocessing|run_sensor'   stop: pkill -f multiprocessing.spawn")
        out_q.put(("error", spec.get("name"), msg))
        return
    view = dict(draw_arrow=True, show_depth=bool(cfg.enable_depth), slip_max=spec["slip_max"])
    t_card, card_dt = 0.0, 1.0 / spec.get("send_hz", 10.0)   # display data only (processing runs at camera rate)
    last_sent = None
    running = True
    while running:
        try:
            _w = max(0.0, t_card + card_dt - time.time())     # sleep until the next card
            m = cmd_q.get(timeout=_w) if _w > 0 else cmd_q.get_nowait()
            while True:
                op = m[0]
                if op == "stop":
                    running = False
                elif op == "set":                     # pipe request flags / thresholds
                    setattr(pipe, m[1], m[2])
                elif op == "cfg":
                    for k_, v_ in m[1].items():
                        setattr(pipe.cfg, k_, v_)
                elif op == "track":
                    rsf.set_tracking(pipe.cfg, m[1]); pipe.req_reset = True
                elif op == "aref":
                    pipe.req_aref = m[1]
                elif op == "refine":
                    pipe.refine.on = m[1]; pipe.req_refine_reset = True
                elif op == "view":
                    view.update(m[1])
                elif op == "detail":
                    DETAIL["mode"] = m[1]
                elif op == "arrow":
                    ARROW.update(m[1])
                elif op == "slog":                    # (recording, slip_reset)
                    slog.recording = m[1]; pipe.blocked = m[1]
                    if m[2]:
                        pipe.req_slip_reset = True
                elif op == "save_cfg":
                    try:
                        rsf.save_config(copy.deepcopy(pipe.cfg), m[1], m[2] if len(m) > 2 else None)
                        out_q.put(("msg", f"config saved -> {m[1]}"))
                    except Exception as e:
                        out_q.put(("msg", f"config not saved ({e})"))
                elif op == "snapshot":
                    sn = pipe.snapshot()
                    if sn is not None:
                        cv2.imwrite(os.path.join(m[1], f"raw_{nm}.png"), sn["frame"])
                        if sn["roi"] is not None:
                            cv2.imwrite(os.path.join(m[1], f"roi_{nm}.png"), sn["roi"])
                elif op == "rec_start":
                    rec.start_rec(m[1])
                elif op == "rec_stop":
                    rec.recording = False
                    rows = rec.stop_rec()
                    out_q.put(("rec", nm, rows, rec.summary()))
                m = cmd_q.get_nowait()
        except _q.Empty:
            pass
        orphan = os.getppid() != parent
        if orphan:
            print(f"[INFO] {nm}: main program is gone -> stopping", flush=True)
            out_q.cancel_join_thread()     # nobody reads the queue any more: do not wait for it
            running = False
        now = time.time()
        if now - t_card >= card_dt:
            t_card = now
            sn = pipe.snapshot()
            data = None
            if sn is not None and sn is not last_sent:
                last_sent = sn
                data = {k: sn.get(k) for k in ("roi", "flow", "depth", "depth_mm", "E", "dE",
                                               "flow_mag", "detail", "detail_rms",
                                               "detail_mode")}
                data["frame"] = sn["frame"] if sn.get("roi") is None else None
            tl = pipe.tact.last if pipe.tact is not None else None
            ts = pipe.tact.summary() if tl is not None else ("", False)
            st = pipe_stats(pipe)
            n_ = max(pipe.prof_n, 1)
            st["prof"] = {k_: v_ / n_ * 1000 for k_, v_ in pipe.prof.items()}
            if pipe.dworker is not None:
                st["prof"]["| depth"] = pipe.dworker.ms               # ms per depth frame
                st["prof"]["depth fps"] = pipe.dworker.fps
            if spec["prof"] and pipe.prof_n >= 30:
                pipe.prof, pipe.prof_n = {}, 0
            try:
                out_q.put_nowait(("data", nm, data, st, tl, ts))
            except Exception:
                pass                                   # GUI busy: skip this update
    pipe.running = False
    rec.running = False
    pipe.join(timeout=2)
    if pipe.dworker is not None:
        pipe.dworker.stop()
    reader.stop()
    if ros is not None:
        ros.close()
    if os.getppid() == parent:
        out_q.put(("stopped", nm))
    else:
        os._exit(0)                        # orphan: leave now, nothing left to hand over


def _main_procs(devs, base, args, opts):
    """Main process of PROCESS MODE: windows, trackbars, keys, slip CSV, raw meta."""
    import multiprocessing as mp
    import queue as _q
    ctx = mp.get_context("spawn")                     # clean child, safe with threads
    disp_hz, depth_hz = 10.0, 7.5
    for a in args:
        if a.startswith("--display-hz="):
            disp_hz = max(1.0, float(a.split("=", 1)[1]))
        elif a.startswith("--depth-hz="):
            depth_hz = float(a.split("=", 1)[1])      # 0 = every frame
    names = []
    for dev in devs:
        b = os.path.basename(dev)
        names.append(b.split("_Tactile")[0].replace("usb-", "") if "_Tactile" in b else b)
    colors = [C_CSI, C_USB, C_CAM3]
    metas = [dict(title=nm.upper(), color=colors[i % len(colors)], info="Pi Zero 640x480")
             for i, nm in enumerate(names)]
    slog = SlipLogger("slip_log_dual.csv", opts["slip_rows"], opts["slip_cols"],
                      opts["slip_max"], "dual")
    out_q = ctx.Queue(maxsize=8)
    procs, cmd_qs = [], []
    ncores = os.cpu_count() or 4
    gui = opts.get("gui") or {}
    roi0 = [int(gui.get(f"roi_thresh_{nm}", base.roi_thresh)) for nm in names]
    for i, (nm, dev) in enumerate(zip(names, devs)):
        cfg_i = copy.deepcopy(base); cfg_i.roi_thresh = roi0[i]
        spec = dict(name=nm, dev=dev, cfg=cfg_i, sensor_w=opts["sensor_w"],
                    cv_threads=max(1, (ncores - 1) // len(devs)),
                    flow_method=opts["flow_method"], nocalib=("nocalib" in args),
                    calib_dir=opts["calib_dir"], topt=opts["topt"], aopt=opts["aopt"],
                    edges=slog.edges, slip_rows=opts["slip_rows"], slip_cols=opts["slip_cols"],
                    slip_max=opts["slip_max"], refine=opts["refine"],
                    raw_fmt=opts["raw_fmt"], ram=opts["ram"], pre_s=opts["pre_s"],
                    title=metas[i]["title"], color=metas[i]["color"], info=metas[i]["info"],
                    prof=("prof" in args), send_hz=disp_hz,
                    par_depth=("nopardepth" not in args), depth_hz=depth_hz,
                    gpu=("gpu" in args))
        cq = ctx.Queue()
        pr = ctx.Process(target=_cam_proc, args=(spec, cq, out_q), daemon=False,
                         name=f"cam-{nm}")   # not daemonic: it starts its own depth process
        pr.start()
        procs.append(pr); cmd_qs.append(cq)
    send_init = [("detail", DETAIL["mode"])]
    for op in send_init:
        for cq in cmd_qs:
            cq.put(op)
    print(f"[INFO] PROCESS MODE: {len(procs)} cameras, one process each "
          f"('threads' = old single-process mode)")

    def send(op, *a, only=None):
        for i, cq in enumerate(cmd_qs):
            if only is None or i == only:
                cq.put((op,) + a)

    csock = cmd_socket()
    if "nogui" in args:
        # measurement only: no windows at all (does the desktop/GUI slow the cameras?)
        secs = 20.0
        for a in args:
            if a.startswith("--seconds="):
                secs = float(a.split("=", 1)[1])
        print(f"[INFO] NOGUI: running {secs:.0f} s without any window, then stopping")
        rp_stats = {nm: {} for nm in names}
        t0, t_prof = time.time(), time.time()
        try:
            while time.time() - t0 < secs:
                try:
                    m = out_q.get(timeout=0.2)
                except _q.Empty:
                    continue
                if m[0] == "data":
                    rp_stats[m[1]] = m[3]
                elif m[0] == "error":
                    print(f"[ERROR] {m[1]}: {m[2]}")
                if time.time() - t_prof >= 3.0:
                    t_prof = time.time()
                    try:
                        _l = open("/proc/loadavg").read().split()[0]
                    except OSError:
                        _l = "?"
                    print(f"[PROF] nogui, load {_l}\n       " + "\n       ".join(
                        f"{n}: cam {st.get('cam_fps', 0):.1f} proc {st.get('proc_fps', 0):.1f} fps | "
                        + " ".join(f"{k_} {v_:.1f}{'' if 'fps' in k_ else 'ms'}" for k_, v_ in st.get("prof", {}).items())
                        for n, st in rp_stats.items()), flush=True)
        finally:
            send("stop")
            for p_ in procs:
                p_.join(timeout=3)
                if p_.is_alive():
                    p_.terminate()
            slog.close()
        return

    WIN, TBW = "Tactile DUAL", "Tactile DUAL - controls"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(TBW, cv2.WINDOW_NORMAL); cv2.resizeWindow(TBW, 430, 520)

    def nb(n, v, mx): cv2.createTrackbar(n, TBW, int(v), mx, lambda _: None)
    tb_names = [f"roi_thresh {m['title']}" for m in metas]
    for tn, r0 in zip(tb_names, roi0):
        nb(tn, r0, 200)
    nb("arrow_scale", gui.get("arrow_scale", 5), 25); nb("arrow_step", gui.get("arrow_step", 16), 40)
    nb("deadzone x10", base.flow_deadzone * 10, 40)
    nb("ema x100", base.flow_ema * 100, 95)
    nb("release x100", base.flow_ema_release * 100, 95)
    nb("tavg", max(base.temporal_avg, 1), 8)
    nb("depth_min x100", base.depth_min * 100, 30)
    nb("depth_open", base.depth_open, 9)
    nb("mask margin", max(base.valid_margin, 0), 60)
    nb("rebase hold", base.rebaseline_hold, 30)
    nb("rebase thr x10", base.rebaseline_flow_thresh * 10, 40)
    prev_thr = list(roi0)
    prev = dict(mgn=max(base.valid_margin, 0), vals=None, arrow=None)

    def read_controls(force=False):
        arrow = dict(scale=max(1, cv2.getTrackbarPos("arrow_scale", TBW)),
                     step=max(6, cv2.getTrackbarPos("arrow_step", TBW)))
        if force or arrow != prev["arrow"]:
            ARROW.update(arrow); prev["arrow"] = arrow          # arrows are drawn here
        vals = dict(flow_deadzone=cv2.getTrackbarPos("deadzone x10", TBW) / 10.0,
                    flow_ema=cv2.getTrackbarPos("ema x100", TBW) / 100.0,
                    flow_ema_release=cv2.getTrackbarPos("release x100", TBW) / 100.0,
                    temporal_avg=max(1, cv2.getTrackbarPos("tavg", TBW)),
                    depth_min=cv2.getTrackbarPos("depth_min x100", TBW) / 100.0,
                    depth_open=cv2.getTrackbarPos("depth_open", TBW),
                    auto_rebaseline=False,
                    rebaseline_hold=cv2.getTrackbarPos("rebase hold", TBW),
                    rebaseline_flow_thresh=cv2.getTrackbarPos("rebase thr x10", TBW) / 10.0)
        if force or vals != prev["vals"]:
            send("cfg", vals); prev["vals"] = vals
        mgn = cv2.getTrackbarPos("mask margin", TBW)
        if mgn != prev["mgn"]:
            send("set", "req_margin", mgn); prev["mgn"] = mgn

    CLEAN_TB = {"deadzone x10": 0, "ema x100": 0, "release x100": 0, "tavg": 1, "rebase hold": 0}
    state = dict(ae=False, slip_rec=False, trial=0, label=0, raw_t0=None, msg="", msg_t=0.0)
    show_depth = bool(base.enable_depth)
    flags = dict(draw_arrow=True, refine=opts["refine"], aref=opts["aopt"]["enabled"],
                 track=getattr(base, "tracking_mode", "single"))
    cards = {nm: None for nm in names}
    rpipes = {nm: _RemotePipe(nm) for nm in names}
    dirty = {nm: False for nm in names}
    stats = {nm: dict(cam_fps=0.0, proc_fps=0.0, aref_n=0, aref_on=flags["aref"], prof={})
             for nm in names}
    raw_folder, panel, sized, tick = None, None, False, 0
    t_prof = time.time()
    # Display rate (disp_hz, default 10): drawing + showing a 1300x630 window costs a lot of
    # CPU and memory bandwidth on a Pi 5 desktop (cameras went 60 -> 100 ms per frame with
    # the GUI). Processing stays at the camera rate; only the screen is refreshed less often.
    t_disp = 0.0

    def say(msg):
        print(msg); state["msg"], state["msg_t"] = msg.replace("[INFO] ", ""), time.time()

    def collect(block_for=None, want=None):
        """Drain messages from the camera processes. want: collect 'rec' replies."""
        got, t_end = [], time.time() + (block_for or 0)
        while True:
            try:
                m = out_q.get(timeout=0.05) if block_for else out_q.get_nowait()
            except _q.Empty:
                if block_for and time.time() < t_end and (want is None or len(got) < want):
                    continue
                return got
            if m[0] == "data":
                rp = rpipes[m[1]]
                if m[2] is not None:
                    rp.latest = m[2]
                stats[m[1]] = m[3]
                rp.cam_fps, rp.proc_fps = m[3]["cam_fps"], m[3]["proc_fps"]
                rp.tact.last, rp.tact._sum = m[4], m[5]
                dirty[m[1]] = True
            elif m[0] == "slip":
                if slog.recording:
                    slog.write(*m[1])
            elif m[0] == "msg":
                say("[INFO] " + m[1])
            elif m[0] == "rec":
                got.append(m)
            elif m[0] == "error":
                say(f"[ERROR] camera {m[1]} process failed: {m[2]}")

    read_controls(force=True)
    say("[INFO] press r with NO contact for the baseline")
    try:
        while True:
            for i, tn in enumerate(tb_names):
                v = cv2.getTrackbarPos(tn, TBW)
                if v != prev_thr[i]:
                    send("set", "req_thresh", v, only=i); prev_thr[i] = v
            if tick % 10 == 0:
                read_controls()
            tick += 1
            collect()
            state["slip_rec"], state["trial"], state["label"] = slog.recording, slog.trial, slog.label
            show_now = time.time() - t_disp >= 1.0 / disp_hz
            for i, nm in enumerate(names):                 # draw cards HERE (not in the camera processes)
                if show_now and (dirty[nm] or cards[nm] is None):
                    m_ = metas[i]
                    cards[nm] = render_camera(rpipes[nm], m_["title"], m_["color"], m_["info"],
                                              flags["draw_arrow"], show_depth, opts["slip_max"])
                    dirty[nm] = False
            if show_now and all(c is not None for c in cards.values()):
                t_disp = time.time()
                panel = compose([cards[n] for n in names], [stats[n] for n in names],
                                metas, state)
                if not sized:
                    _sc = min(1.0, MAX_WIN_H / panel.shape[0])     # 3 cameras: fit the screen
                    cv2.resizeWindow(WIN, int(panel.shape[1] * _sc), int(panel.shape[0] * _sc))
                    sized = True
                cv2.imshow(WIN, panel)
            if "prof" in args and time.time() - t_prof >= 3.0:
                t_prof = time.time()
                try:
                    _f = int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) // 1000
                    _l = open("/proc/loadavg").read().split()[0]
                    print(f"[PROF] cpu {_f} MHz, load {_l}", flush=True)
                except OSError:
                    pass
                print("[PROF] " + "\n       ".join(
                    f"{n}: cam {stats[n]['cam_fps']:.1f} proc {stats[n]['proc_fps']:.1f} fps | "
                    + " ".join(f"{k_} {v_:.1f}{'' if 'fps' in k_ else 'ms'}" for k_, v_ in stats[n].get("prof", {}).items())
                    for n in names), flush=True)
            if any(not p.is_alive() for p in procs) and not state.get("dead"):
                state["dead"] = True
                say("[ERROR] a camera process stopped (see messages above); press q")

            k = cv2.waitKey(30) & 0xFF
            for c_ in cmd_poll(csock):
                if c_ == "reset":
                    send("set", "req_reset", True); say("[INFO] baseline reset (all cameras) - requested by the grasp program")
                elif c_ in ("aref hold", "aref release"):
                    # hold: grasping -> no automatic re-reference (it would take the held object
                    # into the reference); release: back to the state from before the hold
                    if c_ == "aref hold" and "aref_prev" not in flags:
                        flags["aref_prev"] = flags["aref"]; want = False
                    elif c_ == "aref release" and "aref_prev" in flags:
                        want = flags.pop("aref_prev")
                    else:
                        want = flags["aref"]
                    if want != flags["aref"]:
                        flags["aref"] = want; send("aref", want)
                        say(f"[INFO] adaptive referencing {'ON' if want else 'OFF'} - grasp program "
                            + ("(fingers open)" if want else "(grasping: the reference is kept)"))
            if k in (27, ord("q")):
                break
            elif k == ord("r"):
                send("set", "req_reset", True); say("[INFO] baseline reset (all cameras)")
            elif k == ord("d"):
                send("set", "req_redetect", True); say("[INFO] ROI re-detected (all cameras)")
            elif k == ord("a"):
                flags["draw_arrow"] = not flags["draw_arrow"]
            elif k == ord("x"):
                DETAIL["mode"] = rsf.DETAIL_MODES[(rsf.DETAIL_MODES.index(DETAIL["mode"]) + 1)
                                                  % len(rsf.DETAIL_MODES)]
                send("detail", DETAIL["mode"]); sized = False
                for nm in names:
                    dirty[nm] = True
                say(f"[INFO] texture detail: {DETAIL['mode'].upper()} (x = off/fine/residual/depth)")
            elif k == ord("t"):
                flags["track"] = "single" if flags["track"] == "adaptive" else "adaptive"
                send("track", flags["track"])
                say(f"[INFO] tracking {flags['track'].upper()} (baseline reset)")
            elif k == ord("w"):
                flags["aref"] = not flags["aref"]; send("aref", flags["aref"])
                say(f"[INFO] adaptive referencing {'ON' if flags['aref'] else 'OFF'}")
            elif k == ord("u"):
                for n_, v_ in CLEAN_TB.items():
                    cv2.setTrackbarPos(n_, TBW, v_)
                read_controls(force=True)
                say("[INFO] controls cleaned (deadzone/EMA/release 0, tavg 1, rebase hold 0) - release, press r")
            elif k == ord("c"):
                read_controls(force=True)
                _g = {f"roi_thresh_{nm}": cv2.getTrackbarPos(tn, TBW)
                      for nm, tn in zip(names, tb_names)}
                _g.update({n_: cv2.getTrackbarPos(n_, TBW) for n_ in ("arrow_scale", "arrow_step")})
                send("save_cfg", opts["config_file"], _g, only=0)
            elif k == ord("f"):
                flags["refine"] = not flags["refine"]; send("refine", flags["refine"])
                say(f"[INFO] depth refinement {'ON' if flags['refine'] else 'OFF'}")
            elif k == ord("e"):
                say("[INFO] USB cameras: AE/AWB is set on the Zero, cannot be locked from here")
            elif k == ord("g"):
                if not slog.recording:
                    trial = slog.start(time.monotonic())
                    send("slog", True, True)
                    say(f"[INFO] recording slip trial {trial} -> {slog.path}")
                else:
                    send("slog", False, False)
                    say(f"[INFO] stop slip trial {slog.stop()} -> {slog.path}")
            elif k == ord("k"):
                lab = slog.toggle_label()
                say(f"[INFO] label -> {'SLIP' if lab else 'static'}")
            elif k == ord("v"):
                if state["raw_t0"] is None:
                    raw_folder = os.path.join(opts["out_root"], time.strftime("rec_%Y%m%d_%H%M%S"))
                    os.makedirs(raw_folder, exist_ok=True)
                    send("rec_start", raw_folder)
                    state["raw_t0"] = time.monotonic()
                    say(f"[INFO] recording raw -> {raw_folder}/")
                else:
                    _finish_raw_remote(raw_folder, send, collect, names, opts)
                    state["raw_t0"] = None
                    say(f"[INFO] raw saved -> {raw_folder}/")
            elif k == ord("s") and panel is not None:
                d = time.strftime("dual_%Y%m%d_%H%M%S"); os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, "panel.png"), panel)
                send("snapshot", os.path.abspath(d))
                say(f"[INFO] snapshot -> {d}/")
    finally:
        if state["raw_t0"] is not None:
            _finish_raw_remote(raw_folder, send, collect, names, opts)
        send("stop")
        t_end = time.time() + 10.0               # camera processes also stop their depth process
        for p in procs:
            while p.is_alive() and time.time() < t_end:
                try:
                    out_q.get(timeout=0.1)       # keep draining so their queues can flush
                except Exception:
                    pass
            if p.is_alive():
                p.terminate()
        slog.close()
        cv2.destroyAllWindows()


def _finish_raw_remote(folder, send, collect, names, opts):
    """Stop raw recording in all camera processes, then write frames.csv + meta.json."""
    import csv
    import json
    send("rec_stop")
    got = collect(block_for=30.0, want=len(names))
    rows, result = [], {}
    for _, nm, r_, summ in got:
        rows += r_; result[nm] = summ
    rows.sort(key=lambda x: float(x[3]))
    with open(os.path.join(folder, "frames.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cam", "idx", "seq", "t_host", "t_sensor_ns", "file"])
        w.writerows(rows)
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), format=opts["raw_fmt"],
                cameras={nm: dict(size=[640, 480], fps_target=15) for nm in names},
                ae_locked_csi=False, result=result, mode="process per camera",
                note="t_host = time.monotonic() (same clock in all processes)")
    with open(os.path.join(folder, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[REC] STOP -> {folder}/")
    for nm, s in result.items():
        print(f"      {nm}: {s.get('frames')} frames, {s.get('fps')} fps, "
              f"camera gaps {s.get('camera_gaps')}, green discarded {s.get('dropped_glitch')}")
    missing = [n for n in names if n not in result]
    if missing:
        print(f"[WARN] no recording result from {missing}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = [a.lower() for a in sys.argv[1:]]
    # Two camera threads + the GUI thread each calling multi-threaded OpenCV = cores
    # oversubscribed (seen on the Pi 5: flow+depth 100+ ms instead of ~25 ms). One OpenCV
    # worker per caller; the cameras run in parallel as threads. --cv-threads=N overrides.
    nthr = 1
    for a in args:
        if a.startswith("--cv-threads="):
            nthr = max(1, int(a.split("=", 1)[1]))
    cv2.setNumThreads(nthr)
    pk = ("full" if "full" in args else "turbo" if "turbo" in args else
          "fast" if "fast" in args else "quality" if "quality" in args else "turbo")
    csi_profile = dict(rsf.PROFILES[pk])
    cam_id = 1 if "cam1" in args else 0
    usb_dev, usb2_dev, usb_sensor_w, out_root, raw_fmt = None, None, 1280, ".", "png"
    usb3_dev = None
    use_ram, pre_s = "ram" in args, 0.0
    flow_method = None
    slip_rows, slip_cols, slip_max_px = 7, 9, 8.0
    depth_refine = False                      # default: calibrated raw depth (no refinement)
    for a in sys.argv[1:]:
        al = a.lower()
        if al.startswith("--usb="):
            usb_dev = a.split("=", 1)[1]
        elif al.startswith("--usb2="):
            usb2_dev = a.split("=", 1)[1]
        elif al.startswith("--usb3="):
            usb3_dev = a.split("=", 1)[1]
        elif al.startswith("--sensor-w="):
            usb_sensor_w = int(al.split("=", 1)[1])
        elif al.startswith("--out="):
            out_root = a.split("=", 1)[1]
        elif al.startswith("--fmt="):
            raw_fmt = al.split("=", 1)[1]
        elif al.startswith("--pre="):
            pre_s = float(al.split("=", 1)[1])
        elif al.startswith("--flow="):
            flow_method = al.split("=", 1)[1]
        elif al.startswith("--window="):
            v = al.split("=", 1)[1]
            csi_profile["window"] = None if v in ("off", "none") else tuple(int(t) for t in v.split(","))
        elif al.startswith("--depth-refine="):
            depth_refine = al.split("=", 1)[1] not in ("off", "0", "none", "no")
        elif al.startswith("--slip-max="):
            slip_max_px = float(al.split("=", 1)[1])
        elif al.startswith("--flow-vis="):
            rsf.FLOW_VIS_MAX = max(0.1, float(al.split("=", 1)[1]))

    # ── base config, then one copy per camera (different focal) ──
    here = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(here, "sensor_config.json")   # same file as run_sensor_fast
    calib_dir = os.path.join(here, "calib")
    clean_cfg = "clean" in args or "--clean" in args
    gui = {} if clean_cfg else rsf.load_gui(config_file)
    base = None
    if not clean_cfg:
        try:
            base = dp.SensorConfig.load(config_file); print(f"[INFO] config loaded: {config_file}")
        except Exception:
            base = None
    if base is None:
        base = dp.SensorConfig()
    if clean_cfg:
        rsf.apply_clean_config(base)
        print("[INFO] CLEAN settings (deadzone 0, EMA 0, tavg 1, rebaseline off); saved config not used")
    if "track=adaptive" in args or "--track=adaptive" in args:
        rsf.set_tracking(base, "adaptive")
    elif "track=single" in args or "--track=single" in args:
        rsf.set_tracking(base, "single")
    print(f"[TRACK] reference tracking: {getattr(base, 'tracking_mode', 'single').upper()} "
          f"('t' = toggle single / adaptive flow chaining)")
    if "detail" in args or "texture" in args:
        DETAIL["mode"] = "fine"
    for _m in ("residual", "depth", "fine"):
        if f"detail={_m}" in args:
            DETAIL["mode"] = _m
    if "nodepth" in args:
        base.enable_depth = False
    base.dist = DIST
    if clean_cfg or not os.path.exists(config_file):
        base.roi_thresh = ROI_THRESH            # otherwise keep the saved value
    import glob

    def usb_name(dev):
        b = os.path.basename(os.path.realpath(dev) if "/by-id/" not in dev else dev)
        return b.split("_Tactile")[0].replace("usb-", "") if "_Tactile" in b else b

    def open_usb(dev):
        prof = dict(rsf.PROFILES["usb"])
        rd = rsf.UsbCameraReader(dev, prof["cw"], prof["ch"], prof["fr"])
        prof["cw"], prof["ch"] = rd.size
        c = copy.deepcopy(base)
        c.focal = FOCAL_AT_1080 / (usb_sensor_w / float(prof["cw"]))
        rd.start()
        print(f"[INFO] USB {usb_name(dev)} {dev} {prof['cw']}x{prof['ch']} | focal={c.focal:.1f}")
        return rd, c, prof

    n_cams = 3 if ("triple" in args or usb3_dev is not None) else 2
    usb_only = ("usb2" in args or usb2_dev is not None or "nocsi" in args or n_cams == 3
                or rsf.Picamera2 is None or not rsf.csi_cameras())
    if n_cams == 3 and "threads" in args:
        sys.exit("[ERROR] three cameras run in process mode only (remove 'threads')")
    if usb_only and "threads" not in args:
        # ── PROCESS MODE: one process per Pi Zero camera (default) ──
        given = [rsf.resolve_usb(d) for d in (usb_dev, usb2_dev, usb3_dev)[:n_cams] if d]
        found = sorted(glob.glob("/dev/v4l/by-id/*Tactile*video-index0"),
                       key=lambda h: os.path.basename(h).split("_Tactile")[0])
        devs = given + [f for f in found if f not in given]   # named first, then the rest
        if len(devs) < n_cams:
            sys.exit(f"[ERROR] {n_cams} Pi Zero cameras needed, found {len(devs)}: "
                     f"{[os.path.basename(f) for f in devs]}. Check: ls /dev/v4l/by-id/")
        devs = devs[:n_cams]
        # Lighter DIS for two cameras on one Pi 5: gradient-descent 16 -> 12, variational
        # refinement 5 -> 2 iterations. Measured on the Pi 5 with real frames: flow
        # 15.7 -> 12.2 ms (alone); flow differs by ~0.03 px median (~ the sensor's static
        # flow noise). --dis=full keeps the deltact_plus settings.
        if "--dis=full" not in args:
            base.dis_grad_iters, base.dis_var_iters = 12, 2
            print("[INFO] DIS light for dual (grad 12, var 2; ~0.03 px from full). "
                  "--dis=full = deltact_plus default (16, 5)")
        aopt = adaptive_ref.parse_args(sys.argv[1:])
        print(f"[AREF] adaptive referencing {'ON' if aopt['enabled'] else 'OFF'} ('w' = toggle)")
        print("[INFO] depth = " + ("REFINED" if depth_refine else
              "CALIBRATED raw depth -> mm") + f" for all {n_cams} cameras ('f' = toggle refinement)")
        return _main_procs(devs, base, args, dict(
            sensor_w=usb_sensor_w, flow_method=flow_method, calib_dir=calib_dir,
            topt=tactile_ros.parse_args(sys.argv[1:]), aopt=aopt,
            slip_rows=slip_rows, slip_cols=slip_cols, slip_max=slip_max_px,
            refine=depth_refine, raw_fmt=raw_fmt, ram=use_ram, pre_s=pre_s,
            out_root=out_root, config_file=config_file, gui=gui))
    cams = []                                # (name, reader, cfg, profile, window, has_md, is_usb, meta)
    if usb_only:
        # ── two Pi Zero USB cameras ──
        if usb_dev and usb2_dev:
            devs = [rsf.resolve_usb(usb_dev), rsf.resolve_usb(usb2_dev)]
        else:
            found = sorted(glob.glob("/dev/v4l/by-id/*Tactile*video-index0"),
                           key=lambda h: os.path.basename(h).split("_Tactile")[0])
            if usb_dev:
                first = rsf.resolve_usb(usb_dev)
                found = [first] + [f for f in found if f != first]
            if len(found) < 2:
                sys.exit(f"[ERROR] dual USB mode needs two Pi Zero cameras, found {len(found)}: "
                         f"{[os.path.basename(f) for f in found]}. Check: ls /dev/v4l/by-id/")
            devs = found[:2]
        for dev in devs:
            rd, c, prof = open_usb(dev)
            nm = usb_name(dev)
            cams.append((nm, rd, c, prof, None, False, True,
                         dict(device=dev, size=[prof["cw"], prof["ch"]], fps_target=prof["fr"])))
    else:
        # ── CSI camera ──
        cfg_csi = copy.deepcopy(base)
        csi = rsf.csi_cameras()
        if cam_id >= len(csi):
            sys.exit(f"[ERROR] CSI camera cam{cam_id} does not exist ({len(csi)} CSI camera(s) detected).")
        picam2 = rsf.open_camera(csi[cam_id], csi_profile["cw"], csi_profile["ch"],
                                 csi_profile["fr"], csi_profile.get("raw"))
        cfg_csi.focal = FOCAL_AT_1080 * (csi_profile["ch"] / 1080.0)
        try:
            crop = None
            for _ in range(5):
                crop = picam2.capture_metadata().get("ScalerCrop")
                if crop:
                    break
            if crop:
                cfg_csi.focal = FOCAL_AT_1080 / (float(crop[2]) / csi_profile["cw"])
        except Exception as e:
            print(f"[WARN] failed to read CSI ScalerCrop ({e})")
        csi_reader = rsf.CameraReader(picam2, name="csi"); csi_reader.start()
        print(f"[INFO] CSI cam{cam_id} {pk} {csi_profile['cw']}x{csi_profile['ch']} "
              f"@{csi_profile['fr']} | focal={cfg_csi.focal:.1f}")
        cams.append(("csi", csi_reader, cfg_csi, csi_profile, csi_profile.get("window"), True, False,
                     dict(profile=pk, size=[csi_profile["cw"], csi_profile["ch"]],
                          fps_target=csi_profile["fr"], cam_index=cam_id)))
        # ── USB camera ──
        dev = rsf.resolve_usb(usb_dev)
        rd, c, prof = open_usb(dev)
        cams.append(("usb", rd, c, prof, None, False, True,
                     dict(device=dev, size=[prof["cw"], prof["ch"]], fps_target=prof["fr"])))
    csi_reader = next((cm[1] for cm in cams if cm[0] == "csi"), None)

    # ── shared slip log ──
    slog = SlipLogger("slip_log_dual.csv", slip_rows, slip_cols, slip_max_px, "dual")
    slog_lock = threading.Lock()

    rsf.install_flow_method(flow_method)      # applies to both cameras
    topt = tactile_ros.parse_args(sys.argv[1:])
    ros = (tactile_ros.TactileROS([cm[0] for cm in cams], topt["ns"], topt["img_hz"],
                                  udp=topt["udp"], ros=topt["ros"]) if topt["on"] else None)

    aopt = adaptive_ref.parse_args(sys.argv[1:])
    print(f"[AREF] adaptive referencing {'ON' if aopt['enabled'] else 'OFF'} ('w' = toggle): "
          f"re-baseline after release / residual flow / brightness change, never during contact")

    def load_calib(nm):
        dc = rsf.DepthCalib()
        if "nocalib" in args:
            return None
        f = os.path.join(calib_dir, f"depth_calib_{nm}.json")
        if os.path.exists(f):
            dc.load(f)
        else:
            print(f"[CAL] {nm}: no {f} -> built-in SYNTHETIC calibration (approximate)")
            dc.load_default()
        return dc

    pipes = [CamPipe(nm, rd, c, prof["cw"], prof["ch"], win, slog, slog_lock, slip_rows, slip_cols,
                     dcal=load_calib(nm), refine_on=depth_refine,
                     tact=tactile_ros.make_features(topt), ros=ros, aref=adaptive_ref.make(aopt))
             for nm, rd, c, prof, win, _, _, _ in cams]
    csock = cmd_socket()
    print("[INFO] depth = " + ("REFINED" if depth_refine else
          "CALIBRATED raw depth -> mm") + " for both cameras ('f' = toggle refinement)")
    for p in pipes:
        p.start()
    colors = [C_CSI, C_USB]
    metas = []
    for i, (nm, rd, c, prof, win, has_md, is_usb, meta) in enumerate(cams):
        if nm == "csi":
            metas.append(dict(title="CSI", color=colors[i],
                              info=f"cam{cam_id} {pk} {prof['cw']}x{prof['ch']}"))
        else:
            metas.append(dict(title=("USB" if nm == "usb" else nm.upper()), color=colors[i],
                              info=f"Pi Zero {prof['cw']}x{prof['ch']}"))

    # ── raw frame recording (Recorder from run_sensor_fast.py) ──
    raw_recs = [rsf.Recorder(nm, rd, has_md, raw_fmt, is_usb, ram=use_ram, pre_s=pre_s)
                for nm, rd, c, prof, win, has_md, is_usb, meta in cams]
    for r in raw_recs:
        r.start()
    raw_meta = {nm: meta for nm, rd, c, prof, win, has_md, is_usb, meta in cams}
    raw_folder = None
    tb_names = [f"roi_thresh {m['title']}" for m in metas]

    WIN = "Tactile DUAL"
    TBW = "Tactile DUAL - controls"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(TBW, cv2.WINDOW_NORMAL); cv2.resizeWindow(TBW, 430, 520)

    def nb(n, v, mx): cv2.createTrackbar(n, TBW, int(v), mx, lambda _: None)
    for tn in tb_names:                                  # per camera
        nb(tn, ROI_THRESH, 200)
    # shared by both cameras (same meaning as in run_sensor_fast)
    nb("arrow_scale",    5,  25)
    nb("arrow_step",     16, 40)
    nb("deadzone x10",   base.flow_deadzone * 10,          40)
    nb("ema x100",       base.flow_ema * 100,              95)
    nb("release x100",   base.flow_ema_release * 100,      95)
    nb("tavg",           max(base.temporal_avg, 1),         8)
    nb("depth_min x100", base.depth_min * 100,             30)
    nb("depth_open",     base.depth_open,                   9)
    nb("mask margin",    max(base.valid_margin, 0),        60)
    nb("rebase hold",    base.rebaseline_hold,             30)
    nb("rebase thr x10", base.rebaseline_flow_thresh * 10, 40)
    prev_thr = [ROI_THRESH, ROI_THRESH]
    prev_mgn = max(base.valid_margin, 0)
    tick = 0

    def read_controls():
        nonlocal prev_mgn
        ARROW["scale"] = max(1, cv2.getTrackbarPos("arrow_scale", TBW))
        ARROW["step"] = max(6, cv2.getTrackbarPos("arrow_step", TBW))
        vals = dict(flow_deadzone=cv2.getTrackbarPos("deadzone x10", TBW) / 10.0,
                    flow_ema=cv2.getTrackbarPos("ema x100", TBW) / 100.0,
                    flow_ema_release=cv2.getTrackbarPos("release x100", TBW) / 100.0,
                    temporal_avg=max(1, cv2.getTrackbarPos("tavg", TBW)),
                    depth_min=cv2.getTrackbarPos("depth_min x100", TBW) / 100.0,
                    depth_open=cv2.getTrackbarPos("depth_open", TBW),
                    auto_rebaseline=False,
                    rebaseline_hold=cv2.getTrackbarPos("rebase hold", TBW),
                    rebaseline_flow_thresh=cv2.getTrackbarPos("rebase thr x10", TBW) / 10.0)
        for p in pipes:
            for k_, v_ in vals.items():
                setattr(p.cfg, k_, v_)
        mgn = cv2.getTrackbarPos("mask margin", TBW)
        if mgn != prev_mgn:
            for p in pipes: p.req_margin = mgn
            prev_mgn = mgn

    CLEAN_TB = {"deadzone x10": 0, "ema x100": 0, "release x100": 0, "tavg": 1, "rebase hold": 0}

    state = dict(ae=False, slip_rec=False, trial=0, label=0, raw_t0=None, msg="", msg_t=0.0)
    draw_arrow, panel = True, None
    show_depth = bool(base.enable_depth)
    sized = False

    def say(msg):
        print(msg); state["msg"], state["msg_t"] = msg.replace("[INFO] ", ""), time.time()

    do_prof = "prof" in args
    t_prof, prof_main = time.time(), {"render": 0.0, "imshow+keys": 0.0, "n": 0}
    say("[INFO] press r with NO contact for the baseline")
    try:
        while True:
            for i, name in enumerate(tb_names):
                v = cv2.getTrackbarPos(name, TBW)
                if v != prev_thr[i]:
                    pipes[i].req_thresh = v; prev_thr[i] = v
            if tick % 10 == 0:
                read_controls()
            tick += 1

            state["slip_rec"], state["trial"], state["label"] = slog.recording, slog.trial, slog.label
            for p in pipes:
                p.blocked = slog.recording
            _r0 = time.perf_counter()
            panel = render(pipes, metas, state, draw_arrow, show_depth, slip_max_px)
            _r1 = time.perf_counter()
            if not sized:
                cv2.resizeWindow(WIN, panel.shape[1], panel.shape[0]); sized = True
            cv2.imshow(WIN, panel)

            k = cv2.waitKey(33) & 0xFF          # display ~30 Hz; processing runs in threads
            if do_prof:
                prof_main["render"] += _r1 - _r0
                prof_main["imshow+keys"] += time.perf_counter() - _r1 - 0.033
                prof_main["n"] += 1
                if time.time() - t_prof >= 3.0:
                    nm = max(prof_main["n"], 1)
                    line = [f"[PROF] GUI render {prof_main['render'] / nm * 1000:.1f}ms "
                            f"imshow {max(0.0, prof_main['imshow+keys'] / nm * 1000):.1f}ms"]
                    for p in pipes:
                        n_ = max(p.prof_n, 1)
                        st = " ".join(f"{k_} {v_ / n_ * 1000:.1f}ms" for k_, v_ in p.prof.items())
                        line.append(f"{p.name_}: cam {p.cam_fps:.1f} proc {p.proc_fps:.1f} fps | {st}")
                        p.prof, p.prof_n = {}, 0
                    print("\n       ".join(line), flush=True)
                    t_prof, prof_main = time.time(), {"render": 0.0, "imshow+keys": 0.0, "n": 0}
            for c_ in cmd_poll(csock):
                if c_ == "reset":
                    for p in pipes: p.req_reset = True
                    say("[INFO] baseline reset (all cameras) - requested by the grasp program")
                elif c_ in ("aref hold", "aref release"):
                    want = pipes[0].aref.enabled
                    if c_ == "aref hold" and not hasattr(pipes[0], "aref_prev"):
                        pipes[0].aref_prev = pipes[0].aref.enabled; want = False
                    elif c_ == "aref release" and hasattr(pipes[0], "aref_prev"):
                        want = pipes[0].aref_prev; del pipes[0].aref_prev
                    if want != pipes[0].aref.enabled:
                        for p in pipes:
                            p.req_aref = want
                        say(f"[INFO] adaptive referencing {'ON' if want else 'OFF'} - grasp program")
            if k in (27, ord("q")):
                break
            elif k == ord("r"):
                for p in pipes: p.req_reset = True
                say("[INFO] baseline reset (both cameras)")
            elif k == ord("d"):
                for p in pipes: p.req_redetect = True
                say("[INFO] ROI re-detected (both cameras)")
            elif k == ord("a"):
                draw_arrow = not draw_arrow
            elif k == ord("x"):
                DETAIL["mode"] = rsf.DETAIL_MODES[(rsf.DETAIL_MODES.index(DETAIL["mode"]) + 1)
                                                  % len(rsf.DETAIL_MODES)]
                sized = False
                say(f"[INFO] texture detail: {DETAIL['mode'].upper()} (x = off/fine/residual/depth)")
            elif k == ord("t"):
                mode = "single" if getattr(pipes[0].cfg, "tracking_mode", "single") == "adaptive" \
                    else "adaptive"
                for p in pipes:
                    rsf.set_tracking(p.cfg, mode); p.req_reset = True
                say(f"[INFO] tracking {mode.upper()} (baseline reset)")
            elif k == ord("w"):
                want = not pipes[0].aref.enabled
                for p in pipes:
                    p.req_aref = want
                say(f"[INFO] adaptive referencing {'ON' if want else 'OFF'}")
            elif k == ord("u"):
                for n_, v_ in CLEAN_TB.items():
                    cv2.setTrackbarPos(n_, TBW, v_)
                read_controls()
                say("[INFO] controls cleaned (deadzone/EMA/release 0, tavg 1, rebase hold 0) - release, press r")
            elif k == ord("c"):
                read_controls()
                save = copy.deepcopy(pipes[0].cfg)
                try:
                    rsf.save_config(save, config_file); print(f"[INFO] config saved -> {config_file}")
                    say("[INFO] config saved (sensor_config.json)")
                except Exception as e:
                    say(f"[WARN] config not saved ({e})")
            elif k == ord("f"):
                for p in pipes:
                    p.refine.on = not p.refine.on; p.req_refine_reset = True
                say(f"[INFO] depth refinement {'ON' if pipes[0].refine.on else 'OFF'}")
            elif k == ord("e") and csi_reader is None:
                say("[INFO] USB cameras: AE/AWB is set on the Zero, cannot be locked from here")
            elif k == ord("e"):
                state["ae"] = not state["ae"]
                if state["ae"]:
                    md = csi_reader.get_metadata()
                    csi_reader.set_controls(AeEnable=False, AwbEnable=False,
                                            ExposureTime=int(md.get("ExposureTime") or 8000),
                                            AnalogueGain=float(md.get("AnalogueGain") or 2.0))
                    say("[INFO] CSI AE/AWB locked (USB cannot be locked)")
                else:
                    csi_reader.set_controls(AeEnable=True, AwbEnable=True)
                    say("[INFO] AE/AWB CSI auto")
            elif k == ord("g"):
                with slog_lock:
                    if not slog.recording:
                        for p in pipes: p.req_slip_reset = True
                        trial = slog.start(time.monotonic())
                        say(f"[INFO] recording slip trial {trial} -> {slog.path}")
                    else:
                        say(f"[INFO] stop slip trial {slog.stop()} -> {slog.path}")
            elif k == ord("k"):
                with slog_lock:
                    lab = slog.toggle_label()
                say(f"[INFO] label -> {'SLIP' if lab else 'static'}")
            elif k == ord("v"):
                if state["raw_t0"] is None:
                    os.makedirs(out_root, exist_ok=True)
                    raw_folder = os.path.join(out_root, time.strftime("rec_%Y%m%d_%H%M%S"))
                    os.makedirs(raw_folder, exist_ok=True)
                    for r in raw_recs:
                        r.start_rec(raw_folder)
                    state["raw_t0"] = time.monotonic()
                    say(f"[INFO] recording raw -> {raw_folder}/")
                else:
                    rsf.finish_recording(raw_folder, raw_recs, raw_meta, raw_fmt, state["ae"])
                    state["raw_t0"] = None
                    say(f"[INFO] raw saved -> {raw_folder}/")
            elif k == ord("s") and panel is not None:
                d = time.strftime("dual_%Y%m%d_%H%M%S"); os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, "panel.png"), panel)
                for p in pipes:
                    sn = p.snapshot()
                    if sn is not None:
                        cv2.imwrite(os.path.join(d, f"raw_{p.name_}.png"), sn["frame"])
                        if sn["roi"] is not None:
                            cv2.imwrite(os.path.join(d, f"roi_{p.name_}.png"), sn["roi"])
                say(f"[INFO] snapshot -> {d}/")
    finally:
        if state["raw_t0"] is not None:
            rsf.finish_recording(raw_folder, raw_recs, raw_meta, raw_fmt, state["ae"])
        for p in pipes:
            p.running = False
        for r in raw_recs:
            r.running = False
        for p in pipes:
            p.join(timeout=2)
        for cm in cams:
            cm[1].stop()
        slog.close()
        if ros is not None:
            ros.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
