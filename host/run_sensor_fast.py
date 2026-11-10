"""
run_sensor_fast.py  —  SINGLE CAMERA, HIGH-FPS (Raspberry Pi 5 / Picamera2)
===========================================================================
Single-sensor version optimised for high frame rate. Uses the same deltact_plus
pipeline (DIS flow + density-difference depth), but:
  - ONE camera   -> no ISP/bandwidth contention (main cause of low FPS in dual)
  - Open3D OFF by default (3D heightmap = biggest FPS cost)
  - arrows drawn on a grid (cheap) and can be disabled ('a')
  - depth can be computed every N frames ('depth stride') or disabled entirely
  - exposure / gain / AWB can be LOCKED ('e')  <-- important for experiment data

  *** MOD 1: CENTER SUPPRESSION (Gaussian taper) to remove the indenter peak ***
  *** MOD 2: NO-CONTACT PROBE (m / M) for Table IV data ***
  *** MOD 3: NOISE FLOOR TEST (n) through the same deltact_plus pipeline ***

USAGE
  python3 run_sensor_fast.py                 # default 1280x720 @ 60
  python3 run_sensor_fast.py turbo           # 640x480  @ 120  (highest FPS)
  python3 run_sensor_fast.py fast            # 1280x720 @ 90
  python3 run_sensor_fast.py quality         # 1920x1080 @ 30
  python3 run_sensor_fast.py full            # 820x616 @ 40, FULL sensor FOV (18x24 mm window)
  python3 run_sensor_fast.py cam1            # use camera index 1 (default 0)
  python3 run_sensor_fast.py nodepth         # flow only (even higher FPS)
  python3 run_sensor_fast.py 3d              # 3D depth view (see below)
  (arguments can be combined, e.g.: python3 run_sensor_fast.py turbo nodepth)

  TWO CAMERAS (CSI + USB at once, one window):
  python3 run_sensor_fast.py dual                  # CSI turbo + USB
  python3 run_sensor_fast.py dual full nodepth     # CSI full FOV, no depth
  keys: g = slip log for both cameras, v = record raw frames from both cameras
  (needs run_sensor_dual.py in the same folder)

  USB CAMERA (Pi Zero as a UVC webcam, ~15 fps):
  python3 run_sensor_fast.py usb                   # find 'Tactile UVC Cam' automatically
  python3 run_sensor_fast.py usb nodepth           # flow only
  python3 run_sensor_fast.py --usb=/dev/video8     # force a specific device
  python3 run_sensor_fast.py --usb=camera2         # pick a Zero by hostname (2 Zeros plugged in)
  --usb-size=384x288 / 320x256   smaller USB frame -> the Zero W holds 15 fps more easily
  --usb-fps=10                   lock the USB camera at 10 fps (640x480 offers 15 and 10)
  --sensor-w=N   sensor width used by the Zero (1280 = 640x480 crop mode,
                 3280 = 1640x1232 full-FOV mode) -> sets the undistort focal length
  --focal=N      force the focal length (px)
  Note: over USB, exposure/gain/AWB can NOT be locked from the Pi 5 (e/u only
  print a warning). Slip data must be re-recorded with this camera.

  RAW FRAME RECORDING ('v' = start/stop, one folder rec_.../ per recording):
  ram          keep frames in RAM while recording, write to disk after stop
               -> no frames lost to a slow disk (uses the 16 GB RAM)
  --pre=N      the N seconds BEFORE 'v' is pressed are also saved (pre-trigger)
  --out=DIR    output folder (e.g. --out=/mnt/ssd)    --fmt=npy  faster than png
  Example: python3 run_sensor_fast.py usb ram --pre=5 --out=/mnt/ssd

  OPTICAL FLOW METHOD (--flow=..., default = built-in DIS of deltact_plus):
  medium      DIS medium: lowest practical static noise (0.027 px, ~9 ms)  <- recommended
  medium_ref  DIS medium + variational refinement (~11 ms)
  farneback   Farneback (~25 ms; on static data NOT lower noise than medium)
  pcaflow     very smooth (~55 ms; may smooth out local deformation, careful)
  tvl1        TV-L1 (heaviest; needs opencv-contrib)
  ultrafast / fast   fastest, higher noise
  Time budget: USB 15 fps = 67 ms/frame (all fit). CSI 100 fps = 10 ms (DIS only).

  CLEAN SETTINGS:  python3 run_sensor_fast.py usb clean
               ignores the saved sensor_config.json FOR THIS RUN (the file is kept; 'c' saves)
               and starts neutral: deadzone 0, EMA 0, tavg 1, rebaseline off, margin 15,
               center_suppress 0. Then 'r' (no contact) and 'c' to keep it.
               The config is always the one NEXT TO THIS SCRIPT, whatever folder you start from.

  DEPTH REFINEMENT (OFF by default = calibrated raw depth; 'f' = toggle, --depth-refine=on = start with it):
  post-processing of the deltact_plus depth map for a solid, stable surface:
  5x5 median (spikes) -> floor/offset removal -> motion-adaptive temporal filter
  (static = averaged, real change > 3 sigma = taken immediately, no lag) -> bilateral
  -> contact mask (hysteresis depth_min / 2x depth_min, small blobs dropped)
  -> hole filling inside the contact -> feathered edge. The noise sigma is estimated
  online. Status bar: 'DF 4ms' = refinement time. Calibration ('p'/'o'/'l') and
  the 3D view use the refined depth.

  gpu          depth (density + blur + guided filter) on an NVIDIA GPU with CuPy
               (Jetson: pip install cupy-cuda12x); checked against the CPU at start,
               falls back to the CPU if CuPy is missing or the results differ.
  pardepth     (DEFAULT) depth + refinement in a SEPARATE PROCESS (other cores):
               flow/slip never wait for depth. 'nopardepth' = old inline mode.
               Status bar: 'D 25fps 30ms' = depth rate & processing time.

  3d           3D depth view with OpenCV (no OpenGL, works on the Pi 5):
               left drag = rotate, scroll = zoom, right drag = z scale, double click = reset
  o3d          old Open3D window (needs OpenGL >= 4.1, does not run on the Pi 5 GPU)

  LENS CALIBRATION
  --lens=checker   use the checkerboard calibration from the patent deck (slide 28):
                   k1=-0.857, k2=1.113, focal 356.5 px @820x616, centre (419.7, 307.3).
                   CSI camera, 'full' profile only. Default = k1 -0.60, k2 -0.21 (old).

  DEPTH -> mm CALIBRATION (deck method: 0.1 mm steps, MTS / micrometer)
  1. 'r' without contact. 2. Lower the indenter 0.1 mm, hold ~1 s, press 'p'.
  3. Repeat for every step (0.2, 0.3 ... mm). 'z' = delete last point.
  4. 'o' = linear fit + RMSE/MAE -> depth_calib_usb.json / depth_calib_csi.json
  5. Saved as calib/depth_calib_<camera>.json next to this script (e.g. depth_calib_camera2.json)
     and loaded AUTOMATICALLY next time for that camera. --depth-calib=FILE overrides it.
  With no calib file and no --depth-calib: the BUILT-IN calibration (synthetic, approx. +-30 %) is used.
  --depth-calib=none  -> relative depth units as before.
  Automatic with MTS files: 'l' = start/stop per-frame depth log (depthlog_*.csv),
  run the MTS program at the same time, then: python3 mts_calib.py depthlog_*.csv exp1_test*.txt
  --calib-step=0.1   step per point (mm)   --calib-metric=max|p99  (default max = depth_max)

  FPS OPTIONS:
  --depth-stride=N   compute depth every N frames (e.g. 3); flow & slip still every frame
  --display-hz=N     limit display to N Hz (default 30); processing still every frame
  prof               print per-stage timing every 100 frames:
                     [PROF] process=.. | fitur=.. | viz=.. | gui=.. | loop=..fps cam=..fps
  The status bar shows 'cam=' (camera FPS) next to the loop FPS.
  Example: python3 run_sensor_fast.py turbo prof --depth-stride=3

CONTROLS
  r = reset baseline (do this WITHOUT contact)
  d = re-detect ROI             a = toggle arrows on/off (FPS)
  x = texture DETAIL panel: off -> fine (2nd full-res flow on what the main flow missed)
      -> residual (photometric) -> depth high-pass.  Start with it: detail
      (or detail=residual / detail=depth)
  t = tracking single / ADAPTIVE (paper flow chaining: tracks large deformation)
      start with it: track=adaptive
  w = adaptive referencing on/off (contact-aware auto re-baseline, see adaptive_ref.py)
  f = depth refinement on/off (compare raw vs refined depth)
  e = toggle LOCK exposure/gain/AWB (for experiments)
  [ / ] = exposure -/+          - / = : gain -/+   (while locked)
  s = snapshot: display panel + RAW camera frame + raw ROI (see RAW SNAPSHOTS)
  b = burst of BURST_N consecutive raw frames -> folder raw_<timestamp>/ (for calibration)
      (S works too)
  c = save config
  g = start/stop RECORDING slip features -> slip_log_1cam.csv (E, dE/dt, Vx, Vy)
  k = flip the CSV row label: static <-> slip (press EXACTLY when slip starts)
  v = start/stop raw frame recording    l = start/stop per-frame depth log
  p / z / o = depth calibration: add point / undo / fit & save
  q / ESC = quit

NO-CONTACT MEASUREMENT CONTROLS (Table IV)
  u = MEASURE MODE (M works too): zero every disturbing setting + lock the camera at a
      FIXED operating point (MEASURE_EXP_US / MEASURE_GAIN), fixed AWB, ISP NR &
      sharpening off
  m = start/cancel a 120-frame recording for one decal
  n = NOISE FLOOR TEST (no contact, a few seconds, the window freezes briefly)

  The second status line shows ||F||, false depth and low_f live.
  States: settling -> STABLE -> press m. If BLOCKED, the reason is printed
  on the red line above it.

  IMPORTANT: three features in this script corrupt the no-contact measurement and
  all of them are on by default:
    center_suppress  multiplies flow by a mask -> shrinks what is measured
    rebase hold      resets the baseline when idle -> erases the drift being measured
    deadzone/ema/tavg  runtime stage of Sec. VII -> Table IV is reported WITHOUT these
  Press 'M' to disable all three at once. The probe refuses to record until
  they are clean, and marks result rows NONCONFORMING if forced.

  Why M locks at FIXED numbers instead of the current AE values: AE picks a
  different exposure for each decal (dark decal -> long exposure), so locking
  "at the AE value" still measures decals at different operating points.
  All decals must be measured at the same exposure/gain.

RAW SNAPSHOTS ('s' and 'b')
  's' used to save only the display panel (status text, flow/depth panels, resized
  to the working grid). That can NOT be used for calibration. Now:
    raw_XX.png   the camera frame AS IS (profile resolution, fisheye, not undistorted,
                 no text, no gain/bright trackbars)
    roi_XX.png   the raw ROI fed to DIS (undistorted + cropped + resized by
                 deltact_plus), also without display gain/bright
    meta.json    exposure, gain, colour gains, NR, measure mode, profile, undistort, ROI
  'b' saves BURST_N DIFFERENT consecutive frames without blocking the loop.
  For calibration: press 'u' first (fixed operating point), wait for STABLE, then 'b'.

NOISE FLOOR TEST ('n')
  Grabs N no-contact frames, then runs TWO sequences through a new
  dp.TactileSensor with the same cfg (identical undistort, ROI, DIS):
    measured     the real frames
    noise-only   a clean image (mean of N frames) + synthetic sensor noise
                 measured from the data (sigma per channel, per intensity). There is
                 no motion -> flow here is pure noise.
  plus 'gain-matched' (real frames normalised in brightness to the baseline) to
  detect LED flicker, and motion structure (global shift / coherent) to
  separate rig vibration, gel motion and noise.
  Results are printed and saved to noise_floor_<timestamp>.json.

CENTER SUPPRESSION CONTROLS (Controls panel)
  - "center_suppress" : 0..100 (suppression strength, 0=off, 100=max)
  - "supp_radius %" : 10..60 (Gaussian radius, % of ROI)

FPS NOTES
  Center suppression (Gaussian mask) is very cheap: O(WxH) -> negligible overhead.
  The no-contact probe is cheap too (~1 FFT per 10 frames).
  The bottlenecks are unchanged: (1) Open3D, (2) depth guided filter, (3) drawing arrows.

SEPARATE 3D VIEWER (LIGHTWEIGHT)
  This script PUBLISHES depth to shared memory (/dev/shm) every frame (very
  cheap, ~memcpy). Run the 3D view in a SEPARATE PROCESS so a heavy viewer does
  NOT lower this loop's FPS:
      Terminal 1:  python3 run_sensor_fast.py turbo      # do NOT use the '3d' arg
      Terminal 2:  python3 view_3d_live.py
  Needs sensor_share.py & view_3d_live.py in the same folder.
  Note: the viewer needs depth -> do NOT use 'nodepth' if you want to see 3D.
"""
from __future__ import annotations

import copy
import json
import queue
import csv
import collections
import os
import sys
import time
import threading
from typing import Optional

import numpy as np
import cv2
import deltact_plus as dp
import tactile_ros                         # << contact/force/shear/slip features + ROS 2 topics
import adaptive_ref                        # << contact-aware automatic re-baseline
try:
    from dt3_slip import (SLIP_BINS, SlipLogger,  # << slip features (Hu et al. 2024)
                          make_slip_grid, slip_features)
except ImportError:
    sys.exit("[ERROR] dt3_slip.py is missing: copy it next to run_sensor_fast.py "
             "(it holds the slip features).")
try:
    from sensor_share import DepthPublisher   # << depth to shared memory (separate 3D viewer)
except ImportError:
    class DepthPublisher:                     # optional: only needed for view_3d_live.py
        def __init__(self, *a, **k): pass
        def publish(self, *a, **k): pass
        def close(self): pass
try:
    from nocontact_probe import NoContactProbe  # << no-contact measurement (Table IV)
except ImportError:
    class NoContactProbe:                     # optional: only the 'm' no-contact recording
        recording = False
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): pass
        def overlay(self, panel): return panel
        def toggle_record(self, *a, **k):
            print("[INFO] nocontact_probe.py not present -> 'm' recording unavailable")

try:
    from picamera2 import Picamera2
    from libcamera import controls as libcontrols  # noqa: F401 (for AE/AWB enums when available)
except ImportError:
    Picamera2 = None   # may be missing when only the USB camera ('usb') is used

# Use all cores for OpenCV operations (blur/resize/remap)
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 4))
except Exception:
    pass


# FIXED operating point for measure mode (same as capture_nocontact.py / paper)
MEASURE_EXP_US = 4536
MEASURE_GAIN = 1.0
MEASURE_COLOUR_GAINS = (1.2, 2.0)

# Raw frame burst ('b') for calibration / test strip
BURST_N = 10
BURST_SKIP = 5          # frames DISCARDED before the burst is saved (brightness still rises ~1%)
SETTLE_AFTER_U_S = 3.0  # burst is delayed until >= 3 s after 'u'

# Gel window (x0, y0, x1, y1) in raw-frame pixels. Everything outside this box is
# zeroed BEFORE entering deltact_plus, so ROI detection (brightness threshold)
# does not pick up the LED ring, which is brighter than the paper. SAVED raw
# frames stay intact. None = no masking.
WINDOW_BOX = None


def apply_window(frame, box):
    if box is None:
        return frame
    x0, y0, x1, y1 = box
    out = np.zeros_like(frame)
    out[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Center Suppression (Gaussian taper)
# ─────────────────────────────────────────────────────────────────────────────
def make_gaussian_center_mask(h, w, radius_pct=10.0, sigma_scale=1.0):
    """
    Build a Gaussian taper mask that suppresses artifacts in the ROI centre.

    Args:
        h, w: mask size (ROI height/width)
        radius_pct: suppression radius as % of min(h,w) (default 10%)
        sigma_scale: Gaussian steepness (> 1 = sharper, < 1 = softer)

    Returns:
        mask (h, w, float32) in [0, 1], 0 at the centre (suppressed), 1 at the edges
    """
    if radius_pct <= 0:                   # radius 0 = no suppression (avoids 0/0 -> NaN)
        return np.ones((h, w), np.float32)
    cy, cx = h / 2.0, w / 2.0
    min_dim = min(h, w)
    radius = max((radius_pct / 100.0) * min_dim, 1.0)
    sigma = radius / (2.0 * sigma_scale)  # sigma_scale > 1 = sharper

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    # Gaussian: exp(-dist^2 / (2*sigma^2))
    # Inverted: 1 - Gaussian, so 0 at the centre and 1 at the edges
    mask = 1.0 - np.exp(-dist2 / (2.0 * sigma ** 2))
    return mask.astype(np.float32)


def apply_center_suppression(data, mask, strength=50):
    """
    Apply centre suppression to data (depth or flow magnitude).

    Args:
        data: (h, w) or (h, w, c) array
        mask: (h, w) Gaussian taper mask [0..1]
        strength: 0..100, suppression strength

    Returns:
        suppressed data in the same format
    """
    if data is None or strength == 0:
        return data

    alpha = strength / 100.0

    if len(data.shape) == 2:
        # Scalar field (depth)
        return data * (1.0 - alpha * (1.0 - mask[:, :])).astype(data.dtype)
    else:
        # Vector field (flow): suppress magnitude
        h, w = mask.shape
        mask_3d = (1.0 - alpha * (1.0 - mask[:h, :w]))
        if data.shape[2] == 2:  # flow (h, w, 2)
            return (data * mask_3d[:, :, np.newaxis]).astype(data.dtype)
        else:  # rgb or other
            return (data * mask_3d[:, :, np.newaxis]).astype(data.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  TEXTURE DETAIL ('x' / arg detail): fine surface detail that flow/depth smooth away
# ─────────────────────────────────────────────────────────────────────────────
DETAIL_MODES = ("off", "fine", "residual", "depth")
DETAIL_FLOOR = {"fine": 0.02, "residual": 4.0, "depth": 0.01}   # display floor (no noise blow-up)
_FINE_DIS = {}


def _fine_dis():
    """Second DIS at FULL working resolution (finest scale 0) for the leftover motion."""
    d = _FINE_DIS.get("d")
    if d is None:
        d = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        d.setFinestScale(0); d.setPatchSize(6); d.setPatchStride(3)
        d.setGradientDescentIterations(8); d.setVariationalRefinementIterations(0)
        _FINE_DIS["d"] = d
    return d


def texture_detail(sensor, res, depth=None, mode="residual", hp_sigma=4.0):
    """-> (signed float32 detail map, rms) or (None, 0).
    fine:     (default) the current frame is warped back by the main flow, then a second
              full-resolution flow measures what is LEFT (motion the main flow smoothed away);
              its negative divergence = fine relief (bumps bright, grooves dark).
    residual: current frame warped back by the flow minus the reference frame. What the
              flow explains cancels; what is left is shading of small features the flow
              is too coarse to follow -> the surface texture.
    depth:    high-pass of the depth map (depth - blur): large shape removed, relief left.
    Both are high-passed (sigma hp_sigma px) so lighting/global shape do not dominate."""
    if res is None or mode == "off":
        return None, 0.0
    if mode in ("residual", "fine"):
        tr = sensor.tracker
        adaptive = getattr(sensor.cfg, "tracking_mode", "single") == "adaptive"
        ref = tr.stack[0].gray if (adaptive and tr.stack) else tr.baseline
        gray, flow = res.get("gray"), res.get("flow")
        if ref is None or gray is None or flow is None or ref.shape != gray.shape:
            return None, 0.0
        warped = dp.warp_by_flow(gray, flow)
        if mode == "fine":
            fl = _fine_dis().calc(ref, warped.astype(np.uint8), None)
            r = -(cv2.Sobel(fl[..., 0], cv2.CV_32F, 1, 0, ksize=3) +
                  cv2.Sobel(fl[..., 1], cv2.CV_32F, 0, 1, ksize=3)) / 8.0
        else:
            r = warped.astype(np.float32) - ref.astype(np.float32)
    else:
        if depth is None:
            return None, 0.0
        r = depth.astype(np.float32)
    r = r - cv2.GaussianBlur(r, (0, 0), hp_sigma)
    if sensor.mask is not None and sensor.mask.shape == r.shape:
        r = r * sensor.mask
    fm = sensor.undist.fold_mask
    if fm is not None and fm.shape == r.shape:
        r = r * fm
    return r, float(np.sqrt(np.mean(r * r)))


def detail_to_color(r, mode):
    """Grey relief image: 128 = flat, brighter/darker = detail."""
    vmax = max(float(np.percentile(np.abs(r), 99)), DETAIL_FLOOR.get(mode, 1e-3))
    u8 = np.clip(128.0 + r / vmax * 127.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


FLOW_VIS_MAX = 3.0        # px of flow shown at full brightness (--flow-vis=N)


def _usb_buffers():
    """V4L2 buffers per USB camera (--usb-buffers=N, default 8). OpenCV's default is 4:
    when the host is a few ms late returning a buffer (several cameras, busy CPU), the
    next frame finds no free buffer and uvcvideo drops it. More buffers absorb that
    jitter. No added delay: the reader keeps only the newest frame. Read from argv or
    the environment, so camera processes started with 'spawn' get the same value."""
    v = os.environ.get("TACTILE_USB_BUFFERS")
    for a in sys.argv[1:]:
        if a.lower().startswith("--usb-buffers="):
            v = a.split("=", 1)[1]
    try:
        n = max(2, min(10, int(v))) if v else 8      # OpenCV V4L2 accepts 1..10
    except ValueError:
        n = 8
    os.environ["TACTILE_USB_BUFFERS"] = str(n)
    return n


USB_BUFFERS = _usb_buffers()


def flow_to_color_fixed(flow, vmax=None):
    """Flow colour image with a FIXED brightness scale (0..vmax px). deltact_plus's
    flow_to_color normalises to each frame's own maximum, so a 0.3 px residue after a
    release looks as bright as a real press. Hue = direction, as before."""
    vmax = FLOW_VIS_MAX if vmax is None else vmax
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.empty(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * (90.0 / np.pi)).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * (255.0 / max(vmax, 1e-6)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def label_bottom(img, t, c=(0, 255, 0)):
    """Write a label at the BOTTOM of the image (keeps the content clear)"""
    h, w = img.shape[:2]
    y_pos = h - 8  # 8 pixels from the bottom
    # dark box behind the text (a thick black outline drawn under the text does not line
    # up with newer OpenCV fonts and looks like doubled, overlapping letters)
    (tw, th_), bl = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x0, y0 = 6, y_pos - th_ - 4
    roi = img[max(0, y0):min(h, y_pos + bl + 2), x0:min(w, 10 + tw + 4)]
    roi[:] = (roi * 0.35).astype(roi.dtype)
    cv2.putText(img, t, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return img


def draw_top_lines(img, text, color, scale=0.48, bottom=False):
    """Text at the top of a panel, readable size, wrapped at double spaces / spaces to
    the panel width, on a dark band."""
    font, w = cv2.FONT_HERSHEY_SIMPLEX, img.shape[1]
    words = [p for p in text.replace("  ", "\x00").split("\x00") if p]
    lines, cur = [], ""
    for wd in words:
        cand = wd if not cur else cur + "  " + wd
        if cur and cv2.getTextSize(cand, font, scale, 1)[0][0] > w - 8:
            lines.append(cur); cur = wd
        else:
            cur = cand
    if cur:
        lines.append(cur)
    lh = int(20 * scale / 0.5)
    bh = lh * len(lines) + 6
    y0 = img.shape[0] - bh if bottom else 0
    band = img[y0:y0 + bh]
    band[:] = (band * 0.3).astype(band.dtype)
    for i, ln in enumerate(lines):
        cv2.putText(img, ln, (4, y0 + lh * (i + 1)), font, scale, color, 1, cv2.LINE_AA)
    return img


def add_status_strip(panel, status, first=None, scale=0.45):
    """Black strip under the panels with the status text, wrapped at ' | ' so a long
    status takes 2-3 lines instead of running off the window or over the panels.
    first = (text, colour): an extra line on top (the tactile summary)."""
    w = panel.shape[1]
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 1
    lines, cur = [], ""
    for part in status.split(" | "):
        cand = part if not cur else cur + " | " + part
        if cur and cv2.getTextSize(cand, font, scale, th)[0][0] > w - 16:
            lines.append(cur); cur = part
        else:
            cur = cand
    if cur:
        lines.append(cur)
    lines = [(ln, (255, 255, 255), scale) for ln in lines]
    if first is not None and first[0]:
        lines.insert(0, (first[0], first[1], 0.55))
    lh = int(22 * scale / 0.5)
    hs = [int(lh * sc / scale) for _, _, sc in lines]
    strip = np.zeros((sum(hs) + 8, w, 3), np.uint8)
    y = 0
    for (ln, col, sc), h_ in zip(lines, hs):
        y += h_
        cv2.putText(strip, ln, (8, y), font, sc, col, th, cv2.LINE_AA)
    return np.vstack([panel, strip])


def quiver_fast(img, flow, step, scale, thresh2=0.04):
    """Draw flow vectors only at grid points (cheap: O(grid), not O(pixels))."""
    h, w = flow.shape[:2]
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    fy = flow[gy, gx]                       # (ny, nx, 2)
    mag2 = fy[..., 0] ** 2 + fy[..., 1] ** 2
    sel = np.where(mag2 > thresh2)
    for r, c in zip(*sel):
        x = int(xs[c]); y = int(ys[r])
        dx, dy = fy[r, c]
        cv2.arrowedLine(img, (x, y), (int(x + dx * scale), int(y + dy * scale)),
                        (0, 255, 0), 1, tipLength=0.4)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Camera (Picamera2) + latest-frame reader thread
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#  Optical-flow method selection (--flow=...) without modifying deltact_plus.py
# ─────────────────────────────────────────────────────────────────────────────
# deltact_plus creates DIS via cv2.DISOpticalFlow_create(...) and then calls .calc().
# That function is swapped for a "shim" that runs the selected method behind the same
# interface (.calc; DIS setters are ignored so the chosen preset is not overwritten).
# Time per frame pair on the Pi 5, ROI 320x240 (bench_flow.py): ultrafast 0.8 ms, fast 2.8,
# medium 8.9, medium+ref 10.5, farneback 25, pcaflow 55. Static noise: medium 0.027 px
# (lowest practical), farneback 0.031, ultrafast 0.038, fast 0.045.
FLOW_METHODS = ("ultrafast", "fast", "medium", "medium_ref", "farneback", "pcaflow", "tvl1")
FLOW_USED = {"n_create": 0}


class _FlowShim:
    def __init__(self, method, orig_create):
        self.method = method
        P = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
             "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST}
        if method in ("ultrafast", "fast", "medium", "medium_ref"):
            self._impl = orig_create(P.get(method, cv2.DISOPTICAL_FLOW_PRESET_MEDIUM))
            if method == "medium_ref":
                self._impl.setVariationalRefinementIterations(10)
        elif method == "pcaflow":
            self._impl = cv2.optflow.createOptFlow_PCAFlow()
        elif method == "tvl1":
            self._impl = cv2.optflow.DualTVL1OpticalFlow_create()
        else:
            self._impl = None                       # farneback (a function, not an object)

    def calc(self, I0, I1, flow=None, *a, **k):
        if I0.ndim == 3:
            I0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
        if I1.ndim == 3:
            I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I0, I1 = np.ascontiguousarray(I0), np.ascontiguousarray(I1)
        if self.method == "farneback":
            return cv2.calcOpticalFlowFarneback(I0, I1, None, pyr_scale=0.5, levels=3,
                                                winsize=15, iterations=3, poly_n=5,
                                                poly_sigma=1.2, flags=0)
        return self._impl.calc(I0, I1, None)

    def __getattr__(self, name):                    # setPatchSize, setFinestScale, ... from deltact
        if name.startswith("set"):
            return lambda *a, **k: None             # ignored: the selected preset stays in effect
        if self._impl is not None and hasattr(self._impl, name):
            return getattr(self._impl, name)
        return lambda *a, **k: 0


def install_flow_method(method):
    """Replace deltact_plus's built-in DIS with the selected method. Call BEFORE TactileSensor is created."""
    if method in (None, "", "default"):
        return
    if method not in FLOW_METHODS:
        sys.exit(f"[ERROR] --flow={method} is unknown. Choices: {', '.join(FLOW_METHODS)}")
    if method in ("pcaflow", "tvl1") and not hasattr(cv2, "optflow"):
        sys.exit(f"[ERROR] --flow={method} needs opencv-contrib (cv2.optflow is missing).")
    orig = cv2.DISOpticalFlow_create

    def create(*a, **k):
        FLOW_USED["n_create"] += 1
        return _FlowShim(method, orig)
    cv2.DISOpticalFlow_create = create


# CHECKERBOARD lens calibration (ChromaSense patent deck, slide 28):
#   CSI IMX219 200 deg camera, 820x616 mode (full 3280 px sensor, 4x scale), 40 fps
#   pinhole + 2 radial terms, fixed focal 356.5 px (= 1426 / 4)
#   k1 = -0.857, k2 = 1.113, principal point (419.7, 307.3) px, corner RMS 0.71 px (~0.04 mm)
#   working ROI roi_undist = (270, 123, 271, 361) [x, y, w, h] -> 240x320 grid (~14.8 px/mm)
LENS_CHECKER = dict(ref_w=820, focal=356.5, k1=-0.857, k2=1.113, cx=419.7, cy=307.3,
                    roi_undist=(270, 123, 271, 361), sensor_w=3280)


def apply_lens_checker(cfg, cw, crop_w):
    """Apply the checkerboard calibration to cfg. Only valid in full-FOV mode (3280 px crop).
    -> list of the setting names that were actually applied."""
    L = LENS_CHECKER
    if crop_w is not None and abs(crop_w - L["sensor_w"]) > 8:
        print(f"[WARN] --lens=checker was made in full-FOV mode (crop {L['sensor_w']} px), "
              f"this profile crops {crop_w} px -> NOT applied. Use the 'full' profile.")
        return []
    s = cw / float(L["ref_w"])
    cfg.dist = (L["k1"], L["k2"], 0.0, 0.0, 0.0)
    cfg.focal = L["focal"] * s
    done = ["focal", "k1", "k2"]
    for a, v in (("cx", L["cx"] * s), ("cy", L["cy"] * s)):   # if deltact_plus supports it
        if hasattr(cfg, a):
            setattr(cfg, a, v); done.append(a)
    if hasattr(cfg, "roi_undist"):
        x, y, w, h = L["roi_undist"]
        cfg.roi_undist = tuple(int(round(t * s)) for t in (x, y, w, h)); done.append("roi_undist")
    return done


# ─────────────────────────────────────────────────────────────────────────────
#  Depth -> mm calibration (deck method, slides 26-27: "Depth Points 0.1 mm Iteration")
# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT depth -> mm calibration (used when no --depth-calib=FILE is given).
# SYNTHETIC / APPROXIMATE: produced by make_synth_calib.py from MTS files exp1_test1-3,
# anchored to deck Table VI (500 g -> depth 0.53) & Table VII (range 0.40-1.20 mm).
# Gain varies 0.50-0.99 between trials -> mm values may be off by ~+-30 %. Not a measurement.
# Disable: --depth-calib=none
DEFAULT_DEPTH_CALIB = dict(gain=0.723586, offset=0.041380, range_mm=[0.40, 1.20],
                           rmse_mm=0.0569, source="SYNTHETIC (make_synth_calib.py)")


class DepthCalib:
    """Step-wise calibration with an MTS / micrometer:
      hold the indenter at each step (0.1, 0.2, ... mm) -> press 'p' -> point recorded
      (median depth_max over the last second). 'z' = delete the last point.
      'o' = linear fit mm = gain * depth + offset, compute RMSE/MAE, save JSON.
    When in use (--depth-calib=file.json): depth_mm = gain*depth + offset inside the
    contact area (depth > depth_min), 0 outside contact.
    """

    def __init__(self, step_mm=0.1, metric="max", window=15):
        self.step, self.metric = float(step_mm), metric
        self.recent = collections.deque(maxlen=window)
        self.points = []                   # [(disp_mm, depth_med, depth_std)]
        self.gain = self.offset = None
        self.info = {}

    def observe(self, depth):
        if depth is None:
            return
        v = float(np.percentile(depth, 99)) if self.metric == "p99" else float(np.max(depth))
        self.recent.append(v)

    def add_point(self):
        if len(self.recent) < 3:
            print("[CAL] not enough depth frames yet; wait a moment"); return
        r = np.array(self.recent)
        disp = round((len(self.points) + 1) * self.step, 4)
        self.points.append((disp, float(np.median(r)), float(r.std())))
        print(f"[CAL] point {len(self.points)}: displacement {disp:.2f} mm <- depth_{self.metric} "
              f"{np.median(r):.4f} (std {r.std():.4f}, {len(r)} frame)")

    def undo(self):
        if self.points:
            d = self.points.pop(); print(f"[CAL] point {d[0]:.2f} mm deleted")

    def fit(self, path, lo=None, hi=None, meta=None):
        pts = [p for p in self.points if (lo is None or p[0] >= lo) and (hi is None or p[0] <= hi)]
        if len(pts) < 1:
            print("[CAL] no points yet (press 'p' first)"); return False
        x = np.array([p[1] for p in pts]); y = np.array([p[0] for p in pts])
        if len(pts) >= 3:
            gain, offset = np.polyfit(x, y, 1)
        else:                                       # 1-2 points: through the origin (rough estimate)
            gain, offset = float(np.sum(x * y) / max(np.sum(x * x), 1e-12)), 0.0
            print(f"[CAL] only {len(pts)} point(s) -> fit through origin (APPROXIMATE). "
                  f"Add points (>=3) for better accuracy.")
        pred = gain * x + offset
        rmse = float(np.sqrt(np.mean((pred - y) ** 2))); mae = float(np.mean(np.abs(pred - y)))
        r2 = float(1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))
        self.gain, self.offset = float(gain), float(offset)
        self.info = dict(metric=f"depth_{self.metric}", gain=self.gain, offset=self.offset,
                         rmse_mm=round(rmse, 4), mae_mm=round(mae, 4), r2=round(r2, 4),
                         range_mm=[float(y.min()), float(y.max())], n_points=len(pts),
                         step_mm=self.step, points=self.points,
                         time=time.strftime("%Y-%m-%d %H:%M:%S"), **(meta or {}))
        with open(path, "w") as fh:
            json.dump(self.info, fh, indent=2)
        print(f"[CAL] fit: mm = {gain:.4f} * depth + {offset:+.4f} | RMSE {rmse:.3f} mm, "
              f"MAE {mae:.3f} mm, R2 {r2:.4f} ({len(pts)} points, {y.min():.2f}-{y.max():.2f} mm) "
              f"-> {path}  (deck: RMSE 0.069 mm)")
        return True

    def load_default(self):
        self.info = dict(DEFAULT_DEPTH_CALIB)
        self.gain, self.offset = float(self.info["gain"]), float(self.info["offset"])
        print(f"[CAL] DEFAULT depth -> mm (SYNTHETIC, approximate): mm = {self.gain:.4f} * depth "
              f"{self.offset:+.4f}, range {self.info['range_mm']} mm. Disable: --depth-calib=none")

    def load(self, path):
        with open(path) as fh:
            self.info = json.load(fh)
        self.gain, self.offset = float(self.info["gain"]), float(self.info["offset"])
        print(f"[CAL] depth -> mm: {path} | mm = {self.gain:.4f} * depth + {self.offset:+.4f} "
              f"(RMSE {self.info.get('rmse_mm')} mm, range {self.info.get('range_mm')} mm)")

    def to_mm(self, depth, depth_min):
        if depth is None or self.gain is None:
            return depth
        out = self.gain * depth.astype(np.float32) + self.offset
        out[depth <= depth_min] = 0.0               # outside contact: 0 mm
        return np.clip(out, 0.0, None)


# ─────────────────────────────────────────────────────────────────────────────
#  Depth refinement ('f' = on/off, --depth-refine=off): post-processing of the
#  deltact_plus depth map so the reconstructed surface is solid and stable.
# ─────────────────────────────────────────────────────────────────────────────
class DepthRefiner:
    """Cleans the raw depth map from deltact_plus, frame by frame:

      1. spike removal   5x5 median -> isolated wrong pixels (bad flow) are dropped
      2. floor removal   median of the non-contact area is subtracted -> no slow
                         offset/drift under the whole surface
      3. temporal filter motion-adaptive per-pixel EMA: where depth is static the
                         noise is averaged (alpha_min), where it really changes
                         (> ~3 sigma) the new value is taken at once -> no lag
      4. spatial filter  edge-preserving bilateral (range = 3 sigma of noise)
      5. contact mask    hysteresis (weak = depth_min, strong = 2x depth_min) +
                         drop blobs smaller than min_area -> no floating islands
      6. hole filling    holes inside a contact are filled by normalised
                         convolution of the surrounding depth
      7. feathered edge  soft mask edge instead of a hard 0/1 step

    The noise level sigma is estimated online from frame-to-frame differences in
    the non-contact area (robust MAD), so all thresholds follow the real sensor.
    Works on any depth scale (relative units or mm), only depth_min sets the scale.
    """

    def __init__(self, alpha_min=0.2, min_area_frac=0.002, bilateral=True,
                 fill_sigma=3.0, feather=1.2):
        self.alpha_min = float(alpha_min)
        self.min_area_frac = float(min_area_frac)
        self.bilateral = bilateral
        self.fill_sigma = float(fill_sigma)
        self.feather = float(feather)
        self.on = True
        self.reset()

    def reset(self):
        self.prev = None          # last filtered depth (before masking)
        self.prev_raw = None      # last raw depth after spike removal (noise estimate)
        self.bg = None            # last non-contact mask
        self.sigma = None         # running noise estimate (depth units)
        self.floor = 0.0
        self.ms = 0.0
        self.n_blobs = 0

    def __call__(self, depth, depth_min):
        if depth is None:
            return None
        t0 = time.perf_counter()
        d = np.nan_to_num(depth.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        if self.prev is not None and self.prev.shape != d.shape:
            self.reset()
        lo = max(float(depth_min), 1e-6)

        # 1. spike removal
        d = cv2.medianBlur(d, 5)

        # 2. floor (offset) removal from the previous non-contact area
        bg = (d < 2.0 * lo) if self.bg is None else (self.bg & (d < 2.0 * lo))
        if bg.mean() > 0.2:
            self.floor = 0.9 * self.floor + 0.1 * float(np.median(d[::4, ::4][bg[::4, ::4]]))
            d = d - self.floor

        # noise estimate: robust std of frame-to-frame difference in the background
        if self.prev_raw is not None and bg.mean() > 0.2:
            diff = (d[::4, ::4] - self.prev_raw[::4, ::4])[bg[::4, ::4]]
            s = 1.4826 * float(np.median(np.abs(diff - np.median(diff)))) / 1.41421356
            s = max(s, 1e-5)
            self.sigma = s if self.sigma is None else 0.95 * self.sigma + 0.05 * s
        self.prev_raw = d
        sig = self.sigma if self.sigma is not None else 0.1 * lo

        # 3. motion-adaptive temporal filter
        if self.prev is not None:
            tau = 3.0 * sig
            a = np.clip(np.abs(d - self.prev) / tau, 0.0, 1.0) ** 2
            a = np.maximum(a, self.alpha_min)
            d = (self.prev + a * (d - self.prev)).astype(np.float32)
        self.prev = d

        # 4. edge-preserving spatial smoothing
        if self.bilateral:
            d = cv2.bilateralFilter(np.ascontiguousarray(d, np.float32), 5, max(3.0 * sig, 1e-4), 2.0)

        # 5. contact mask: hysteresis + minimum blob size
        hi = max(2.0 * lo, lo + 4.0 * sig)
        weak = (d > lo).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
        mask = np.zeros(d.shape, np.uint8)
        min_area = max(4, int(self.min_area_frac * d.size))
        keep = []
        if n > 1:
            strong_lab = np.flatnonzero(np.bincount(lab[d > hi], minlength=n))
            for i in strong_lab:
                if i != 0 and stats[i, cv2.CC_STAT_AREA] >= min_area:
                    keep.append(i)
            if keep:
                lut = np.zeros(n, np.uint8); lut[keep] = 1
                mask = lut[lab]
        self.n_blobs = len(keep)
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

        # 6. fill holes inside the contact (normalised convolution)
        holes = (closed > 0) & ~(mask > 0)
        if holes.any():
            m = mask.astype(np.float32)
            num = cv2.GaussianBlur(d * m, (0, 0), self.fill_sigma)
            den = cv2.GaussianBlur(m, (0, 0), self.fill_sigma)
            d = d.copy()
            d[holes] = num[holes] / np.maximum(den[holes], 1e-6)
        self.bg = cv2.dilate(closed, k5) == 0

        # 7. feathered mask -> smooth surface edge
        mf = closed.astype(np.float32)
        if self.feather > 0:
            mf = np.minimum(cv2.GaussianBlur(mf, (0, 0), self.feather) * 1.5, 1.0) * (
                cv2.dilate(closed, k5) > 0)
        out = np.clip(d, 0.0, None) * mf
        self.ms = (time.perf_counter() - t0) * 1000.0
        return out.astype(np.float32)


def csi_cameras():
    """Picamera2 indices of CSI cameras ONLY.

    Newer libcamera (0.7+) also lists USB/UVC webcams (e.g. the Pi Zero) as cameras,
    with an Id containing 'usb@'. Those cannot be opened as CSI cameras (no
    FrameRate/FrameDurationLimits), so they are filtered out here.
    """
    if Picamera2 is None:
        return []
    return [i for i, c in enumerate(Picamera2.global_camera_info())
            if "usb@" not in str(c.get("Id", "")).lower()]


def open_camera(cam_num, cw, ch, fr, raw=None):
    picam2 = Picamera2(cam_num)
    kw = {}
    if raw is not None:
        # force a specific sensor mode (e.g. 1640x1232 = full FOV, 2x2 binned)
        kw["raw"] = {"size": tuple(raw)}
    cfg = picam2.create_video_configuration(
        main={"size": (cw, ch), "format": "RGB888"},  # RGB888 -> BGR byte order in numpy (ready for OpenCV)
        controls={"FrameRate": float(fr)},
        buffer_count=4,                                 # smoother pipeline at high FPS
        **kw,
    )
    picam2.configure(cfg)
    picam2.start()
    return picam2


def _jsonable(v):
    try:
        json.dumps(v); return v
    except TypeError:
        return str(v)


def save_raw_bundle(folder, idx, frame, res):
    """Save the raw camera frame + raw ROI (no overlay/display gain)."""
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, f"raw_{idx:02d}.png"), frame)
    roi = None if res is None else res.get("roi")
    if roi is not None:
        cv2.imwrite(os.path.join(folder, f"roi_{idx:02d}.png"), roi)
    return roi is not None


def frame_record(seq, md, frame, box):
    """Per-frame record: actual exposure/gain + mean brightness inside the window."""
    if box is not None:
        x0, y0, x1, y1 = box; roi = frame[y0:y1, x0:x1]
    else:
        roi = frame
    bgr = roi.reshape(-1, 3).mean(0)
    rec = dict(seq=int(seq), mean_rgb=[round(float(bgr[2]) / 255, 5), round(float(bgr[1]) / 255, 5),
                                       round(float(bgr[0]) / 255, 5)])
    if md:
        for k in ("ExposureTime", "AnalogueGain", "DigitalGain", "Lux", "SensorTimestamp", "ColourGains"):
            if k in md:
                rec[k] = _jsonable(md[k])
    return rec


def summarize_frames(recs):
    lum = np.array([0.299 * r["mean_rgb"][0] + 0.587 * r["mean_rgb"][1] + 0.114 * r["mean_rgb"][2] for r in recs])
    out = dict(brightness_drift_pct=round(float(np.ptp(lum) / lum.mean() * 100), 3))
    for k in ("ExposureTime", "AnalogueGain", "DigitalGain"):
        v = [r[k] for r in recs if k in r and isinstance(r[k], (int, float))]
        if v:
            out[k + "_range"] = [min(v), max(v)]
    ok = out["brightness_drift_pct"] <= 0.3
    for k in ("ExposureTime_range", "AnalogueGain_range"):
        if k in out and out[k][0] != out[k][1]:
            ok = False
    out["stable"] = bool(ok)
    return out


def write_meta(folder, cam, cfg, profile_name, profile, ae_locked, measure_mode,
               n_saved, seqs, note="", frames=None):
    md = cam.get_metadata()
    keys = ("ExposureTime", "AnalogueGain", "DigitalGain", "ColourGains",
            "ColourTemperature", "Lux", "SensorTemperature", "FrameDuration", "ScalerCrop",
            "NoiseReductionMode", "Sharpness", "AeLocked")
    meta = dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"), note=note,
        n_frames=n_saved, frame_seq=seqs,
        ae_locked=ae_locked, measure_mode=measure_mode,
        measure_point=dict(exposure_us=MEASURE_EXP_US, gain=MEASURE_GAIN,
                           colour_gains=MEASURE_COLOUR_GAINS),
        camera_metadata={k: _jsonable(md.get(k)) for k in keys if k in md},
        profile=profile_name, capture_size=[profile["cw"], profile["ch"]],
        fps_target=profile["fr"],
        undistort=dict(focal=float(cfg.focal), dist=list(cfg.dist),
                       roi_thresh=int(cfg.roi_thresh), valid_margin=int(cfg.valid_margin)),
        work_size=[int(cfg.work_w), int(cfg.work_h)],
        window_box=None if WINDOW_BOX is None else list(WINDOW_BOX),
        burst_skip=BURST_SKIP,
        frames=frames,
        frames_summary=None if not frames else summarize_frames(frames),
        files="raw_XX.png = raw camera frame (BGR, fisheye); "
              "roi_XX.png = raw deltact_plus ROI (undistort+crop+resize)")
    with open(os.path.join(folder, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    fsum = meta.get("frames_summary")
    if fsum:
        print(f"[SNAP] burst check: brightness drift {fsum['brightness_drift_pct']} %, "
              f"exposure {fsum.get('ExposureTime_range')}, gain {fsum.get('AnalogueGain_range')} -> "
              f"{'STABLE' if fsum['stable'] else 'NOT STABLE — REPEAT the burst (press b again)'}")
    if not measure_mode:
        print("[SNAP] WARNING: not in measure mode ('u'); exposure/AWB/NR may "
              "differ from the measurement operating point.")


class CameraReader:
    """Always keeps the LATEST frame (old frames dropped) -> capture never blocks processing.

    Each frame gets a sequence number (seq). The main loop skips frames whose seq
    equals the previous one: if the loop is faster than the camera, the same frame
    would be processed twice and yield a fake ~0 flow, lowering the mean ||F||
    without any real improvement.
    """
    def __init__(self, picam2, name="cam"):
        self._picam2 = picam2
        self._frame = None
        self._md = None
        self._seq = 0
        self._lock = threading.Condition()   # notify on every new frame
        self._ok = False
        self._running = False
        self._t = threading.Thread(target=self._run, daemon=True, name=name)

    def start(self):
        self._running = True
        self._t.start()

    def stop(self):
        self._running = False
        self._t.join(timeout=2)
        try:
            self._picam2.stop(); self._picam2.close()
        except Exception:
            pass

    def wait_new(self, last_seq, timeout=0.1):
        """Block until a frame newer than last_seq arrives (or timeout). No polling:
        polling threads steal the GIL from the processing thread hundreds of times per
        frame (measured on a Pi 5: 25 ms -> 110-250 ms per frame)."""
        with self._lock:
            if not self._ok or self._seq == last_seq:
                self._lock.wait(timeout)
            return self._ok, (self._frame if self._ok else None), self._seq

    def read(self):
        """-> (ok, frame, seq)"""
        with self._lock:
            return self._ok, (self._frame if self._ok else None), self._seq

    def get_metadata(self):
        """Values CURRENTLY used by the sensor."""
        try:
            return self._picam2.capture_metadata()
        except Exception:
            return {}

    def set_controls(self, **kw):
        try:
            self._picam2.set_controls(kw)
        except Exception as e:
            print(f"[WARN] set_controls failed: {e}")

    def frame_metadata(self):
        """Metadata of the LAST captured frame (per frame, not 'current')."""
        with self._lock:
            return self._seq, (dict(self._md) if self._md else None)

    def _run(self):
        use_req = hasattr(self._picam2, "capture_request")
        while self._running:
            md = None
            try:
                if use_req:
                    req = self._picam2.capture_request()
                    try:
                        f = req.make_array("main"); md = req.get_metadata()
                    finally:
                        req.release()
                else:
                    f = self._picam2.capture_array("main")
            except Exception:
                use_req = False
                continue
            if f is not None:
                with self._lock:
                    self._frame = f
                    self._md = md
                    self._seq += 1
                    self._ok = True
                    self._lock.notify_all()


def find_usb_camera(name=None):
    """Find a Pi Zero webcam ('Tactile UVC Cam') via its stable by-id path.

    The by-id name contains the Zero's hostname (= USB manufacturer string), e.g.
      /dev/v4l/by-id/usb-camera_Tactile_UVC_Cam_<serial>-video-index0
      /dev/v4l/by-id/usb-camera2_Tactile_UVC_Cam_<serial>-video-index0
    name=None -> first one found; name='camera2' -> that Zero only."""
    import glob
    if name:
        hits = sorted(glob.glob(f"/dev/v4l/by-id/usb-{name}_Tactile*video-index0"))
        if not hits:
            found = [os.path.basename(h) for h in glob.glob("/dev/v4l/by-id/*Tactile*video-index0")]
            sys.exit(f"[ERROR] USB camera '{name}' not found. Found: {found or 'none'}")
        return hits[0]
    hits = sorted(glob.glob("/dev/v4l/by-id/*Tactile*video-index0"),     # camera, camera2, ...
                  key=lambda h: os.path.basename(h).split("_Tactile")[0])
    if not hits:
        hits = sorted(glob.glob("/dev/v4l/by-id/*video-index0"))
    if not hits:
        sys.exit("[ERROR] USB camera not found. Check: v4l2-ctl --list-devices")
    if len(hits) > 1:
        print(f"[INFO] {len(hits)} USB cameras found, using the first. "
              f"Pick one with --usb=<hostname> (e.g. --usb=camera2): "
              f"{[os.path.basename(h) for h in hits]}")
    return hits[0]


def resolve_usb(spec=None):
    """--usb value -> device path. None = auto, '/dev/...' = as given, else = Zero hostname."""
    if spec and spec.startswith("/dev/"):
        return spec
    return find_usb_camera(spec or None)


def wait_frame(reader, last_seq, timeout=0.1):
    """Next frame newer than last_seq, sleeping until it arrives (event, not polling)."""
    w = getattr(reader, "wait_new", None)
    if w is not None:
        return w(last_seq, timeout)
    r = reader.read()
    if r[2] == last_seq:
        time.sleep(0.005)
    return r


def usb_frame_is_glitch(f, frac=0.005):
    """Corrupt UVC frame: missing/zero YUYV data shows up as GREEN
    (Y=U=V=0 -> BGR ~ (0, 135, 0)). A normal speckle image almost never has pure
    green pixels like this, so > 0.5% such pixels = frame is discarded."""
    s = f[::8, ::8]
    g = (s[..., 0] < 20) & (s[..., 2] < 20) & (s[..., 1] > 60)
    return float(g.mean()) > frac


class UsbCameraReader:
    """CameraReader replacement for a UVC webcam (Pi Zero as a USB camera).

    Same interface (read / get_metadata / set_controls / frame_metadata), but:
      - no sensor metadata (exposure, gain, ScalerCrop, temperature) -> {} / None
      - exposure/gain/AWB/NR CANNOT be set from here; they are all set by
        uvc-gadget on the Zero. Keys e / u / [ ] / - only print a warning.
    """
    def __init__(self, dev, w, h, fps, name="usbcam"):
        self.dev = dev
        self._req = (w, h, fps)
        self._cap = self._open(dev)
        if self._cap is None:
            sys.exit(f"[ERROR] cannot open USB camera {dev}")
        self.n_reconnect = 0
        self.size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._frame = None
        self._seq = 0
        self._ok = False
        self._lock = threading.Condition()   # notify on every new frame
        self._running = False
        self._warned = False
        self.n_glitch = 0          # number of green/corrupt frames discarded
        self._t = threading.Thread(target=self._run, daemon=True, name=name)

    def _open(self, dev):
        w, h, fps = self._req
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release(); return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))  # uncompressed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, USB_BUFFERS)   # more V4L2 buffers: no drops on jitter
        return cap

    def _reconnect(self):
        """The Zero dropped off USB (power dip / gadget restart): wait until it is back
        and reopen it, instead of failing forever. Frame seq keeps counting."""
        import glob
        try:
            self._cap.release()
        except Exception:
            pass
        print("[WARN] USB camera lost (Zero disconnected?) -> waiting to reconnect... "
              "check on the Pi 5: dmesg | tail -20")
        t0 = time.time()
        while self._running:
            if os.path.exists(self.dev):
                cands = [self.dev]
            elif "/by-id/" in self.dev:
                cands = []                         # stable name: wait for THIS camera only
            else:
                cands = sorted(glob.glob("/dev/v4l/by-id/*Tactile*video-index0"))
            if cands:
                time.sleep(1.0)                    # let the gadget finish starting
                cap = self._open(cands[0])
                if cap is not None:
                    ok, f = cap.read()
                    if ok and f is not None:
                        self._cap, self.dev = cap, cands[0]
                        self.n_reconnect += 1
                        print(f"[INFO] USB camera RECONNECTED after {time.time() - t0:.1f} s "
                              f"(reconnect #{self.n_reconnect}). Press 'r' for a new baseline.")
                        return True
                    cap.release()
            time.sleep(0.5)
        return False

    def start(self):
        self._running = True
        self._t.start()

    def stop(self):
        self._running = False
        self._t.join(timeout=2)
        try:
            self._cap.release()
        except Exception:
            pass

    def wait_new(self, last_seq, timeout=0.1):
        """Block until a frame newer than last_seq arrives (or timeout). No polling:
        polling threads steal the GIL from the processing thread hundreds of times per
        frame (measured on a Pi 5: 25 ms -> 110-250 ms per frame)."""
        with self._lock:
            if not self._ok or self._seq == last_seq:
                self._lock.wait(timeout)
            return self._ok, (self._frame if self._ok else None), self._seq

    def read(self):
        with self._lock:
            return self._ok, (self._frame if self._ok else None), self._seq

    def get_metadata(self):
        return {}

    def frame_metadata(self):
        with self._lock:
            return self._seq, None

    def set_controls(self, **kw):
        if not self._warned:
            print("[WARN] USB camera: exposure/gain/AWB/NR cannot be set from the Pi 5 "
                  "(set by uvc-gadget on the Zero). Camera command ignored.")
            self._warned = True

    def _run(self):
        fails = 0
        while self._running:
            ok, f = self._cap.read()
            if not ok or f is None:
                fails += 1
                if fails == 5:
                    print(f"[WARN] USB camera: {fails} failed frames.")
                if fails >= 30 or (str(self.dev).startswith("/dev/") and not os.path.exists(self.dev)):
                    with self._lock:
                        self._ok = False          # main loop waits instead of using a stale frame
                    self._reconnect(); fails = 0
                    continue
                time.sleep(0.01)
                continue
            fails = 0
            if usb_frame_is_glitch(f):
                self.n_glitch += 1
                if self.n_glitch in (1, 10, 100) or self.n_glitch % 500 == 0:
                    print(f"[WARN] USB camera: corrupt (green) frame discarded, total {self.n_glitch}")
                continue                      # never let it reach flow/depth/slip
            with self._lock:
                self._frame = f
                self._seq += 1
                self._ok = True
                self._lock.notify_all()


# ─────────────────────────────────────────────────────────────────────────────
#  Raw frame recording (RAM / disk, pre-trigger) — also used by record_dual.py & dual
def mem_available_bytes():
    """RAM still available (MemAvailable from /proc/meminfo)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class Recorder(threading.Thread):
    """Grab EVERY new frame from the reader (seq increases) and save it.

    Disk mode (default): frame -> queue -> writer thread -> disk, while recording.
    RAM mode (ram=True): frames are kept in RAM while recording (no disk I/O
        -> no frames lost to a slow disk), then written to disk AFTER stop.
        Recording stops automatically when free RAM drops below ram_reserve.
    Pre-trigger (pre_s > 0): the last N seconds are ALWAYS kept in RAM and are
        included when recording starts -> events just before the key press are kept.
    """

    def __init__(self, name, reader, has_md, fmt, is_usb, ram=False, pre_s=0.0,
                 ram_reserve=1.5e9):
        super().__init__(daemon=True, name=f"rec-{name}")
        self.name_, self.reader, self.has_md = name, reader, has_md
        self.fmt, self.is_usb = fmt, is_usb
        self.ram, self.pre_s, self.ram_reserve = ram, float(pre_s), ram_reserve
        self.running = True
        self.recording = False
        self._start_req = False
        self.q = queue.Queue(maxsize=400)       # disk-mode buffer
        self.prebuf = collections.deque()       # [(t_host, seq, t_sensor, frame)] pre-trigger
        self.ram_frames = []                    # [(fname, frame)] RAM mode
        self.out_dir = None
        self.rows, self.rows_lock = [], threading.Lock()
        self.auto_stopped = False
        self.reset_stats()
        self.writer = threading.Thread(target=self._write_loop, daemon=True)
        self.writer.start()
        self.live_fps, self.latest = 0.0, None

    def reset_stats(self):
        self.idx = 0
        self.n_saved = self.n_gap = self.n_qfull = self.n_glitch = self.n_pre = 0
        self.t_first = self.t_last = None

    def start_rec(self, folder):
        self.out_dir = os.path.join(folder, self.name_)
        os.makedirs(self.out_dir, exist_ok=True)
        with self.rows_lock:
            self.rows = []
        self.ram_frames = []
        self.reset_stats()
        self.auto_stopped = False
        self._start_req = True                  # the recorder thread moves the pre-buffer

    def stop_rec(self):
        self.recording = False
        self._start_req = False
        if self.ram:
            frames, self.ram_frames = self.ram_frames, []
            n = len(frames)
            for k, (fname, frame) in enumerate(frames):
                self._save(fname, frame)
                if n > 300 and (k + 1) % 300 == 0:
                    print(f"[REC] {self.name_}: writing {k + 1}/{n} frames from RAM to disk...")
            del frames
        else:
            self.q.join()                        # wait until all frames are written
        with self.rows_lock:
            return list(self.rows)

    def _store(self, t_host, seq, t_sensor, frame):
        fname = f"{self.idx:06d}.{self.fmt}"
        if self.ram:
            self.ram_frames.append((fname, frame))
        else:
            try:
                self.q.put_nowait((fname, frame))
            except queue.Full:
                self.n_qfull += 1; return            # disk too slow
        with self.rows_lock:
            self.rows.append([self.name_, self.idx, int(seq), f"{t_host:.6f}",
                              t_sensor, f"{self.name_}/{fname}"])
        self.t_first = self.t_first or t_host
        self.t_last = t_host
        self.idx += 1
        if self.ram and self.idx % 50 == 0:
            avail = mem_available_bytes()
            if avail is not None and avail < self.ram_reserve:
                self.recording = False; self.auto_stopped = True
                print(f"[REC] {self.name_}: {avail / 1e9:.1f} GB RAM left -> recording "
                      f"STOPPED automatically ({self.idx} frames). Press the record key again to save.")

    def run(self):
        last_seq, n, t0 = None, 0, time.time()
        while self.running:
            ok, frame, seq = wait_frame(self.reader, last_seq)
            if not ok or seq == last_seq:
                continue
            t_host = time.monotonic()
            if last_seq is not None and self.recording and seq > last_seq + 1:
                self.n_gap += seq - last_seq - 1      # camera frames that were missed
            last_seq = seq
            self.latest = frame
            n += 1
            if time.time() - t0 >= 1.0:
                self.live_fps, n, t0 = n / (time.time() - t0), 0, time.time()

            if self._start_req:                       # start recording: pre-buffer goes in first
                for item in self.prebuf:
                    self._store(*item)
                self.n_pre = len(self.prebuf)
                self.prebuf = collections.deque()
                self._start_req = False
                self.recording = True
            if not self.recording and self.pre_s <= 0:
                continue
            if self.is_usb and usb_frame_is_glitch(frame):
                if self.recording:
                    self.n_glitch += 1
                continue
            t_sensor = ""
            if self.has_md:
                mseq, md = self.reader.frame_metadata()
                if mseq == seq and md:
                    t_sensor = md.get("SensorTimestamp", "")
            if self.recording:
                self._store(t_host, seq, t_sensor, frame.copy())
            elif not self.auto_stopped:               # pre-trigger: keep the last N seconds
                self.prebuf.append((t_host, seq, t_sensor, frame.copy()))
                while self.prebuf and t_host - self.prebuf[0][0] > self.pre_s:
                    self.prebuf.popleft()

    def _save(self, fname, frame):
        path = os.path.join(self.out_dir, fname)
        if self.fmt == "npy":
            np.save(path, frame)
        else:
            cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        self.n_saved += 1

    def _write_loop(self):
        while True:
            fname, frame = self.q.get()
            try:
                self._save(fname, frame)
            finally:
                self.q.task_done()

    def ram_used_mb(self):
        if not self.ram_frames:
            return 0.0
        return len(self.ram_frames) * self.ram_frames[0][1].nbytes / 1e6

    def summary(self):
        dur = (self.t_last - self.t_first) if self.t_first and self.t_last else 0.0
        return dict(frames=self.n_saved, duration_s=round(dur, 3),
                    fps=round((self.idx - 1) / dur, 2) if dur > 0 else 0.0,
                    camera_gaps=self.n_gap, dropped_disk=self.n_qfull,
                    dropped_glitch=self.n_glitch, pre_trigger_frames=self.n_pre,
                    mode="ram" if self.ram else "disk", auto_stopped_low_ram=self.auto_stopped)


# ─────────────────────────────────────────────────────────────────────────────
def finish_recording(folder, recs, meta_cams, fmt, ae_locked=False, extra=None):
    """Stop all Recorders TOGETHER, write frames.csv + meta.json, print a summary."""
    for r in recs:                     # stop ALL cameras first, then wait for the disk
        r.recording = False
    rows = []
    for r in recs:
        rows += r.stop_rec()
    rows.sort(key=lambda x: float(x[3]))             # sort by time: cameras interleaved
    with open(os.path.join(folder, "frames.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cam", "idx", "seq", "t_host", "t_sensor_ns", "file"])
        w.writerows(rows)
    meta = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), format=fmt,
                cameras=meta_cams, ae_locked_csi=ae_locked,
                result={r.name_: r.summary() for r in recs},
                note="t_host = time.monotonic() shared by all cameras")
    if extra:
        meta.update(extra)
    with open(os.path.join(folder, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[REC] STOP -> {folder}/")
    out = {}
    for r in recs:
        s = r.summary(); out[r.name_] = s
        warn = ""
        if s["dropped_disk"]:
            warn += f"  !! {s['dropped_disk']} frames lost (slow disk: use --out=SSD or --fmt=npy)"
        print(f"      {r.name_}: {s['frames']} frames, {s['fps']} fps, "
              f"camera gaps {s['camera_gaps']}, green discarded {s['dropped_glitch']}{warn}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  PARALLEL depth ('pardepth'): depth is computed in a separate PROCESS (another CPU core)
# ─────────────────────────────────────────────────────────────────────────────
# The main loop computes the flow (cfg.enable_depth=False during process) and sends THAT
# flow to the depth process, which only does Gaussian density -> guided filter -> mask.
# Depth is therefore always computed from exactly the same flow/baseline that is shown
# (previously the depth process tracked the frames itself with its own baseline, so flow
# and depth could disagree). Flow/slip never wait for depth; depth lags by 1-2 frames.
def _depth_worker(in_q, out_q, cfg, cw, ch, flow_method, refine_on=True, n_threads=2,
                  use_gpu=False):
    parent = os.getppid()                        # exit when the parent dies (no orphans)
    try:
        cv2.setNumThreads(int(n_threads))        # leave cores for the main loop (flow)
        install_flow_method(flow_method)
        dest = None                              # depth from the MAIN loop's flow
        if use_gpu:                              # 'gpu': CuPy on an NVIDIA GPU (Jetson)
            try:
                import gpu_depth
                dest = gpu_depth.make(cfg)       # checked against the CPU, else refused
            except BaseException as e:           # any GPU/CuPy problem -> CPU, depth never stops
                dest = None
                print(f"[GPU] not used ({type(e).__name__}: {e}) -> depth on the CPU. "
                      f"Jetson: .venv/bin/pip install cupy-cuda12x", flush=True)
        if dest is None:
            dest = dp.DepthEstimator(cfg)
        refiner = DepthRefiner(); refiner.on = bool(refine_on)
    except Exception as e:                       # report to the main process
        out_q.put(("error", repr(e))); return
    while True:
        try:
            msgs = [in_q.get(timeout=1.0)]
        except queue.Empty:
            if os.getppid() != parent:           # the program that started us is gone
                return
            continue
        while True:                              # take all queued messages
            try:
                msgs.append(in_q.get_nowait())
            except queue.Empty:
                break
        frame_msg = None
        for m in msgs:
            if m[0] == "stop":
                out_q.cancel_join_thread()       # do not hang on exit
                return
            if m[0] == "flow":
                frame_msg = m                    # only the last flow is processed
            elif m[0] == "cfg":
                cfg = m[1]; cfg.enable_depth = True; dest.cfg = cfg
            elif m[0] == "cmd":
                if m[1] in ("reset", "redetect_roi"):
                    dest.baseline = None; refiner.reset()
            elif m[0] == "refine":
                refiner.on = bool(m[1]); refiner.reset()
        if frame_msg is None:
            continue
        t0 = time.perf_counter()
        flow, wmask = frame_msg[2], frame_msg[3]
        depth = dest.compute(flow)               # same steps as deltact_plus TactileSensor
        if wmask is not None:
            depth = depth * wmask.astype(np.float32)
        if cfg.depth_min > 0:
            depth[depth < cfg.depth_min] = 0.0
        if cfg.depth_open > 1:
            kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.depth_open, cfg.depth_open))
            depth = cv2.morphologyEx(depth, cv2.MORPH_OPEN, kk)
        t1 = time.perf_counter()
        if depth is not None and refiner.on:
            depth = refiner(depth, cfg.depth_min)   # refinement also runs off the main loop
        out = ("depth", frame_msg[1], depth, t1 - t0,
               refiner.ms if refiner.on else 0.0)
        try:
            out_q.put_nowait(out)
        except queue.Full:
            try:
                out_q.get_nowait()
            except queue.Empty:
                pass
            out_q.put_nowait(out)


class DepthWorker:
    def __init__(self, cfg, cw, ch, flow_method=None, refine_on=True, n_threads=2,
                 use_gpu=False):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")            # safe with camera threads (not fork)
        self.in_q, self.out_q = ctx.Queue(), ctx.Queue(maxsize=2)
        c = copy.copy(cfg); c.enable_depth = True
        self.proc = ctx.Process(target=_depth_worker, daemon=True,
                                args=(self.in_q, self.out_q, c, cw, ch, flow_method,
                                      refine_on, n_threads, use_gpu))
        self.proc.start()
        self.busy = False
        self.depth, self.seq, self.ms, self.refine_ms = None, None, 0.0, 0.0
        self.n, self.t0, self.fps = 0, time.time(), 0.0

    def submit(self, seq, flow, wmask=None):
        """Send the main loop's flow only when the depth process is idle (always the latest)."""
        if not self.busy:
            self.in_q.put(("flow", seq, flow, wmask)); self.busy = True

    def cmd(self, name):
        self.in_q.put(("cmd", name))

    def set_cfg(self, cfg):
        self.in_q.put(("cfg", copy.copy(cfg)))

    def set_refine(self, on):
        self.in_q.put(("refine", bool(on)))

    def poll(self):
        """-> latest depth (or None if none yet)."""
        while True:
            try:
                m = self.out_q.get_nowait()
            except queue.Empty:
                break
            if m[0] == "error":
                print(f"[ERROR] depth process failed: {m[1]}"); self.busy = False; continue
            _, self.seq, self.depth, dt, self.refine_ms = m
            self.ms = dt * 1000.0; self.busy = False
            self.n += 1
            if time.time() - self.t0 >= 1.0:
                self.fps, self.n, self.t0 = self.n / (time.time() - self.t0), 0, time.time()
        return self.depth

    def stop(self):
        try:
            self.in_q.put(("stop",))
            t_end = time.time() + 2.0
            while self.proc.is_alive() and time.time() < t_end:
                try:
                    self.out_q.get(timeout=0.05)   # drain so the process can exit
                except queue.Empty:
                    pass
        finally:
            if self.proc.is_alive():
                self.proc.terminate()
            self.proc.join(timeout=1)


def gpu_report():
    """Check which acceleration OpenCV can use (OpenCL / CUDA)."""
    out = []
    try:
        out.append(f"OpenCL={'yes' if cv2.ocl.haveOpenCL() else 'no'}")
    except Exception:
        out.append("OpenCL=no")
    try:
        n = cv2.cuda.getCudaEnabledDeviceCount()
        out.append(f"CUDA={n} device")
    except Exception:
        out.append("CUDA=no")
    out.append(f"CPU threads OpenCV={cv2.getNumThreads()}")
    return ", ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
#  Noise floor test (through the same deltact_plus pipeline)
# ─────────────────────────────────────────────────────────────────────────────
NF_BINS = 32
NF_COHERENT_SIGMA = 16.0   # px on the working grid, ~2x the DIS patch


def _flow_stats(flow: np.ndarray, tau: float) -> tuple[float, float, float, float]:
    """-> (||F|| mean, % pixels |F|>tau, global shift, coherent ||F||)"""
    fl = flow.astype(np.float32)
    m = np.linalg.norm(fl, axis=2)
    shift = float(np.hypot(fl[..., 0].mean(), fl[..., 1].mean()))
    sx = cv2.GaussianBlur(fl[..., 0], (0, 0), NF_COHERENT_SIGMA, borderType=cv2.BORDER_REFLECT)
    sy = cv2.GaussianBlur(fl[..., 1], (0, 0), NF_COHERENT_SIGMA, borderType=cv2.BORDER_REFLECT)
    coh = float(np.hypot(sx, sy).mean())
    return float(m.mean()), 100.0 * float((m > tau).mean()), shift, coh


def _run_sequence(cfg, cw, ch, frames, warmup: int, tau: float) -> np.ndarray:
    """Run frames through a NEW TactileSensor (baseline from the first frame)."""
    s = dp.TactileSensor(cfg, cw, ch)
    out = []
    for k, f in enumerate(frames):
        res = s.process(f)
        if res is None or k < warmup:
            continue
        out.append(_flow_stats(res["flow"], tau))
    return np.array(out) if out else np.zeros((0, 4))


def run_noise_floor(cam, cfg, cw, ch, n_frames=40, tau=0.5, warmup=2, seed=0, window_box=None):
    print("\n[NF] ===== NOISE FLOOR TEST (do not touch the sensor) =====")
    unsafe = []
    if cfg.flow_deadzone > 0: unsafe.append(f"deadzone={cfg.flow_deadzone}")
    if cfg.flow_ema > 0: unsafe.append(f"ema={cfg.flow_ema}")
    if cfg.flow_ema_release > 0: unsafe.append(f"release={cfg.flow_ema_release}")
    if cfg.temporal_avg > 1: unsafe.append(f"tavg={cfg.temporal_avg}")
    if unsafe:
        print(f"[NF] WARNING: runtime filters active ({', '.join(unsafe)}). "
              "Press 'M' first so the numbers are comparable with Table IV.")

    md = cam.get_metadata()
    exp, gain = md.get("ExposureTime"), md.get("AnalogueGain")
    dgain, cgain = md.get("DigitalGain"), md.get("ColourGains")
    print(f"[NF] camera: exp={exp}us gain={gain} digital={dgain} colour={cgain}")
    if exp is not None and abs(float(exp) - MEASURE_EXP_US) > 0.1 * MEASURE_EXP_US:
        print(f"[NF] WARNING: exposure {exp}us != measurement operating point "
              f"{MEASURE_EXP_US}us (did you press 'M'?)")

    # ---- collect distinct frames ------------------------------------------
    frames, last_seq = [], None
    t_end = time.time() + 15.0 + n_frames * 0.2
    while len(frames) < n_frames and time.time() < t_end:
        ok, f, seq = cam.read()
        if ok and seq != last_seq:
            frames.append(apply_window(f.copy(), window_box)); last_seq = seq
        else:
            time.sleep(0.002)
    if len(frames) < 10:
        print("[NF] failed: too few frames"); return
    K = len(frames)
    print(f"[NF] {K} frames captured, computing...")

    # ---- clean image + sigma per channel per intensity ---------------------
    acc = np.zeros(frames[0].shape, np.float64)
    for f in frames:
        acc += f
    clean = (acc / K).astype(np.float32)
    g = cv2.cvtColor(np.clip(clean, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = grad <= np.percentile(grad, 30)

    sigma_maps, sigma_flat = [], []
    for c in range(clean.shape[2]):
        bins = np.clip((clean[..., c] / 256.0 * NF_BINS).astype(int), 0, NF_BINS - 1)
        bf = bins[flat]
        s2, cnt = np.zeros(NF_BINS), np.zeros(NF_BINS)
        for i in range(1, K):
            d = frames[i][..., c].astype(np.float32) - frames[i - 1][..., c]
            s2 += np.bincount(bf, d[flat].astype(np.float64) ** 2, NF_BINS)
            cnt += np.bincount(bf, minlength=NF_BINS)
        valid = cnt > 500
        if not valid.any():
            print("[NF] failed: noise could not be estimated"); return
        idx = np.where(valid)[0]
        sig = np.sqrt(s2[idx] / cnt[idx] / 2.0)
        sigma_maps.append(np.interp(np.arange(NF_BINS), idx, sig)[bins].astype(np.float32))
        sigma_flat.append(float(np.sqrt(s2[valid].sum() / cnt[valid].sum() / 2.0)))
    sigma_map = np.dstack(sigma_maps)

    means = np.array([float(f.mean()) for f in frames])
    flicker_pct = 100.0 * float(np.std(np.diff(means)) / np.sqrt(2) / max(means.mean(), 1e-6))

    # ---- three sequences through the same pipeline ------------------------
    c2 = copy.deepcopy(cfg)
    try:
        c2.enable_depth = False          # depth is not used here, saves time
    except Exception:
        pass

    rng = np.random.default_rng(seed)

    def synthetic():
        for _ in range(K):
            n = rng.standard_normal(clean.shape, dtype=np.float32) * sigma_map
            yield np.clip(np.rint(clean + n), 0, 255).astype(np.uint8)

    ref_mean = float(frames[0].mean())

    def gain_matched():
        for f in frames:
            k = ref_mean / max(float(f.mean()), 1e-6)
            yield np.clip(np.rint(f.astype(np.float32) * k), 0, 255).astype(np.uint8)

    meas = _run_sequence(c2, cw, ch, frames, warmup, tau)
    noise = _run_sequence(c2, cw, ch, synthetic(), warmup, tau)
    meas_g = _run_sequence(c2, cw, ch, gain_matched(), warmup, tau)
    if len(meas) == 0 or len(noise) == 0:
        print("[NF] failed: TactileSensor returned no flow (ROI?)"); return

    ratio = meas[:, 0].mean() / max(noise[:, 0].mean(), 1e-9)
    g_drop = 1.0 - meas_g[:, 0].mean() / max(meas[:, 0].mean(), 1e-9)
    coh_ratio = meas[:, 3].mean() / max(noise[:, 3].mean(), 1e-9)
    shift_frac = meas[:, 2].mean() / max(meas[:, 3].mean(), 1e-9)

    print(f"\n[NF] noise sensor (flat px, DN)  B {sigma_flat[0]:.2f}  "
          f"G {sigma_flat[1]:.2f}  R {sigma_flat[2]:.2f}   flicker {flicker_pct:.3f}%")
    print(f"[NF] {'':<24}{'||F|| px':>14}{f'|F|>{tau} %':>12}{'shift':>9}{'coherent':>10}")
    for lab, a in (("noise-only floor", noise), ("measured", meas),
                   ("measured, gain-matched", meas_g)):
        print(f"[NF] {lab:<24}{a[:, 0].mean():>8.4f}+/-{a[:, 0].std():.3f}"
              f"{a[:, 1].mean():>12.2f}{a[:, 2].mean():>9.4f}{a[:, 3].mean():>10.4f}")
    print(f"[NF] measured / noise floor = {ratio:.2f}   "
          f"gain matching removes {100*g_drop:.0f}%   "
          f"coherent ratio {coh_ratio:.2f}   shift/coherent {shift_frac:.2f}")

    diag = []
    if ratio < 1.25:
        diag.append("AT NOISE FLOOR: ||F|| is sensor noise, not motion.")
    else:
        if g_drop > 0.3:
            diag.append("FLICKER: brightness normalisation removes most of the flow. "
                        "Check the LED driver (PWM vs 2 ms exposure).")
        if coh_ratio > 2.0 and shift_frac > 0.5:
            diag.append("RIGID MOTION: the whole image shifts. Camera/rig vibration.")
        elif coh_ratio > 2.0:
            diag.append("LOCAL COHERENT MOTION: gel/decal moving (creep, temperature) or airflow.")
        if g_drop <= 0.3 and coh_ratio <= 2.0:
            diag.append("INCOHERENT EXCESS: not flicker, not motion. Noise not captured "
                        "by the white-noise model (ISP NR/sharpening, correlated noise).")
    for d in diag:
        print(f"[NF] -> {d}")

    fn = f"noise_floor_{int(time.time())}.json"
    with open(fn, "w") as fh:
        json.dump(dict(
            frames=K, tau=tau, warmup=warmup,
            camera=dict(exposure_us=exp, analogue_gain=gain, digital_gain=dgain,
                        colour_gains=cgain),
            runtime_filters_active=unsafe,
            sigma_flat_bgr=sigma_flat, flicker_pct=flicker_pct,
            noise_floor=noise.mean(axis=0).tolist(),
            measured=meas.mean(axis=0).tolist(),
            measured_gain_matched=meas_g.mean(axis=0).tolist(),
            columns=["flow_mean_px", "pct_above_tau", "global_shift_px", "coherent_px"],
            ratio=ratio, gain_drop=g_drop, coherent_ratio=coh_ratio,
            shift_over_coherent=shift_frac, diagnosis=diag), fh, indent=2, default=str)
    print(f"[NF] saved to {fn}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  3D views: Cv3DView (arg '3d') and the optional Open3D window (arg 'o3d')
# ─────────────────────────────────────────────────────────────────────────────
class Cv3DView:
    """3D depth view WITHOUT OpenGL (OpenCV + numpy only) -> always runs on the Pi 5.

    The depth surface is drawn as dense points (simple z-buffer), coloured by
    height (colormap) blended with the speckle texture. Mouse in the '3D' window:
      left drag = rotate | scroll = zoom | right drag = height (z) scale | double click = reset
    Same interface as the Open3D window: start(), push(depth, tex), stop(), running.
    """
    WIN = "3D depth (drag=rotate, scroll=zoom, right drag=z scale, double click=reset)"

    def __init__(self, z_scale=5.0, size=(640, 480), grid=(200, 150), max_hz=15.0):
        self.W, self.H = size
        self.gw, self.gh = grid
        self.z0 = z_scale
        self.max_dt = 1.0 / max_hz
        self.running = False
        self._t = 0.0
        self._drag = None
        self._job, self._img, self._new = None, None, False
        self._lock = threading.Lock()
        self._ev = threading.Event()
        self.reset_view()

    def reset_view(self):
        self.yaw, self.pitch, self.zoom, self.zs = -25.0, 55.0, 1.0, self.z0

    def start(self):
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, self.W, self.H)
        cv2.setMouseCallback(self.WIN, self._mouse)
        self.running = True
        # render in a separate thread -> sensor loop never waits; imshow stays on the main thread
        threading.Thread(target=self._render_loop, daemon=True, name="cv3d").start()

    def _render_loop(self):
        while self.running:
            if not self._ev.wait(0.2):
                continue
            self._ev.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            img = self.render(*job)
            with self._lock:
                self._img, self._new = img, True

    def stop(self):
        self._ev.set()
        if self.running:
            try:
                cv2.destroyWindow(self.WIN)
            except cv2.error:
                pass
        self.running = False

    def _mouse(self, ev, x, y, flags, _):
        if ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self._drag = (ev, x, y, self.yaw, self.pitch, self.zs)
        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self._drag = None
        elif ev == cv2.EVENT_MOUSEMOVE and self._drag is not None:
            kind, x0, y0, yaw0, pitch0, zs0 = self._drag
            if kind == cv2.EVENT_LBUTTONDOWN:
                self.yaw = yaw0 + (x - x0) * 0.4
                self.pitch = float(np.clip(pitch0 + (y - y0) * 0.4, 0.0, 89.0))
            else:
                self.zs = float(np.clip(zs0 * (1.0 + (y0 - y) * 0.01), 0.2, 200.0))
        elif ev == cv2.EVENT_MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (1.1 if flags > 0 else 1 / 1.1), 0.3, 5.0))
        elif ev == cv2.EVENT_LBUTTONDBLCLK:
            self.reset_view()

    def push(self, depth, tex=None):
        if not self.running or depth is None:
            return
        try:
            if cv2.getWindowProperty(self.WIN, cv2.WND_PROP_VISIBLE) < 1:
                self.stop(); return                 # window closed by the user
        except cv2.error:
            self.stop(); return
        now = time.perf_counter()
        if now - self._t >= self.max_dt:            # submit a render job (latest only)
            self._t = now
            with self._lock:
                self._job = (depth, None if tex is None else tex.copy())
            self._ev.set()
        with self._lock:
            img, new, self._new = self._img, self._new, False
        if new and img is not None:
            cv2.imshow(self.WIN, img)

    def render(self, depth, tex=None):
        gw, gh = self.gw, self.gh
        d = cv2.resize(depth.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
        d = np.nan_to_num(d)
        dmax = float(np.percentile(np.abs(d), 99.5)) if d.size else 0.0
        # normalised grid coords: x -1..1, y by aspect ratio, z = depth * scale
        aspect = gh / float(gw)
        xs = np.linspace(-1.0, 1.0, gw, dtype=np.float32)
        ys = np.linspace(-aspect, aspect, gh, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)
        Z = d * (self.zs * 0.5)
        # rotation: yaw (z axis), then pitch (x axis)
        a, b = np.radians(self.yaw), np.radians(self.pitch)
        x1 = X * np.cos(a) - Y * np.sin(a)
        y1 = X * np.sin(a) + Y * np.cos(a)
        y2 = y1 * np.cos(b) - Z * np.sin(b)
        z2 = y1 * np.sin(b) + Z * np.cos(b)
        s = 0.42 * min(self.W, self.H) * self.zoom
        u = (self.W / 2 + x1 * s).astype(np.int32).ravel()
        v = (self.H / 2 + y2 * s).astype(np.int32).ravel()
        zz = z2.ravel()
        # colour: height colormap blended with the speckle texture
        hn = np.clip(d / dmax, 0, 1) if dmax > 1e-6 else np.zeros_like(d)
        col = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        if tex is not None:
            t = cv2.resize(tex, (gw, gh), interpolation=cv2.INTER_AREA)
            if t.ndim == 2:
                t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            col = cv2.addWeighted(col, 0.65, t, 0.35, 0)
        col = col.reshape(-1, 3)
        img = np.full((self.H, self.W, 3), (34, 28, 26), np.uint8)
        order = np.argsort(zz)                       # far first, near overwrites
        u, v, c = u[order], v[order], col[order]
        filled = np.zeros((self.H, self.W), bool)
        for du in (0, 1, 2):                         # 3x3 splat for a dense surface
            for dv in (0, 1, 2):
                uu, vv = u + du - 1, v + dv - 1
                ok = (uu >= 0) & (uu < self.W) & (vv >= 0) & (vv < self.H)
                img[vv[ok], uu[ok]] = c[ok]
                filled[vv[ok], uu[ok]] = True
        # close gaps on steep slopes: fill empty pixels surrounded by surface
        k = np.ones((5, 5), np.uint8)
        hole = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool) & ~filled
        if hole.any():
            grown = cv2.dilate(img, k)
            img[hole] = grown[hole]
        cv2.putText(img, f"depth max {dmax:.3f} | z scale {self.zs:.1f}x | zoom {self.zoom:.1f}x",
                    (10, self.H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        return img


def make_3d_window(z_scale=5.0):
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    import queue as _queue

    STEP, DENT = 4, 0.01

    def build_pcd(depth, color, zs):
        h, w = depth.shape
        rows = np.arange(0, h, STEP); cols = np.arange(0, w, STEP)
        rr, cc = np.meshgrid(rows, cols, indexing="ij"); rr = rr.ravel(); cc = cc.ravel()
        d = depth[rr, cc].astype(np.float32)
        X = cc.astype(np.float32) / max(w - 1, 1)
        Y = (max(h - 1, 1) - rr).astype(np.float32) / max(h - 1, 1)
        d99 = float(np.percentile(d, 99))
        if d99 < DENT:
            Z = np.zeros_like(d); C = np.full((len(Z), 3), 0.35, np.float32)
        else:
            dc = np.clip(d, 0.0, d99); Z = dc * zs
            if color is not None:
                if color.shape[:2] != (h, w):
                    color = cv2.resize(color, (w, h))
                C = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32)[rr, cc] / 255.0
            else:
                rng = float(dc.max() - dc.min()); dn = (dc - dc.min()) / (rng + 1e-6)
                C = cv2.applyColorMap((dn * 255).astype(np.uint8).reshape(-1, 1),
                                      cv2.COLORMAP_JET).reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack([X, Y, Z]))
        pcd.colors = o3d.utility.Vector3dVector(C)
        return pcd

    class W:
        def __init__(self):
            self.q = _queue.Queue(maxsize=1)
            self.zs = z_scale
            self.running = False
            self.pcd = o3d.geometry.PointCloud()
            self._added = False
            self.mat = rendering.MaterialRecord(); self.mat.shader = "defaultUnlit"; self.mat.point_size = 4.0
            self.t = threading.Thread(target=self._run, daemon=True, name="o3d")

        def start(self): self.running = True; self.t.start()

        def stop(self):
            self.running = False
            try: self.q.put_nowait(None)
            except Exception: pass
            self.t.join(timeout=3)

        def push(self, depth, color):
            try: self.q.get_nowait()
            except Exception: pass
            try: self.q.put_nowait((depth.copy(), None if color is None else color.copy()))
            except Exception: pass

        def _run(self):
            app = gui.Application.instance; app.initialize()
            win = app.create_window("Tactile | Heightmap", 720, 600)
            wd = gui.SceneWidget(); wd.scene = rendering.Open3DScene(win.renderer)
            wd.scene.set_background([0.05, 0.05, 0.05, 1.0]); wd.scene.show_axes(True)
            win.add_child(wd)
            rc = win.content_rect; wd.frame = gui.Rect(rc.x, rc.y, rc.width, rc.height)
            win.set_on_layout(lambda th: setattr(wd, "frame", gui.Rect(*[win.content_rect.x, win.content_rect.y, win.content_rect.width, win.content_rect.height])))
            win.set_on_close(lambda: setattr(self, "running", False) or True)

            def tick():
                if not self.running:
                    app.quit(); return True
                try: item = self.q.get_nowait()
                except Exception: win.post_redraw(); return True
                if item is None: self.running = False; app.quit(); return True
                depth, color = item
                try:
                    np_ = build_pcd(depth, color, self.zs)
                    self.pcd.points = np_.points; self.pcd.colors = np_.colors
                    if self._added: wd.scene.remove_geometry("p")
                    wd.scene.add_geometry("p", self.pcd, self.mat)
                    if not self._added:
                        wd.look_at([0.5, 0.5, 0.0], [0.5, -0.3, 2.5], [0.0, 1.0, 0.0])
                    self._added = True; win.post_redraw()
                except Exception as e:
                    print(f"[O3D] skip: {e}")
                return True

            win.set_on_tick_event(tick)
            app.run()
            self.running = False

    return W()


# ─────────────────────────────────────────────────────────────────────────────
PROFILES = {
    "turbo":   dict(cw=640,  ch=480,  fr=120),
    "fast":    dict(cw=1280, ch=720,  fr=90),
    "default": dict(cw=1280, ch=720,  fr=60),
    "quality": dict(cw=1920, ch=1080, fr=30),
    # FULL FOV: sensor mode 1640x1232 (whole IMX219 sensor, 2x2 binned).
    # Other modes (1920x1080, 640x480) are CENTRE CROPS of the sensor -> vertical FOV
    # is not enough for the 18x24 mm window. Max ~40 fps in this mode.
    "full":    dict(cw=820,  ch=616,  fr=40, raw=(1640, 1232),
                    window=(265, 105, 565, 495)),   # gel window in the 820x616 frame
    # USB: Pi Zero as a UVC webcam (YUYV 640x480). Ceiling ~15 fps (Zero W CPU).
    "usb":     dict(cw=640,  ch=480,  fr=15),
}

# Trackbar values that are REQUIRED when taking no-contact data (Table IV).
# Not a preference: each one changes the measured numbers, not just the display.
MEASURE_SAFE = {
    "center_suppress": 0,   # mask multiplies flow -> shrinks what is measured
    "rebase hold":     0,   # baseline reset -> erases the measured drift
    "deadzone x10":    0,   # runtime stage of Sec. VII; Table IV is without it
    "ema x100":        0,
    "release x100":    0,
    "tavg":            1,
}


CLEAN_CONFIG = dict(flow_deadzone=0.0, flow_ema=0.0, flow_ema_release=0.0, temporal_avg=1,
                    rebaseline_hold=0, rebaseline_flow_thresh=0.5, auto_rebaseline=False,
                    valid_margin=15, depth_min=0.05, depth_open=0)


def set_tracking(cfg, mode):
    """'single' = one reference frame. 'adaptive' = DelTact paper Sec. IV-B.1: when the
    photometric error vs the current reference gets too large, the current frame becomes a
    new reference and the flows are CHAINED (composed), so large deformation keeps being
    tracked; loop closure drops references again when the gel returns."""
    cfg.tracking_mode = "adaptive" if mode == "adaptive" else "single"
    cfg.enable_loop_closure = cfg.tracking_mode == "adaptive"


def load_gui(path):
    """GUI-only settings (trackbars that are not part of SensorConfig, per-camera
    roi_thresh in dual) stored in the 'gui' block of sensor_config.json."""
    try:
        with open(path) as f:
            g = json.load(f).get("gui")
        return dict(g) if isinstance(g, dict) else {}
    except Exception:
        return {}


def save_config(cfg, path, gui=None):
    """SensorConfig + 'gui' block -> json. The existing gui block is kept and updated
    (single and dual share the file). Written atomically (no half-written file)."""
    from dataclasses import asdict
    g = load_gui(path)
    if gui:
        g.update(gui)
    d = asdict(cfg)
    d["gui"] = g
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, path)


def apply_clean_config(cfg):
    """Neutral settings: no flow deadzone/EMA/averaging, no auto-rebaseline -> flow & depth agree."""
    for k, v in CLEAN_CONFIG.items():
        setattr(cfg, k, v)


def main():
    args = [a.lower() for a in sys.argv[1:]]
    if "dual" in args or "triple" in args:
        # TWO / THREE CAMERAS in one window -> run_sensor_dual.py
        import run_sensor_dual
        return run_sensor_dual.main()
    # USB camera: 'usb' (auto-detect) or --usb=/dev/... (path is NOT lowercased)
    usb_dev = None
    use_usb = "usb" in args
    for a in sys.argv[1:]:
        if a.lower().startswith("--usb="):
            use_usb, usb_dev = True, a.split("=", 1)[1]
    if not use_usb and not csi_cameras():
        import glob
        if glob.glob("/dev/v4l/by-id/*Tactile*video-index0"):
            print("[INFO] no CSI camera, but 'Tactile UVC Cam' (USB) found -> switching to 'usb' mode")
            use_usb = True
        else:
            sys.exit("[ERROR] neither a CSI camera nor the 'Tactile UVC Cam' USB camera was found. "
                     "Check: rpicam-hello --list-cameras  and  v4l2-ctl --list-devices")
    pk = ("usb" if use_usb else
          "full" if "full" in args else
          "turbo" if "turbo" in args else ("fast" if "fast" in args else
         ("quality" if "quality" in args else "default")))
    profile = PROFILES[pk]

    cam_id = 1 if "cam1" in args else 0
    cam_tag = "csi"            # name used for calibration / log files (USB: Zero hostname)
    # one fixed place for depth-calibration files, next to this script
    CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib")
    # config always next to the script (NOT the current folder) -> same file wherever you start it
    CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_config.json")
    clean_cfg = ("clean" in args) or ("--clean" in args)

    # ── slip features: histogram bin edges & sample grid ─────────────
    slip_max_px = 8.0          # --slip-max=N (pixels, histogram upper edge)
    slip_rows, slip_cols = 7, 9   # --slip-grid=RxC (default 7x9 = 63 points)
    pattern_type = "unknown"   # --pattern=random|palette|...
    usb_sensor_w = 1280        # --sensor-w=N : sensor area width (px) used by the Zero, for focal
    focal_override = None      # --focal=N    : force focal (px) directly
    lens = "default"           # --lens=checker : checkerboard calibration (deck slide 28, CSI full)
    depth_calib_path = None    # --depth-calib=file.json : depth -> mm
    calib_step, calib_metric = 0.1, "max"   # --calib-step=0.1  --calib-metric=max|p99
    flow_method = None         # --flow=medium|farneback|... (default: deltact_plus built-in DIS)
    depth_stride = 1           # --depth-stride=N : depth computed every N frames
    depth_refine = False       # DEFAULT = calibrated raw depth (as before the refiner); --depth-refine=on / 'f' to enable
    display_hz = 30.0          # --display-hz=N   : display capped at N Hz (processing still every frame)
    do_prof = "prof" in args   # 'prof' : print per-stage timing every 100 frames
    for a in args:
        try:
            if a.startswith("--usb="):
                pass                                  # already parsed above
            elif a.startswith("--usb-size=") and use_usb:
                _w, _h = a.split("=", 1)[1].split("x")
                profile = dict(profile, cw=int(_w), ch=int(_h))
            elif a.startswith("--usb-fps=") and use_usb:
                profile = dict(profile, fr=float(a.split("=", 1)[1]))
            elif a.startswith("--sensor-w="):
                usb_sensor_w = max(1, int(a.split("=", 1)[1]))
            elif a.startswith("--depth-calib="):
                depth_calib_path = sys.argv[1:][args.index(a)].split("=", 1)[1]
            elif a.startswith("--calib-step="):
                calib_step = float(a.split("=", 1)[1])
            elif a.startswith("--calib-metric="):
                calib_metric = a.split("=", 1)[1]
            elif a.startswith("--lens="):
                lens = a.split("=", 1)[1]
            elif a.startswith("--focal="):
                focal_override = float(a.split("=", 1)[1])
            elif a.startswith("--flow="):
                flow_method = a.split("=", 1)[1]
            elif a.startswith("--depth-stride="):
                depth_stride = max(1, int(a.split("=", 1)[1]))
            elif a.startswith("--flow-vis="):
                global FLOW_VIS_MAX
                FLOW_VIS_MAX = max(0.1, float(a.split("=", 1)[1]))
            elif a.startswith("--depth-refine="):
                depth_refine = a.split("=", 1)[1] not in ("off", "0", "none", "no")
            elif a.startswith("--display-hz="):
                display_hz = max(1.0, float(a.split("=", 1)[1]))
            elif a.startswith("--slip-max="):
                slip_max_px = max(0.5, float(a.split("=", 1)[1]))
            elif a.startswith("--slip-grid="):
                _r, _c = a.split("=", 1)[1].lower().split("x")
                slip_rows, slip_cols = max(2, int(_r)), max(2, int(_c))
            elif a.startswith("--pattern="):
                pattern_type = a.split("=", 1)[1]
            elif a.startswith("--window="):
                v = a.split("=", 1)[1]
                profile = dict(profile, window=None if v in ("off", "none") else
                               tuple(int(t) for t in v.split(",")))
        except ValueError:
            sys.exit(f"[ERROR] invalid argument '{a}'.")
    exp_cap = int(1e6 / profile["fr"])   # exposure cannot exceed 1 frame
    global WINDOW_BOX
    WINDOW_BOX = profile.get("window")
    if WINDOW_BOX is not None:
        print(f"[INFO] gel window masking: x {WINDOW_BOX[0]}-{WINDOW_BOX[2]}, "
              f"y {WINDOW_BOX[1]}-{WINDOW_BOX[3]} (change with --window=x0,y0,x1,y1 "
              f"or --window=off). Then set roi_thresh ~110 and press 'd'.")

    # config sensor
    cfg_from_file = False                 # True: keep saved values (e.g. roi_thresh)
    if   "fast"    in args and pk == "fast":    cfg = dp.SensorConfig.fast_profile()
    elif "quality" in args:                     cfg = dp.SensorConfig.quality_profile()
    else:
        cfg = None
        if not clean_cfg:
            try:
                cfg = dp.SensorConfig.load(CONFIG_FILE)
                print(f"[INFO] config loaded: {CONFIG_FILE}")
            except Exception:
                cfg = None
        if cfg is None:
            cfg = dp.SensorConfig()
            print("[INFO] config: built-in defaults" + (" (clean)" if clean_cfg else
                  f" ({CONFIG_FILE} not found)"))
        else:
            cfg_from_file = True
    if clean_cfg:
        # neutral settings for THIS run only; the saved file is not touched until 'c'
        apply_clean_config(cfg)
        print(f"[INFO] CLEAN settings for this run; saved {os.path.basename(CONFIG_FILE)} "
              f"is NOT changed (press c to save). Next time start WITHOUT 'clean' to load it.")
    gui = {} if clean_cfg else load_gui(CONFIG_FILE)
    # depth runs in a separate process BY DEFAULT (big FPS gain); 'nopardepth' = old inline mode
    par_depth = ("nopardepth" not in args) and ("nodepth" not in args)
    if "nodepth" in args:
        cfg.enable_depth = False

    # ── Undistortion from the fov200 tuning ────────────────────────────────
    # Calibrated on a 1920x1080 capture. k1,k2 & roi_thresh do not depend on
    # resolution -> applied directly. focal (pixels) depends on the ROI resolution,
    # so it is scaled to the active profile height (focal/height kept constant).
    cfg.dist = (-0.60, -0.21, 0.0, 0.0, 0.0)        # k1=-0.60, k2=-0.21
    if not cfg_from_file:
        cfg.roi_thresh = 104                         # default; a saved value is kept
    FOCAL_AT_1080 = 1426.0
    cfg.focal = FOCAL_AT_1080 * (profile["ch"] / 1080.0)

    # '3d'  = OpenCV 3D view (no OpenGL, runs on the Pi 5)
    # 'o3d' = old Open3D window (needs OpenGL >= 4.1 -> does NOT run on the Pi 5 GPU)
    use_o3d = ("o3d" in args) and cfg.enable_depth
    enable_3d = (("3d" in args) or use_o3d) and cfg.enable_depth
    if use_o3d:
        try:
            import open3d  # noqa: F401
        except ImportError:
            print("[WARN] open3d not installed -> using the OpenCV 3D view"); use_o3d = False

    if use_usb:
        dev = resolve_usb(usb_dev)
        _b = os.path.basename(dev)
        cam_tag = _b.split("_Tactile")[0].replace("usb-", "") if "_Tactile" in _b else "usb"
        cam = UsbCameraReader(dev, profile["cw"], profile["ch"], profile["fr"])
        if cam.size != (profile["cw"], profile["ch"]):
            print(f"[WARN] USB camera delivers {cam.size}, not {profile['cw']}x{profile['ch']}")
            profile = dict(profile, cw=cam.size[0], ch=cam.size[1])
        # focal: no ScalerCrop over USB. The Zero picks its own sensor mode;
        # check on the Zero: journalctl -u uvc-gadget -b | grep "Selected sensor format"
        #   640x480 crop (1280x960 sensor) -> --sensor-w=1280 (default)
        #   1640x1232 full FOV (3280 sensor) -> --sensor-w=3280
        if lens == "checker":
            print("[WARN] --lens=checker is the CSI camera calibration; the Pi Zero camera module "
                  "has its own lens -> not applied. Recalibrate with a checkerboard.")
        native_per_px = usb_sensor_w / float(profile["cw"])
        cfg.focal = focal_override or FOCAL_AT_1080 / native_per_px
        print(f"[INFO] USB {dev} -> {profile['cw']}x{profile['ch']} | focal={cfg.focal:.1f}px "
              f"(sensor-w={usb_sensor_w}{', OVERRIDE' if focal_override else ''})")
        print("[INFO] USB: keys e/[ ]/-/= do not lock the camera (no controls over UVC); "
              "'u' only cleans the trackbars.")
        cam.start()
    else:
        if Picamera2 is None:
            sys.exit("[ERROR] picamera2 is not installed. Install: sudo apt install -y python3-picamera2"
                     "  (or use the USB camera: add the 'usb' argument)")
        cam = None
    if not use_usb:
      # camera info
      csi = csi_cameras()
      print(f"[INFO] {len(csi)} CSI camera(s) detected; using cam{cam_id}.")
      if cam_id >= len(csi):
          sys.exit(f"[ERROR] CSI camera cam{cam_id} does not exist.")

      picam2 = open_camera(csi[cam_id], profile["cw"], profile["ch"], profile["fr"], profile.get("raw"))

      # ── CORRECT focal for the sensor mode in use ─────────────────────────
      # FOCAL_AT_1080 (1426 px) was calibrated in 1920x1080 mode = 1 sensor pixel
      # per image pixel. Other modes are binned and/or cropped, so the correct
      # scale = ScalerCrop width (sensor pixels) / image width, NOT height/1080.
      # (Previously: turbo 640x480 got focal 634, while it should be ~713.)
      try:
          crop = None
          for _ in range(5):
              crop = picam2.capture_metadata().get("ScalerCrop")
              if crop: break
          if crop:
              native_per_px = float(crop[2]) / float(profile["cw"])
              cfg.focal = FOCAL_AT_1080 / native_per_px
              print(f"[INFO] ScalerCrop={tuple(crop)} -> {native_per_px:.3f} px sensor/px "
                    f"-> focal={cfg.focal:.1f}px")
          else:
              print("[WARN] ScalerCrop not available; focal uses the old height/1080 scale.")
      except Exception as e:
          print(f"[WARN] failed to read ScalerCrop ({e}); focal uses the old scale.")
      if lens == "checker":
          done = apply_lens_checker(cfg, profile["cw"], float(crop[2]) if crop else None)
          if done:
              print(f"[INFO] lens: CHECKERBOARD calibration applied ({', '.join(done)})")
              if "cx" not in done:
                  print("[WARN] the sensor module has no principal-point setting (cx, cy): the image "
                        "centre is used, ~10 px off the calibrated (419.7, 307.3).")
      cam = CameraReader(picam2, name=f"cam{cam_id}"); cam.start()
    install_flow_method(flow_method)
    print(f"[INFO] acceleration: {gpu_report()}")
    sensor = dp.TactileSensor(cfg, profile["cw"], profile["ch"])
    if flow_method:
        if FLOW_USED["n_create"]:
            print(f"[INFO] optical flow: {flow_method} (replaces the built-in DIS)")
        else:
            print(f"[WARN] --flow={flow_method} NOT applied: the sensor module does not create DIS "
                  f"via cv2.DISOpticalFlow_create. Flow stays at the default.")
    print(f"[INFO] {pk} | {profile['cw']}x{profile['ch']} @ {profile['fr']}fps | depth={'ON' if cfg.enable_depth else 'OFF'} | 3D={'ON' if enable_3d else 'OFF'}")
    print(f"[INFO] undistort: focal={cfg.focal:.1f}px, "
          f"k1={cfg.dist[0]}, k2={cfg.dist[1]}, roi_thresh={cfg.roi_thresh}")

    live3d = None
    if enable_3d:
        live3d = make_3d_window(z_scale=5.0) if use_o3d else Cv3DView(z_scale=5.0)
        live3d.start()

    WIN = "Tactile Sensor [r=base d=ROI a=arrow e=lockAE u=measure m=record n=noisefloor s=snap b=burst q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 420)
    TB = "Controls"
    cv2.namedWindow(TB, cv2.WINDOW_NORMAL); cv2.resizeWindow(TB, 430, 420)

    def nb(n, v, mx): cv2.createTrackbar(n, TB, int(v), mx, lambda _: None)
    nb("arrow_scale",    gui.get("arrow_scale", 5),  25)
    nb("arrow_step",     gui.get("arrow_step", 16), 40)
    nb("deadzone x10",   int(cfg.flow_deadzone * 10),            40)
    nb("ema x100",       int(cfg.flow_ema * 100),                95)
    nb("release x100",   int(cfg.flow_ema_release * 100),        95)
    nb("tavg",           max(cfg.temporal_avg, 1),                8)
    nb("depth_min x100", int(cfg.depth_min * 100),               30)
    nb("depth_open",     cfg.depth_open,                          9)
    nb("roi_thresh",     cfg.roi_thresh,                        200)
    nb("mask margin",    max(cfg.valid_margin, 0),               60)
    nb("rebase hold",    cfg.rebaseline_hold,                    30)
    nb("rebase thr x10", int(cfg.rebaseline_flow_thresh * 10),   40)
    nb("gain x10",       gui.get("gain x10", 12),  40)
    nb("bright",         gui.get("bright", 30), 150)

    # ─── CENTER SUPPRESSION ───
    nb("center_suppress", 0 if clean_cfg else gui.get("center_suppress", 10), 100)
    nb("supp_radius %",   gui.get("supp_radius %", 10), 60)   # 10..60%, default 10

    # state
    fps_t, fps, fc = time.time(), 0.0, 0
    tact_line = None                    # tactile summary line under the panels
    cam_seq0, cam_fps = 0, 0.0          # camera FPS (from frame sequence numbers)
    tick = 0                            # for trackbar polling
    _csock = None                       # UDP 127.0.0.1:9871 'reset' (servo_grasp.py) = key r
    _aref_prev = None                   # 'aref hold' / 'aref release' from servo_grasp.py
    try:
        import socket as _sk
        _csock = _sk.socket(_sk.AF_INET, _sk.SOCK_DGRAM)
        _csock.bind(("127.0.0.1", 9871)); _csock.setblocking(False)
    except OSError:
        _csock = None
    TB_POLL = 15                        # read trackbars every 15 iterations, not every frame
    depth_on = cfg.enable_depth         # depth stride: keep the original setting
    dcal = DepthCalib(calib_step, calib_metric)
    drefine = DepthRefiner()
    drefine.on = depth_refine
    last_dseq = None                    # parallel depth: only refine NEW depth frames
    dlog, dlog_path = None, None
    calib_file = os.path.join(CALIB_DIR, f"depth_calib_{cam_tag}.json")
    if depth_calib_path and depth_calib_path.lower() == "none":
        print("[CAL] depth -> mm DISABLED (relative units)")
    elif depth_calib_path:
        try:
            dcal.load(depth_calib_path)
        except Exception as e:
            print(f"[WARN] --depth-calib failed to load ({e}); using the default calibration")
            dcal.load_default()
    elif os.path.exists(calib_file):
        dcal.load(calib_file)                      # automatic: calib/depth_calib_<camera>.json
    else:
        dcal.load_default()
        print(f"[CAL] no {calib_file} yet -> built-in synthetic values")
    print(f"[INFO] depth calibration: 'p' = record point (step {calib_step} mm), 'z' = delete, "
          f"'o' = fit & save {calib_file}")
    dworker = None
    if par_depth and depth_on:
        try:
            dworker = DepthWorker(cfg, profile["cw"], profile["ch"], flow_method,
                                  refine_on=drefine.on, use_gpu=("gpu" in args))
            print("[INFO] PARALLEL depth on: depth in a separate process (flow/slip do not wait for depth)")
        except Exception as e:
            dworker = None
            print(f"[WARN] parallel depth failed to start ({e}); depth back in the main loop.")
    frame_i, last_depth = 0, None
    t_last_frame = time.time()
    n_reconnect_seen = 0
    depth_out, p3_cache, p3_dirty, new_depth = None, None, False, False
    DISP_DT = 1.0 / display_hz
    t_disp = 0.0
    prof, prof_n = {"process": 0.0, "features": 0.0, "viz": 0.0, "gui": 0.0}, 0
    if depth_on:
        if drefine.on:
            print("[INFO] depth refinement ON ('f' = toggle): spike removal, floor removal, "
                  "temporal filter, bilateral, contact mask + hole filling")
        else:
            print("[INFO] depth = CALIBRATED raw depth -> mm (no refinement; "
                  "'f' or --depth-refine=on to enable)")
    print(f"[INFO] depth stride={depth_stride} | display {display_hz:.0f} Hz | "
          f"profiler={'ON' if do_prof else 'OFF (add arg prof)'}")
    draw_arrow = bool(gui.get("arrows", True))
    ae_locked = False
    measure_mode = False
    exp_us = 8000      # initial exposure when locked (microseconds)
    gain_a = 2.0       # initial analogue gain when locked
    rest_frames = 0
    prev_mgn = cfg.valid_margin
    prev_thr = cfg.roi_thresh
    last_panel = None
    last_seq = None
    topt = tactile_ros.parse_args(sys.argv[1:])
    tact = tactile_ros.make_features(topt)
    aref = adaptive_ref.make(adaptive_ref.parse_args(sys.argv[1:]))
    detail_mode = "fine" if ("detail" in args or "texture" in args) else "off"
    for _m in ("residual", "depth", "fine"):
        if f"detail={_m}" in args:
            detail_mode = _m
    if "track=adaptive" in args or "--track=adaptive" in args:
        set_tracking(cfg, "adaptive")
    elif "track=single" in args or "--track=single" in args:
        set_tracking(cfg, "single")
    print(f"[TRACK] reference tracking: {getattr(cfg, 'tracking_mode', 'single').upper()} "
          f"('t' = toggle single / adaptive flow chaining)")
    print(f"[AREF] adaptive referencing {'ON' if aref.enabled else 'OFF'} ('w' = toggle): "
          f"re-baseline after release / residual flow / brightness change, never during contact")
    tact_name = cam_tag if use_usb else f"csi{cam_id}"
    ros = (tactile_ros.TactileROS([tact_name], topt["ns"], topt["img_hz"], udp=topt["udp"], ros=topt["ros"])
           if topt["on"] else None)
    depth_pub = DepthPublisher(with_color=True)   # << write depth+color to /dev/shm (cheap)

    # ─── RAW FRAME BURST ('S') ───
    burst_left = 0
    burst_skip = 0
    burst_dir = None
    burst_seqs = []
    burst_recs = []
    t_measure = None          # time of the last 'u'

    # ─── NO-CONTACT PROBE ───
    probe = NoContactProbe(cfg)

    # ─── SLIP FEATURES (Hu et al. 2024, arXiv:2303.00935) ───
    slog = SlipLogger("slip_log_1cam.csv", slip_rows, slip_cols,
                      slip_max_px, pattern_type)
    slip_grid = None
    slip_shape = None
    slip_prev_D = None
    slip_prev_E = None
    slip_t_prev = None
    slip_E = float("nan")
    slip_dE = 0.0
    print(f"[INFO] Slip log -> {slog.path} | grid {slip_rows}x{slip_cols} = "
          f"{slog.n_points} points, {SLIP_BINS} bins 0..{slip_max_px:.1f}px. "
          f"'g' start/stop recording, 'k' toggle label static<->slip.")

    # Cache Gaussian mask (rebuild only when h/w/radius change)
    gaussian_mask = None
    last_supp_params = (None, None, None)

    print("[INFO] Press 'r' with NO contact for a clean baseline.")
    print("[INFO] For experiments: press 'e' to LOCK exposure/gain/AWB (avoids fake flow).")
    print("[INFO] Use the 'center_suppress' & 'supp_radius %' trackbars to control the Gaussian taper.")
    print("[INFO] For Table IV data: press 'u' (measure mode), wait for STABLE, then 'm' to record.")
    print("[INFO] Noise floor: press 'u', wait ~5 s, then 'n' (no contact).")
    print("[INFO] Raw frames: 's' = 1 frame, 'b' = burst of 10 frames (for calibration).")

    # ─── RAW FRAME RECORDING ('v'): disk or RAM, optional pre-trigger ───
    raw_out, raw_fmt, raw_ram, raw_pre = ".", "png", "ram" in args, 0.0
    for a in sys.argv[1:]:
        al = a.lower()
        if al.startswith("--out="):
            raw_out = a.split("=", 1)[1]
        elif al.startswith("--fmt="):
            raw_fmt = al.split("=", 1)[1]
        elif al.startswith("--pre="):
            raw_pre = float(al.split("=", 1)[1])
    raw_name = "usb" if use_usb else "csi"
    raw_rec = Recorder(raw_name, cam, not use_usb, raw_fmt, use_usb,
                          ram=raw_ram, pre_s=raw_pre)
    raw_rec.start()
    raw_meta = {raw_name: dict(profile=pk, size=[profile["cw"], profile["ch"]],
                               fps_target=profile["fr"])}
    raw_folder, raw_t0 = None, None
    _mem = mem_available_bytes()
    print(f"[INFO] Raw recording: 'v' -> {os.path.abspath(raw_out)} | "
          f"mode {'RAM' if raw_ram else 'disk'} | pre-trigger {raw_pre:.1f}s"
          + (f" | free RAM {_mem / 1e9:.1f} GB" if _mem else ""))

    try:
        while True:
            if live3d is not None and not live3d.running:
                live3d = None; print("[INFO] 3D window closed.")

            # ─── trackbars: read every TB_POLL iterations (getTrackbarPos = GUI call) ───
            if tick % TB_POLL == 0:
                a_scale = max(1, cv2.getTrackbarPos("arrow_scale", TB))
                a_step  = max(6, cv2.getTrackbarPos("arrow_step",  TB))
                thr     = cv2.getTrackbarPos("roi_thresh",  TB)
                mgn     = cv2.getTrackbarPos("mask margin", TB)
                gain    = max(1, cv2.getTrackbarPos("gain x10", TB)) / 10.0
                bright  = cv2.getTrackbarPos("bright", TB)

                # ─── CENTER SUPPRESSION ───
                supp_strength = cv2.getTrackbarPos("center_suppress", TB)
                supp_radius_pct = cv2.getTrackbarPos("supp_radius %", TB)

                cfg.flow_deadzone    = cv2.getTrackbarPos("deadzone x10", TB) / 10.0
                cfg.flow_ema         = cv2.getTrackbarPos("ema x100",     TB) / 100.0
                cfg.flow_ema_release = cv2.getTrackbarPos("release x100", TB) / 100.0
                cfg.temporal_avg     = max(1, cv2.getTrackbarPos("tavg",  TB))
                cfg.depth_min        = cv2.getTrackbarPos("depth_min x100", TB) / 100.0
                cfg.depth_open       = cv2.getTrackbarPos("depth_open", TB)
                cfg.auto_rebaseline  = False
                cfg.rebaseline_hold        = cv2.getTrackbarPos("rebase hold", TB)
                cfg.rebaseline_flow_thresh = cv2.getTrackbarPos("rebase thr x10", TB) / 10.0
                reb_hold = cfg.rebaseline_hold
                reb_thr  = cfg.rebaseline_flow_thresh

                if mgn != prev_mgn:
                    cfg.valid_margin = mgn; sensor.rebuild_mask(); prev_mgn = mgn
                    if dworker: dworker.set_cfg(cfg); dworker.cmd("rebuild_mask")
                if thr != prev_thr:
                    cfg.roi_thresh = thr; sensor.redetect_roi(); prev_thr = thr
                    if dworker: dworker.set_cfg(cfg); dworker.cmd("redetect_roi")
            tick += 1

            ok, frame, seq = wait_frame(cam, last_seq, 0.02)
            # USB camera came back after a drop-out: the Zero may restart with a different
            # crop/scale, so the old baseline is invalid (shows as a radial "zoom" flow)
            if getattr(cam, "n_reconnect", 0) != n_reconnect_seen and ok:
                n_reconnect_seen = cam.n_reconnect
                sensor.redetect_roi(); sensor.reset(); drefine.reset(); rest_frames = 0
                aref.on_reference(); tact.reset()
                if dworker: dworker.cmd("redetect_roi"); dworker.cmd("reset")
                print("[INFO] USB camera reconnected -> ROI re-detected and baseline reset automatically")
            if not ok or seq == last_seq:
                # same frame as before: do not reprocess it (fake zero flow).
                # waitKey is NOT called for short waits so key presses are not lost,
                # but during a long stall (USB camera reconnecting) the window stays
                # responsive and 'q' still quits.
                if time.time() - t_last_frame > 0.5:
                    if last_panel is not None:
                        _p = last_panel.copy()
                        cv2.rectangle(_p, (0, 0), (_p.shape[1], 30), (0, 0, 160), -1)
                        cv2.putText(_p, f"NO FRAMES {time.time() - t_last_frame:.0f}s - camera "
                                    f"reconnecting... (q = quit)", (10, 21),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.imshow(WIN, _p)
                    if (cv2.waitKey(30) & 0xFF) in (27, ord('q')):
                        break
                    continue
                continue
            last_seq = seq
            t_last_frame = time.time()

            # ─── depth stride: depth only computed every depth_stride frames ───
            # (assumes deltact_plus reads cfg.enable_depth on every process call)
            frame_i += 1
            if dworker is not None:
                cfg.enable_depth = False            # depth is done by the other process
            elif depth_on and depth_stride > 1:
                cfg.enable_depth = (frame_i % depth_stride == 0)
            t_p0 = time.perf_counter()
            frame_in = apply_window(frame, WINDOW_BOX)
            res = sensor.process(frame_in)
            t_p1 = time.perf_counter()
            cfg.enable_depth = depth_on
            if dworker is not None and res is not None:
                _wm = sensor.mask
                _fm = sensor.undist.fold_mask
                if _fm is not None:
                    _wm = (_fm.astype(np.float32) if _wm is None else _wm * _fm)
                dworker.submit(seq, res["flow"],
                               None if _wm is None else _wm.astype(np.float16))
                if tick % TB_POLL == 1:
                    dworker.set_cfg(cfg)            # so trackbars depth_min/open etc. follow

            # display capped at display_hz; processing & features still every frame
            show = (t_p1 - t_disp) >= DISP_DT
            if show:
                t_disp = t_p1

            # ─── BURST: discard BURST_SKIP frames, then save BURST_N consecutive frames ───
            if burst_left > 0 and t_measure is not None and time.time() - t_measure < SETTLE_AFTER_U_S:
                pass                                        # wait for the camera to settle after 'u'
            elif burst_left > 0 and burst_skip > 0:
                burst_skip -= 1
            elif burst_left > 0:
                i = BURST_N - burst_left
                save_raw_bundle(burst_dir, i, frame, res)
                mseq, fmd = cam.frame_metadata()
                burst_recs.append(frame_record(seq, fmd if mseq == seq else None, frame, WINDOW_BOX))
                burst_seqs.append(int(seq)); burst_left -= 1
                if burst_left == 0:
                    write_meta(burst_dir, cam, cfg, pk, profile, ae_locked,
                               measure_mode, len(burst_seqs), burst_seqs, note="burst",
                               frames=burst_recs)
                    print(f"[SNAP] burst done: {len(burst_seqs)} frames -> {burst_dir}/")
            npanels = 3 if cfg.enable_depth else 2
            panel = None
            t_f0 = time.perf_counter(); t_v = 0.0
            if res is None:
                if show:
                    panel = np.zeros((cfg.work_h, cfg.work_w * npanels, 3), np.uint8)
                    label_bottom(panel, "ROI? adjust roi_thresh", (0, 0, 255))
            else:
                flow = res["flow"]; roi = res["roi"]

                # ─── NO-CONTACT PROBE ───────────────────────────────────────
                # Called here, BEFORE centre suppression touches the flow.
                # Uses the raw `roi`, not `roi_disp`: the gain/bright trackbars
                # are display-only and would change low_f and lambda_min
                # if they were measured too.
                probe.update(flow, roi, supp_strength, reb_hold)

                # ─── SLIP FEATURES ─────────────────────────────────────
                # Also here, BEFORE centre suppression: the mask
                # multiplies the flow, so entropy & Vx/Vy would shrink
                # if computed afterwards.
                if slip_shape != flow.shape[:2]:
                    slip_grid = make_slip_grid(flow.shape[0], flow.shape[1],
                                               slip_rows, slip_cols)
                    slip_shape = flow.shape[:2]
                    slip_prev_D = slip_prev_E = slip_t_prev = None
                t_now = time.monotonic()
                slip_dt = (t_now - slip_t_prev) if slip_t_prev else 0.0
                (slip_E, slip_dE, slip_vx, slip_vy, slip_mg, slip_p90,
                 slip_D) = slip_features(flow, slip_grid, slog.edges,
                                         slip_prev_D, slip_prev_E, slip_dt)
                slip_prev_D, slip_prev_E, slip_t_prev = slip_D, slip_E, t_now
                # depth: reuse the last one if this frame was skipped (depth stride)
                if dworker is not None:
                    depth = dworker.poll()
                    if dworker.seq == last_dseq:
                        depth = None                 # nothing new from the depth process
                    else:
                        last_dseq = dworker.seq
                else:
                    depth = res.get("depth") if depth_on else None
                if depth is not None and drefine.on and dworker is None:
                    depth = drefine(depth, cfg.depth_min)   # solid, stable surface
                new_depth = depth is not None
                if depth is None:
                    depth = last_depth
                else:
                    last_depth = depth
                    dcal.observe(depth)              # for calibration points ('p')
                    if dlog is not None:             # per-frame depth log ('l') -> mts_calib.py
                        _dm = float(np.max(depth))
                        dlog.write(f"{time.time():.4f},{time.monotonic():.4f},{int(seq)},"
                                   f"{_dm:.6f},{float(np.percentile(depth, 99)):.6f},"
                                   f"{float((depth > cfg.depth_min).mean()):.5f}\n")

                # flow magnitude computed ONCE (previously twice per frame)
                flow_mag = float(cv2.magnitude(flow[..., 0], flow[..., 1]).max())
                if slog.recording:
                    _cf = (float((depth > cfg.depth_min).mean())
                           if depth is not None else float("nan"))
                    slog.write(t_now, f"cam{cam_id}", slip_E, slip_dE,
                               slip_vx, slip_vy, slip_mg, slip_p90,
                               flow_mag, _cf, slip_dt)

                roi_lvl = float(roi.mean()) / 255.0   # REAL brightness, not the display one
                roi_disp = roi                        # used for publishing when not shown
                t_v0 = time.perf_counter()
                if show:
                    roi_disp = cv2.convertScaleAbs(roi, alpha=gain, beta=bright)
                    p1 = roi_disp.copy()
                    if draw_arrow:
                        quiver_fast(p1, flow, a_step, a_scale)
                    label_bottom(p1, f"Speckle |flow|={flow_mag:.1f}")
                    p2 = flow_to_color_fixed(flow)
                    label_bottom(p2, "Flow")
                    panels = [p1, p2]
                t_v += time.perf_counter() - t_v0

                # ─── Rebuild the Gaussian mask when parameters change ───
                supp_params = (cfg.work_h, cfg.work_w, supp_radius_pct)
                if supp_params != last_supp_params:
                    gaussian_mask = make_gaussian_center_mask(
                        cfg.work_h, cfg.work_w,
                        radius_pct=float(supp_radius_pct),
                        sigma_scale=1.2  # steepness control (> 1 = sharper)
                    )
                    last_supp_params = supp_params

                # ─── Apply centre suppression ───
                if supp_strength > 0 and gaussian_mask is not None:
                    flow = apply_center_suppression(flow, gaussian_mask, strength=supp_strength)
                    if depth is not None and new_depth:
                        depth = apply_center_suppression(depth, gaussian_mask, strength=supp_strength)
                if new_depth:
                    depth_out = dcal.to_mm(depth, cfg.depth_min)   # mm when calibrated
                    p3_dirty = True

                # ─── tactile features (contact, force, shear, twist, slip) + ROS 2 ───
                t_f, t_ev = tact.update(flow, depth_out if depth_on else None,
                                        slip_E, slip_dE, fps, t_now)
                for e_ in t_ev:
                    print(f"[TACTILE] {e_}")
                if ros is not None:
                    ros.publish(tact_name, t_f, t_ev, depth_out if new_depth else None)
                _reason = aref.update(flow, bool(t_f["contact"]), roi_lvl, t_now,
                                      blocked=probe.recording or slog.recording,
                                      gray=res.get("gray"),
                                      depth_max_mm=(t_f["depth_max_mm"] if depth_on
                                                    and dcal.gain is not None else None))
                if _reason:
                    sensor.reset(); rest_frames = 0; drefine.reset(); tact.reset()
                    if dworker: dworker.cmd("reset")
                    print(f"[AREF] new reference: {_reason}")
                if show:                     # tactile line, shown UNDER the panels
                    _txt, _sl = tact.summary()
                    tact_line = ((_txt, (60, 60, 230) if _sl else (120, 200, 90))
                                 if t_f["contact"] else ("no contact", (150, 150, 150)))

                t_v0 = time.perf_counter()
                if show:
                    if cfg.enable_depth:
                        if depth_out is None:
                            p3 = np.zeros((cfg.work_h, cfg.work_w, 3), np.uint8)
                        elif p3_dirty or p3_cache is None:     # colour map only per NEW depth
                            p3 = dp.depth_to_color(depth)
                            label_bottom(p3, (f"Depth {depth_out.max():.3f} mm" if dcal.gain is not None
                                              else f"Depth {depth.max():.3f}"), (255, 255, 255))
                            p3_cache, p3_dirty = p3, False
                        else:
                            p3 = p3_cache
                        panels.append(p3)
                    if detail_mode != "off":
                        _r, _rms = texture_detail(sensor, res, depth, detail_mode)
                        if _r is None:
                            p4 = np.zeros_like(panels[0])
                        else:
                            p4 = detail_to_color(_r, detail_mode)
                            if p4.shape[:2] != panels[0].shape[:2]:
                                p4 = cv2.resize(p4, (panels[0].shape[1], panels[0].shape[0]))
                        label_bottom(p4, f"Detail {detail_mode} rms {_rms:.3f}", (255, 255, 255))
                        panels.append(p4)
                    panel = np.hstack(panels)
                t_v += time.perf_counter() - t_v0

                # automatic rebaseline (when idle)
                # Skipped while the probe records: a baseline reset would erase
                # exactly the drift being measured.
                if reb_hold > 0 and not probe.recording and not slog.recording:
                    if flow_mag < reb_thr:
                        rest_frames += 1
                        if rest_frames >= reb_hold:
                            sensor.reset(); rest_frames = 0; drefine.reset(); aref.on_reference()
                            if dworker: dworker.cmd("reset")
                    else:
                        rest_frames = 0

                if new_depth:                                  # publish / 3D only per NEW depth
                    depth_pub.publish(depth_out, roi_disp)      # << separate 3D viewer (view_3d_live.py)
                    if live3d is not None:
                        live3d.push(depth_out, roi_disp)

            t_f1 = time.perf_counter()

            # loop FPS + camera FPS
            fc += 1
            if fc >= 20:
                now = time.time(); dt_f = now - fps_t
                fps = fc / dt_f
                cam_fps = (last_seq - cam_seq0) / dt_f if cam_seq0 else 0.0
                cam_seq0 = last_seq
                fps_t, fc = now, 0

            t_g0 = time.perf_counter()
            if show:
                status = f"{fps:.1f}FPS cam={cam_fps:.0f}"
                if res is not None:
                    warn = ("  DARK" if roi_lvl < 0.18 else
                            "  BRIGHT" if roi_lvl > 0.85 else "")
                    status += f" | lvl={roi_lvl:.2f}{warn}"
                if ae_locked:
                    status += f" | LOCK exp={exp_us}us gain={gain_a:.1f}"
                if measure_mode:
                    status += " | MEASURE"
                if supp_strength > 0:
                    status += f" | Supp={supp_strength}% r={supp_radius_pct}%"
                if res is not None and not np.isnan(slip_E):
                    status += f" | E={slip_E:.2f} dE/dt={slip_dE:+.1f}"
                if dworker is not None:
                    status += f" | D {dworker.fps:.0f}fps {dworker.ms:.0f}ms"
                if depth_on and drefine.on:
                    status += f" | DF {(dworker.refine_ms if dworker else drefine.ms):.0f}ms"
                if slog.recording:
                    status += (f" | REC t{slog.trial} "
                               f"{'SLIP' if slog.label else 'static'}")
                if raw_t0 is not None:
                    status += f" | RAW {time.monotonic() - raw_t0:.0f}s n={raw_rec.idx}"
                    if raw_ram:
                        status += f" ({raw_rec.ram_used_mb() / 1000:.1f}GB)"
                    if raw_rec.auto_stopped:
                        status += " RAM FULL-press v"

                # status in its own strip BELOW the panels (never over the panel labels)
                panel = add_status_strip(panel, status, tact_line)

                # ─── probe status line (above the FPS line) ───
                panel = probe.overlay(panel)

                cv2.imshow(WIN, panel); last_panel = panel

                # ── keyboard ── (only when shown; ~display_hz times/s, responsive enough)
                k = cv2.waitKey(1) & 0xFF
            else:
                k = 255
            if _csock is not None:                       # 'reset' from servo_grasp.py = key r
                try:
                    while True:
                        _m, _ = _csock.recvfrom(256)
                        _m = _m.strip().lower()
                        if _m == b"reset":
                            k = ord('r')
                        elif _m == b"aref hold" and _aref_prev is None:
                            _aref_prev, aref.enabled = aref.enabled, False; aref.on_reference()
                            print("[AREF] adaptive referencing OFF while grasping (grasp program)")
                        elif _m == b"aref release" and _aref_prev is not None:
                            aref.enabled, _aref_prev = _aref_prev, None; aref.on_reference()
                            print(f"[AREF] adaptive referencing {'ON' if aref.enabled else 'OFF'} again (grasp program)")
                except (BlockingIOError, OSError):
                    pass

            # ── profiler (arg 'prof') ──
            if do_prof:
                prof["process"] += t_p1 - t_p0
                prof["features"] += (t_f1 - t_f0) - t_v
                prof["viz"] += t_v
                prof["gui"] += time.perf_counter() - t_g0
                prof_n += 1
                if prof_n >= 100:
                    print("[PROF] " + " | ".join(f"{n}={v / prof_n * 1e3:.1f}ms"
                                                  for n, v in prof.items())
                          + f" | loop={fps:.1f}fps cam={cam_fps:.0f}fps")
                    prof = {n: 0.0 for n in prof}; prof_n = 0
            if k in (27, ord('q')):
                break
            elif k == ord('r'):
                sensor.reset(); rest_frames = 0; drefine.reset(); tact.reset(); aref.on_reference()
                print("[INFO] baseline reset")
                if dworker: dworker.cmd("reset")
            elif k == ord('d'):
                sensor.redetect_roi(); drefine.reset(); print("[INFO] ROI re-detected")
                if dworker: dworker.cmd("redetect_roi")
            elif k == ord('f'):
                drefine.on = not drefine.on; drefine.reset()
                if dworker: dworker.set_refine(drefine.on)
                print(f"[INFO] depth refinement: {'ON' if drefine.on else 'OFF (raw depth)'}")
            elif k == ord('x'):
                detail_mode = DETAIL_MODES[(DETAIL_MODES.index(detail_mode) + 1) % len(DETAIL_MODES)]
                print(f"[INFO] texture detail panel: {detail_mode.upper()} "
                      f"(x = off -> fine -> residual -> depth high-pass)")
            elif k == ord('t'):
                set_tracking(cfg, "single" if getattr(cfg, "tracking_mode", "single") == "adaptive"
                             else "adaptive")
                sensor.reset(); drefine.reset(); tact.reset(); aref.on_reference(); rest_frames = 0
                if dworker: dworker.cmd("reset")
                print(f"[TRACK] reference tracking: {cfg.tracking_mode.upper()} (baseline reset)")
            elif k == ord('w'):
                aref.enabled = not aref.enabled; aref.on_reference()
                print(f"[AREF] adaptive referencing {'ON' if aref.enabled else 'OFF'}")
            elif k == ord('a'):
                draw_arrow = not draw_arrow; print(f"[INFO] arrow: {'ON' if draw_arrow else 'OFF'}")
            elif k == ord('c'):
                _g = {n_: cv2.getTrackbarPos(n_, TB) for n_ in
                      ("arrow_scale", "arrow_step", "gain x10", "bright",
                       "center_suppress", "supp_radius %")}
                _g["arrows"] = draw_arrow
                save_config(cfg, CONFIG_FILE, _g)
                print(f"[INFO] config saved -> {CONFIG_FILE} (also gain/bright/arrows/suppression)")
            elif k in (ord('u'), ord('M')) and use_usb:
                for name, val in MEASURE_SAFE.items():
                    cv2.setTrackbarPos(name, TB, val)
                print("[INFO] trackbars cleaned: center_suppress 0, deadzone 0, EMA 0, release 0, "
                      "tavg 1, rebase hold 0 (USB camera exposure stays on the Pi Zero). "
                      "Release the sensor and press 'r'.")
            elif k in (ord('e'), ord('['), ord(']'), ord('-'), ord('=')) and use_usb:
                print("[INFO] USB camera: exposure / AE / AWB are set on the Pi Zero and cannot be "
                      "locked from here (key ignored).")
            elif k == ord('e'):
                ae_locked = not ae_locked
                if ae_locked:
                    # lock at the values AE ALREADY chose, not hard-coded
                    # numbers, so the brightness does not jump when locking
                    md = cam.get_metadata()
                    exp_us = int(md.get("ExposureTime") or exp_us)
                    gain_a = float(md.get("AnalogueGain") or gain_a)
                    cam.set_controls(AeEnable=False, AwbEnable=False,
                                     ExposureTime=int(exp_us), AnalogueGain=float(gain_a))
                    print(f"[INFO] AE/AWB LOCKED: exp={exp_us}us gain={gain_a:.2f}")
                else:
                    cam.set_controls(AeEnable=True, AwbEnable=True)
                    if measure_mode:
                        print("[WARN] measure mode CANCELLED: AE/AWB back to auto.")
                        measure_mode = False
                    print("[INFO] AE/AWB auto (unlocked)")
            elif k in (ord('['), ord(']')) and ae_locked:
                # 10% steps -> feels the same at 2000us and at 30000us
                step = max(200, int(0.10 * exp_us))
                exp_us = int(np.clip(exp_us + (step if k == ord(']') else -step),
                                     100, exp_cap))
                cam.set_controls(ExposureTime=int(exp_us))
                note = "  (at the fps limit)" if exp_us >= 0.95 * exp_cap else ""
                if measure_mode:
                    note += "  [WARNING: no longer at the measurement operating point]"
                print(f"[INFO] exposure={exp_us}us{note}")
            elif k in (ord('-'), ord('=')) and ae_locked:
                gain_a = max(1.0, gain_a + (0.5 if k == ord('=') else -0.5))
                cam.set_controls(AnalogueGain=float(gain_a))
                note = "  [WARNING: no longer at the measurement operating point]" if measure_mode else ""
                print(f"[INFO] gain={gain_a:.1f}  (higher gain = higher sigma_n = "
                      f"higher measured noise floor){note}")
            elif k == ord('g'):
                if not slog.recording:
                    slip_prev_D = slip_prev_E = slip_t_prev = None
                    trial = slog.start(time.monotonic())
                    print(f"[SLIP] RECORDING trial {trial}, label=static. "
                          f"'k' = mark slip, 'g' = stop.")
                    if reb_hold > 0:
                        print(f"[WARN] 'rebase hold'={reb_hold} is still active. "
                              f"It is skipped while recording, but it is safer "
                              f"to set the trackbar to 0.")
                    if supp_strength > 0:
                        print(f"[INFO] center_suppress={supp_strength}% is active, "
                              f"but slip features are computed BEFORE suppression "
                              f"-> unaffected.")
                else:
                    print(f"[SLIP] STOP trial {slog.stop()} -> {slog.path}")
            elif k == ord('l'):
                if dlog is None:
                    os.makedirs(os.path.join(CALIB_DIR, "logs"), exist_ok=True)
                    dlog_path = os.path.join(CALIB_DIR, "logs",
                                             time.strftime(f"depthlog_{cam_tag}_%Y%m%d_%H%M%S.csv"))
                    dlog = open(dlog_path, "w")
                    dlog.write("t_wall,t_host,seq,depth_max,depth_p99,contact_frac\n")
                    print(f"[LOG] depth per frame -> {dlog_path}  (start the MTS run now; 'l' again = stop)")
                else:
                    dlog.close(); dlog = None
                    print(f"[LOG] stop -> {dlog_path}\n      fit: python3 mts_calib.py {dlog_path} "
                          f"<file_MTS.txt> --out={calib_file}")
            elif k == ord('p'):
                dcal.add_point()
            elif k == ord('z'):
                dcal.undo()
            elif k == ord('o'):
                os.makedirs(CALIB_DIR, exist_ok=True)
                _cp = calib_file
                dcal.fit(_cp, meta=dict(camera=cam_tag if use_usb else f"csi cam{cam_id}",
                                        profile=pk, size=[profile["cw"], profile["ch"]],
                                        flow=flow_method or "default", lens=lens))
            elif k == ord('v'):
                if raw_t0 is None:
                    os.makedirs(raw_out, exist_ok=True)
                    raw_folder = os.path.join(raw_out, time.strftime("rec_%Y%m%d_%H%M%S"))
                    os.makedirs(raw_folder, exist_ok=True)
                    raw_rec.start_rec(raw_folder); raw_t0 = time.monotonic()
                    print(f"[REC] START raw recording -> {raw_folder}/  ('v' again to stop)")
                else:
                    finish_recording(raw_folder, [raw_rec], raw_meta, raw_fmt, ae_locked)
                    raw_t0 = None
            elif k == ord('k'):
                _lab = slog.toggle_label()
                print(f"[SLIP] label -> {'SLIP' if _lab else 'static'}"
                      + ("" if slog.recording else "  (not recording; 'g')"))
            elif k == ord('s') and last_panel is not None:
                ts = time.strftime("%Y%m%d_%H%M%S"); fn = f"snap_fast_{ts}.png"
                cv2.imwrite(fn, last_panel)                         # display panel (as before)
                d = f"raw_{ts}"
                has_roi = save_raw_bundle(d, 0, frame, res)         # RAW frame + raw ROI
                write_meta(d, cam, cfg, pk, profile, ae_locked, measure_mode, 1,
                           [int(last_seq)], note="single snap")
                print(f"[INFO] display snap: {fn} | raw: {d}/raw_00.png"
                      f"{' + roi_00.png' if has_roi else ' (no ROI)'}"
                      f" ({frame.shape[1]}x{frame.shape[0]})")
            elif k in (ord('b'), ord('S')):          # b = burst (no Shift)
                if burst_left > 0:
                    print("[SNAP] burst still running.")
                else:
                    burst_dir = f"raw_{time.strftime('%Y%m%d_%H%M%S')}"
                    burst_left, burst_seqs, burst_recs, burst_skip = BURST_N, [], [], BURST_SKIP
                    if not measure_mode:
                        print("[SNAP] WARNING: measure mode is not active — press 'u' first for calibration.")
                    print(f"[SNAP] burst {BURST_N} frames (discard {BURST_SKIP} first) -> {burst_dir}/ "
                          f"(do not touch the sensor)")

            # ─── MEASURE MODE: zero everything that interferes + lock the camera ───
            elif k in (ord('u'), ord('M')):          # u = measure mode (no Shift)
                for name, val in MEASURE_SAFE.items():
                    cv2.setTrackbarPos(name, TB, val)
                exp_us, gain_a = MEASURE_EXP_US, MEASURE_GAIN
                cam.set_controls(AeEnable=False, AwbEnable=False,
                                 ExposureTime=int(exp_us), AnalogueGain=float(gain_a),
                                 ColourGains=MEASURE_COLOUR_GAINS,
                                 NoiseReductionMode=0,     # off: NR = spatial filter
                                 Sharpness=0.0)            # off: sharpening = spatial filter
                ae_locked = True
                measure_mode = True
                t_measure = time.time()
                print(f"[INFO] MEASURE MODE on: suppression/rebaseline/runtime OFF; "
                      f"camera locked exp={exp_us}us gain={gain_a} "
                      f"colour={MEASURE_COLOUR_GAINS}, NR & sharpening OFF.")
                print("[INFO] Wait for STABLE status, then 'm' to record, or 'n' for the noise floor.")

            # ─── record one decal ───
            elif k == ord('m'):
                probe.toggle_record(ae_locked, supp_strength, reb_hold)

            # ─── noise floor ───
            elif k == ord('n'):
                if probe.recording:
                    print("[NF] probe is recording; cancel it first ('m').")
                else:
                    run_noise_floor(cam, cfg, profile["cw"], profile["ch"],
                                    window_box=WINDOW_BOX)

    finally:
        if raw_t0 is not None:
            finish_recording(raw_folder, [raw_rec], raw_meta, raw_fmt, ae_locked)
        raw_rec.running = False
        if dlog is not None:
            dlog.close()
        if dworker is not None:
            dworker.stop()
        cam.stop()
        if live3d is not None:
            live3d.stop()
        depth_pub.close()        # << close shared memory
        if ros is not None:
            ros.close()
        slog.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()