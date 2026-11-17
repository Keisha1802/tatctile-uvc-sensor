#!/usr/bin/env python3
"""
bench_flow.py — compare optical-flow methods on the Raspberry Pi 5
=================================================================
For each method, on consecutive frame pairs from one folder, measures:
  ms/pair   time to compute 1 flow (forward)         -> approximate FPS
  |F| mean  mean flow magnitude (px)                 -> on STATIC data = noise floor
  FB err    forward-backward consistency error (px)  -> smaller = more reliable
  FB>1px %  percentage of inconsistent pixels (>1 px)

Data:
  - a burst folder from 'b' in run_sensor_fast.py (raw_XXXXXXXX_XXXXXX/)
    -> uses roi_XX.png automatically (the same input DIS gets in the pipeline),
       falls back to raw_XX.png
  - record TWO bursts: (1) no contact -> noise floor,
                       (2) while pressing/sliding -> real motion
    (raise BURST_N in run_sensor_fast.py for more pairs)

Usage:
  python3 bench_flow.py raw_20261008_150000
  python3 bench_flow.py raw_20261008_150000 --size 320x240     # other working resolution
  python3 bench_flow.py raw_20261008_150000 --slow             # + TV-L1 & DeepFlow (slow)
  python3 bench_flow.py raw_20261008_150000 --lag 3            # compare frame t vs t-3

Needs the contrib module for RLOF/TV-L1/DeepFlow/PCAFlow:
  sudo apt install -y python3-opencv   (the Debian package already includes contrib)
  or: pip install opencv-contrib-python
Methods whose module is missing are skipped automatically.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np


# ───────────────────────── methods ─────────────────────────
def make_methods(slow):
    m = {}

    def dis(preset, refine=None):
        d = cv2.DISOpticalFlow_create(preset)
        if refine is not None:
            d.setVariationalRefinementIterations(refine)
        return lambda a, b: d.calc(a, b, None)

    m["DIS ultrafast"] = (dis(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST), "gray")
    m["DIS fast"] = (dis(cv2.DISOPTICAL_FLOW_PRESET_FAST), "gray")
    m["DIS medium"] = (dis(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM), "gray")
    m["DIS medium+ref10"] = (dis(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM, 10), "gray")

    m["Farneback"] = (lambda a, b: cv2.calcOpticalFlowFarneback(
        a, b, None, pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0), "gray")
    m["Farneback gauss"] = (lambda a, b: cv2.calcOpticalFlowFarneback(
        a, b, None, 0.5, 4, 21, 5, 7, 1.5,
        cv2.OPTFLOW_FARNEBACK_GAUSSIAN), "gray")

    of = getattr(cv2, "optflow", None)
    if of is None:
        print("[INFO] cv2.optflow (contrib) missing -> RLOF/PCAFlow/TV-L1/DeepFlow skipped.")
        return m

    if hasattr(of, "calcOpticalFlowDenseRLOF"):
        m["RLOF dense"] = (lambda a, b: of.calcOpticalFlowDenseRLOF(a, b, None), "bgr")
    if hasattr(of, "createOptFlow_PCAFlow"):
        pca = of.createOptFlow_PCAFlow()
        m["PCAFlow"] = (lambda a, b: pca.calc(a, b, None), "gray")
    if slow and hasattr(of, "DualTVL1OpticalFlow_create"):
        tvl1 = of.DualTVL1OpticalFlow_create()
        m["TV-L1"] = (lambda a, b: tvl1.calc(a, b, None), "gray")
    if slow and hasattr(of, "createOptFlow_DeepFlow"):
        deep = of.createOptFlow_DeepFlow()
        m["DeepFlow"] = (lambda a, b: deep.calc(a, b, None), "gray")
    skipped = [n for n in ("RLOF dense", "PCAFlow") + (("TV-L1", "DeepFlow") if slow else ())
               if n not in m]
    if skipped:
        print(f"[INFO] not available in this OpenCV, skipped: {', '.join(skipped)}")
    return m


# ───────────────────────── metrics ─────────────────────────
def fb_error(fw, bw):
    """|F_fw(x) + F_bw(x + F_fw(x))| : 0 when forward & backward agree."""
    h, w = fw.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = gx + fw[..., 0]
    my = gy + fw[..., 1]
    bw_w = cv2.remap(bw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = cv2.magnitude(fw[..., 0] + bw_w[..., 0], fw[..., 1] + bw_w[..., 1])
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    e = err[valid]
    return float(e.mean()), 100.0 * float((e > 1.0).mean())


def load_frames(folder, size):
    files = sorted(glob.glob(os.path.join(folder, "roi_*.png")))
    kind = "roi"
    if len(files) < 2:
        files = sorted(glob.glob(os.path.join(folder, "raw_*.png")))
        kind = "raw"
    if len(files) < 2:
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        kind = "png"
    if len(files) < 2:
        sys.exit(f"[ERROR] need at least 2 PNGs in {folder}")
    frames = []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        if im is None:
            continue
        if size:
            im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        frames.append(im)
    return frames, kind, len(files)


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--size", default=None, help="resize, e.g. 320x240")
    ap.add_argument("--lag", type=int, default=1, help="compare frame t vs t-lag")
    ap.add_argument("--slow", action="store_true", help="include TV-L1 & DeepFlow")
    ap.add_argument("--threads", type=int, default=0, help="cv2.setNumThreads (0 = default)")
    a = ap.parse_args()

    if a.threads:
        cv2.setNumThreads(a.threads)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    frames, kind, n = load_frames(a.folder, size)
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    pairs = [(i - a.lag, i) for i in range(a.lag, len(frames))]
    if not pairs:
        sys.exit("[ERROR] too few frames for this lag.")
    h, w = grays[0].shape
    print(f"[INFO] {len(frames)} frames ({kind}_*.png) {w}x{h}, lag={a.lag}, "
          f"{len(pairs)} pairs, OpenCV {cv2.__version__}, threads={cv2.getNumThreads()}\n")

    methods = make_methods(a.slow)
    hdr = f"{'method':<18}{'ms/pair':>9}{'~FPS':>7}{'|F| mean':>10}{'FB err':>9}{'FB>1px %':>10}"
    print(hdr); print("-" * len(hdr))

    for name, (fn, mode) in methods.items():
        src = grays if mode == "gray" else frames
        try:
            fn(src[pairs[0][0]], src[pairs[0][1]])        # warm-up (internal allocation)
        except Exception as e:
            print(f"{name:<18}  failed: {e}")
            continue
        t_tot, mags, fbe, fbp = 0.0, [], [], []
        for i0, i1 in pairs:
            t0 = time.perf_counter()
            fw = fn(src[i0], src[i1])
            t_tot += time.perf_counter() - t0
            bw = fn(src[i1], src[i0])                     # not timed
            mags.append(float(cv2.magnitude(fw[..., 0], fw[..., 1]).mean()))
            e, p = fb_error(fw, bw)
            fbe.append(e); fbp.append(p)
        ms = 1000.0 * t_tot / len(pairs)
        print(f"{name:<18}{ms:>9.1f}{1000.0 / ms:>7.0f}{np.mean(mags):>10.3f}"
              f"{np.mean(fbe):>9.3f}{np.mean(fbp):>10.1f}")

    print("\nHow to read:")
    print("  STATIC data (no contact): small |F| mean & FB err = low noise floor = better")
    print("  MOVING data: small FB err = consistent flow = more reliable")
    print("  ~FPS is for flow only; the real loop FPS is lower (depth, display, etc.)")


if __name__ == "__main__":
    main()
