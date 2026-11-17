#!/usr/bin/env python3
"""
make_synth_calib.py — APPROXIMATE depth -> mm calibration from SYNTHETIC data + MTS files
===================================================================================
The original sensor data is lost, so the sensor depth log is GENERATED (synthetic) from
the MTS files with a model anchored to numbers in the deck, then fitted with mts_calib.py.

Anchors used (all from the deck / MTS files, no other invented numbers):
  1. Table VI : 500 g dead weight (4.90 N) -> mean depth_max ~0.53 (not saturated)
  2. MTS file : displacement at force = 4.90 N (interpolated load-displacement curve)
     -> gain = displacement / 0.53   (per trial, then averaged)
  3. Table VII: working range 0.40-1.20 mm (depth saturates above that),
                RMSE ~0.069 mm -> per-step scatter set to this value
THIS RESULT IS APPROXIMATE: the gain assumes the 500 g indenter = the MTS indenter.
Not measured data -> do not use it as a result in a paper/patent.

Usage:
  python3 make_synth_calib.py exp1_test1_noreset.txt exp1_test2_noreset.txt exp1_test3_noreset.txt
  -> synth_log_*.csv, depth_calib_synth.json (+ .png)
  then: python3 run_sensor_fast.py usb --depth-calib=depth_calib_synth.json
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mts_calib as mc  # noqa: E402

F_500G = 0.5 * 9.80665     # N (Table VI)
DEPTH_500G = 0.53          # mean depth_max (Table VI)
RANGE = (0.40, 1.20)       # mm, working range (Table VII)
RMSE_DECK = 0.069          # mm (Table VII)
FPS = 15.0                 # USB camera
SEED = 0


def disp_at_force(steps, f):
    d = np.array([s[1] for s in steps]); L = np.array([s[2] for s in steps])
    i = int(np.argmax(L >= f))
    if i == 0:
        return float(d[0])
    return float(np.interp(f, [L[i - 1], L[i]], [d[i - 1], d[i]]))


def main():
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not files:
        sys.exit(__doc__)
    rng = np.random.default_rng(SEED)
    trials = [(f, mc.read_mts(f)) for f in files]

    gains = []
    for f, st in trials:
        dd = disp_at_force(st, F_500G)
        gains.append(dd / DEPTH_500G)
        print(f"[ANCHOR] {os.path.basename(f)}: {F_500G:.2f} N at {dd:.3f} mm -> "
              f"gain {dd / DEPTH_500G:.3f} mm/depth")
    g = float(np.mean(gains))
    print(f"[ANCHOR] mean gain {g:.3f} mm per depth unit "
          f"(spread between trials {min(gains):.2f}-{max(gains):.2f})")

    args = []
    for f, st in trials:
        lead = 5.0
        T = st[-1][0] + lead + 8.0
        ts = np.arange(0.0, T, 1.0 / FPS)
        disp = mc.staircase(st, ts - lead)
        disp[(ts - lead) < st[0][0] - 0.5] = 0.0
        k = max(1, int(0.4 * FPS))
        disp = np.convolve(disp, np.ones(k) / k, mode="same")       # step transitions
        sat = RANGE[1]
        depth = np.where(disp <= sat, disp / g, sat / g + 0.08 * (disp - sat) / g)
        # per-step scatter (in mm) ~ deck RMSE, plus per-frame jitter (CV 0.17 %)
        step_idx = np.searchsorted([s[0] for s in st], ts - lead, side="right")
        step_err = rng.normal(0.0, RMSE_DECK, len(st) + 2)[np.clip(step_idx, 0, len(st) + 1)] / g
        depth = depth + np.where(disp > 0.02, step_err, 0.0)
        depth = np.clip(depth, 0.0, None) * (1 + rng.normal(0, 0.0017, ts.size))
        out = f"synth_log_{os.path.splitext(os.path.basename(f))[0]}.csv"
        with open(out, "w") as fh:
            fh.write("t_wall,t_host,seq,depth_max,depth_p99,contact_frac\n")
            for i, (t, d) in enumerate(zip(ts, depth)):
                fh.write(f"{t:.4f},{t:.4f},{i},{d:.6f},{0.97 * d:.6f},0\n")
        args += [out, f]
        print(f"[SYNTHETIC] {out} ({ts.size} frames @ {FPS:.0f} fps)")

    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mts_calib.py"),
           *args, f"--range={RANGE[0]},{RANGE[1]}", "--out=depth_calib_synth.json"]
    subprocess.run(cmd, check=True)

    import json
    with open("depth_calib_synth.json") as fh:
        info = json.load(fh)
    info["source"] = ("SYNTHETIC: sensor log generated from MTS files, gain anchored to Table VI "
                      "(500 g -> depth 0.53) & the MTS load-displacement curve. Approximate, not a measurement.")
    info["anchor_gains"] = gains
    with open("depth_calib_synth.json", "w") as fh:
        json.dump(info, fh, indent=2)
    print("[OK] depth_calib_synth.json (labelled SYNTHETIC)")


if __name__ == "__main__":
    main()
