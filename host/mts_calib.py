#!/usr/bin/env python3
"""
mts_calib.py — sensor depth -> mm calibration with MTS data (deck method, slides 26-27)
====================================================================================
Matches the sensor depth log (key 'l' in run_sensor_fast.py) with an MTS file
("Load (N)","Time (s)","Crosshead (mm)", 0.1 mm steps every ~5.15 s), then fits
linear  mm = gain * depth_max + offset  over the range before saturation.

Time alignment: the MTS and Pi clocks do NOT need to be in sync. The time offset is
found automatically by matching the shape of the MTS displacement staircase to the
sensor depth curve (correlation); then for each MTS step the median depth over the
middle of the 'hold' is taken.

Usage:
  python3 mts_calib.py depthlog_usb_XXXX.csv exp1_test2_noreset.txt
  python3 mts_calib.py log1.csv mts1.txt log2.csv mts2.txt ...   # several trials (pooled)
  python3 mts_calib.py log.csv mts.txt --range=0.4,1.2          # manual working range (mm)
  python3 mts_calib.py log.csv mts.txt --metric=p99 --out=depth_calib_usb.json
Output: depth_calib_*.json (used by: run_sensor_fast.py ... --depth-calib=FILE) + PNG plot.
"""
import csv
import json
import sys
import time

import numpy as np


# ───────────────────────── read files ─────────────────────────
def read_mts(path):
    """-> list of steps [(t_s, disp_mm, load_N)], duplicate rows merged."""
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            try:
                load, t, disp = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            rows.append((t, disp, load))
    if not rows:
        sys.exit(f"[ERROR] no data in {path} (format: Load (N), Time (s), Crosshead (mm))")
    rows.sort()
    steps = []                                   # merge rows with almost identical times
    for t, d, l in rows:
        if steps and t - steps[-1][0] < 0.5:
            steps[-1] = (steps[-1][0], max(steps[-1][1], d), max(steps[-1][2], l))
        else:
            steps.append((t, d, l))
    return steps


def read_log(path, metric):
    t, v = [], []
    col = "depth_p99" if metric == "p99" else "depth_max"
    with open(path) as fh:
        for r in csv.DictReader(fh):
            t.append(float(r["t_wall"])); v.append(float(r[col]))
    if len(t) < 20:
        sys.exit(f"[ERROR] log {path} is too short ({len(t)} rows)")
    t = np.array(t); return t - t[0], np.array(v)


# ───────────────────────── matching ─────────────────────────
def staircase(steps, t):
    """MTS displacement as a function of time (staircase): value of the last step before t."""
    ts = np.array([s[0] for s in steps]); ds = np.array([s[1] for s in steps])
    idx = np.searchsorted(ts, t, side="right") - 1
    out = np.where(idx >= 0, ds[np.clip(idx, 0, None)], 0.0)
    out[t > ts[-1] + 6.0] = 0.0                  # after the last step: treated as released
    return out


def find_offset(steps, ts, v, dt=0.05):
    """Time offset (seconds) such that t_sensor + offset = t_MTS.

    Spearman (rank) correlation between the sensor depth and the MTS displacement
    staircase over the WHOLE sensor recording (before start & after end = 0 mm), so
    depth saturation on the upper steps does not spoil the match. Coarse search 0.25 s, then fine 0.02 s."""
    grid = np.arange(0, ts[-1], dt)
    vi = np.interp(grid, ts, v)
    rv = np.argsort(np.argsort(vi)).astype(float)
    t_mts_end = steps[-1][0] + 6.0
    lo, hi = -(t_mts_end + 5.0), ts[-1] + 5.0

    def score(off):
        d = staircase(steps, grid + off)
        d[(grid + off) < steps[0][0] - 0.5] = 0.0
        if d.std() < 1e-9:
            return -np.inf
        rd = np.argsort(np.argsort(d + 1e-9 * np.arange(d.size))).astype(float)
        return float(np.corrcoef(rv, rd)[0, 1])

    offs = np.arange(lo, hi, 0.25)
    sc = np.array([score(o) for o in offs])
    o0 = offs[int(np.nanargmax(sc))]
    fine = np.arange(o0 - 0.5, o0 + 0.5, 0.02)
    sf = np.array([score(o) for o in fine])
    i = int(np.nanargmax(sf))
    return float(fine[i]), float(sf[i])


def pair_steps(steps, ts, v, off, hold_lo=1.5, hold_hi=0.8):
    """Median sensor depth in the middle of each MTS step (ignores start & end transients)."""
    pairs = []
    for k, (t0, disp, load) in enumerate(steps):
        t1 = steps[k + 1][0] if k + 1 < len(steps) else t0 + 5.15
        a, b = t0 + hold_lo - off, t1 - hold_hi - off   # to sensor time
        m = (ts >= a) & (ts <= b)
        if m.sum() >= 3:
            pairs.append((disp, float(np.median(v[m])), float(v[m].std()), load, int(m.sum())))
    return pairs


def auto_range(pairs, min_mm=0.3):
    """Pre-saturation range: from min_mm until depth stops rising (slope < 25% of median)."""
    d = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
    s = np.gradient(y, d)
    ok = d >= min_mm
    med = np.median(s[ok & (s > 0)]) if np.any(ok & (s > 0)) else 0
    hi = d.max()
    for i in np.where(ok)[0]:
        if med > 0 and s[i] < 0.25 * med and d[i] > min_mm + 0.2:
            hi = d[i - 1]; break
    return min_mm, float(hi)


# ───────────────────────── main ─────────────────────────
def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    if len(pos) < 2 or len(pos) % 2:
        sys.exit(__doc__)
    metric = opt.get("metric", "max")
    out = opt.get("out", "depth_calib.json")
    rng = tuple(float(x) for x in opt["range"].split(",")) if "range" in opt else None

    all_pairs, trials = [], []
    for log_p, mts_p in zip(pos[0::2], pos[1::2]):
        steps = read_mts(mts_p)
        ts, v = read_log(log_p, metric)
        off, corr = find_offset(steps, ts, v)
        pairs = pair_steps(steps, ts, v, off)
        print(f"[MATCH] {log_p} <-> {mts_p}: offset {off:+.1f} s, correlation {corr:.3f}, "
              f"{len(pairs)}/{len(steps)} steps matched")
        if corr < 0.8:
            print("        WARNING: low correlation -> check that the log & MTS file are from the same run")
        trials.append(dict(log=log_p, mts=mts_p, offset_s=off, corr=corr, pairs=pairs))
        all_pairs += pairs

    lo, hi = rng if rng else auto_range(all_pairs)
    sel = [p for p in all_pairs if lo <= p[0] <= hi]
    if len(sel) < 3:
        sys.exit(f"[ERROR] only {len(sel)} points in the range {lo}-{hi} mm")
    x = np.array([p[1] for p in sel]); y = np.array([p[0] for p in sel])
    gain, offset = np.polyfit(x, y, 1)
    pred = gain * x + offset
    rmse = float(np.sqrt(np.mean((pred - y) ** 2))); mae = float(np.mean(np.abs(pred - y)))
    r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    info = dict(metric=f"depth_{metric}", gain=float(gain), offset=float(offset),
                rmse_mm=round(rmse, 4), mae_mm=round(mae, 4), r2=round(r2, 4),
                range_mm=[lo, hi], n_points=len(sel), method="MTS 0.1 mm steps, median of hold",
                trials=[dict(log=t["log"], mts=t["mts"], offset_s=t["offset_s"],
                             corr=round(t["corr"], 3)) for t in trials],
                points=[dict(disp_mm=p[0], depth=p[1], depth_std=p[2], load_N=p[3]) for p in all_pairs],
                time=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"[FIT] range {lo:.2f}-{hi:.2f} mm, {len(sel)} points: mm = {gain:.4f} * depth_{metric} "
          f"{offset:+.4f}\n      RMSE {rmse:.3f} mm | MAE {mae:.3f} mm | R2 {r2:.4f}   "
          f"(deck: RMSE 0.069 mm, MAE 0.056-0.059 mm)\n      -> {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for t in trials:
            d = [p[0] for p in t["pairs"]]; dv = [p[1] for p in t["pairs"]]
            ax[0].plot(d, dv, "o-", ms=4, label=t["mts"].split("/")[-1])
        xx = np.linspace(min(x), max(x), 50)
        ax[0].plot(gain * xx + offset, xx, "k--", label="fit (working range)")
        ax[0].axvspan(lo, hi, color="0.9")
        ax[0].set_xlabel("displacement MTS (mm)"); ax[0].set_ylabel(f"depth_{metric} sensor")
        ax[0].set_title("Depth vs displacement"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].plot(y, pred - y, "o", ms=4)
        ax[1].axhline(0, color="k", lw=.8)
        ax[1].set_xlabel("displacement MTS (mm)"); ax[1].set_ylabel("error (mm)")
        ax[1].set_title(f"Residuals: RMSE {rmse:.3f} mm, MAE {mae:.3f} mm"); ax[1].grid(alpha=.3)
        fig.tight_layout()
        png = out.rsplit(".", 1)[0] + ".png"
        fig.savefig(png, dpi=130); print(f"      plot -> {png}")
    except ImportError:
        print("      (matplotlib not installed: plot skipped)")


if __name__ == "__main__":
    main()
