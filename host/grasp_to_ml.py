#!/usr/bin/env python3
"""Turn servo_grasp.py logs (host/grasp_logs/grasp_*.csv) into ML datasets.

    python3 grasp_to_ml.py grasp_logs/grasp_*.csv  [--out=ml_data] [--win=0.5] [--stride=0.1] [--horizon=0.5]

Writes to --out:
  samples.csv   every log row (~40 Hz), numeric only: servo angles/loads/errors, per-finger tactile
                vector, controller state, plus derived columns and auto labels (see below)
  windows.csv   sliding windows (--win s, every --stride s): mean / std / min / max / slope / last of the
                main signals per finger + FUTURE targets over the next --horizon s
  attempts.csv  one row per grasp attempt (mode GRASP segment) with an automatic outcome
  README.txt    column description

Auto labels (no manual labels needed; a manual label 1/0/2 from the GUI is kept as `label`):
  n_contact            fingers whose sensor touches now
  all_contact          all fingers touch now (= the grasp goal)
  slip_any             any finger's sensor flags slip now
  y_contact_soon       (windows) a finger WITHOUT contact gets contact within the horizon
  y_lose_contact_soon  (windows) a finger WITH contact loses it within the horizon
  y_slip_soon          (windows) slip on any finger within the horizon
  y_overload_soon      (windows) any finger load > --overload Nm within the horizon
  y_all_contact_soon   (windows) all fingers touch within the horizon
Rows where the sensor data is stale (age > --stale s) get `tactile_valid` = 0 (keep or drop them).
"""
import glob, os, sys, warnings
import numpy as np
import pandas as pd
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
warnings.simplefilter("ignore", RuntimeWarning)        # nanmean of an all-NaN window (old logs)

MODES = {"IDLE": 0, "GRASP": 1, "ACTIVE": 2, "OPEN": 3, "HOLD": 4, "STOPPED": 5}
PHASES = {"open": 0, "close": 1, "squeeze": 2, "advance": 3, "miss": 4, "hold": 5, "nodata": 6}
TACT = ["contact", "area_frac", "cx", "cy", "depth_max_mm", "volume", "force", "shear_x", "shear_y",
        "twist", "slip_score", "slip", "E", "dE", "fps"]


def opt(args, k, d):
    for a in args:
        if a.startswith(f"--{k}="):
            return type(d)(a.split("=", 1)[1])
    return d


def load(path, run_id, stale):
    """Read one log. Older servo_grasp versions logged fewer columns: missing ones become NaN
    (depth_hat falls back to depth_filt / depth). Returns (None, reason) if the file is unusable."""
    try:
        d = pd.read_csv(path, low_memory=False)
    except Exception as e:                                       # empty / truncated file
        return None, f"unreadable ({e.__class__.__name__})"
    if len(d) == 0 or "mode" not in d or "t_s" not in d:
        return None, "no rows / not a servo_grasp log"
    d.insert(0, "run", run_id)
    d.insert(1, "file", os.path.basename(path))
    fing = sorted({c.split("_")[0] for c in d.columns if c[:1] == "f" and c[1:2].isdigit() and "_" in c})
    if not fing:
        return None, "no finger columns"
    for c in ("episode", "label", "object"):
        if c not in d:
            d[c] = np.nan
    for f in fing:
        if f"{f}_depth_hat_mm" not in d:
            d[f"{f}_depth_hat_mm"] = d.get(f"{f}_depth_filt_mm", d.get(f"{f}_depth_mm", np.nan))
        for c in ["phase", "sensor_age_s", "contact"] + [f"s_{t}" for t in TACT]:
            if f"{f}_{c}" not in d:
                d[f"{f}_{c}"] = np.nan
        j = int(f[1:])
        for c in ("act_deg", "sent_deg", "load_Nm"):
            if f"id{j}_{c}" not in d:
                d[f"id{j}_{c}"] = np.nan
    if d[[f"{f}_s_fps" for f in fing]].notna().sum().sum() == 0:
        print(f"  note: {os.path.basename(path)} has no tactile data (sensor not running?) - kept, tactile_valid = 0")
    d["mode_code"] = d["mode"].map(MODES).fillna(-1).astype(int)
    for f in fing:
        d[f"{f}_phase_code"] = d[f"{f}_phase"].map(PHASES).fillna(-1).astype(int)
        d[f"{f}_tactile_valid"] = ((d[f"{f}_sensor_age_s"] <= stale) & d[f"{f}_s_fps"].notna()).astype(int)
        j = int(f[1:])
        d[f"{f}_track_err_deg"] = d[f"id{j}_sent_deg"] - d[f"id{j}_act_deg"]
    for j in range(1, 7):
        if f"id{j}_act_deg" in d:
            dt = d["t_s"].diff().replace(0, np.nan)
            d[f"id{j}_vel_degps"] = d[f"id{j}_act_deg"].diff() / dt
    d["n_contact"] = sum(d[f"{f}_contact"].fillna(0).astype(int) for f in fing)
    d["all_contact"] = (d["n_contact"] == len(fing)).astype(int)
    d["slip_any"] = (sum(d[f"{f}_s_slip"].fillna(0) for f in fing) > 0).astype(int)
    d["tactile_valid"] = np.prod([d[f"{f}_tactile_valid"] for f in fing], axis=0).astype(int)
    # grasp attempts = GRASP segments; time since the attempt started
    seg = (d["mode"] != d["mode"].shift()).cumsum()
    d["attempt"] = -1
    k = 0
    d["t_in_attempt_s"] = np.nan
    for _, idx in d.groupby(seg).groups.items():
        if d.loc[idx[0], "mode"] == "GRASP":
            d.loc[idx, "attempt"] = k
            d.loc[idx, "t_in_attempt_s"] = d.loc[idx, "t_s"] - d.loc[idx[0], "t_s"]
            k += 1
    d["attempt_uid"] = np.where(d["attempt"] >= 0, d["run"].astype(str) + "_" + d["attempt"].astype(str), "")
    return d, fing


def numeric(d):
    keep = [c for c in d.columns if c in ("run", "file", "attempt_uid") or pd.api.types.is_numeric_dtype(d[c])]
    return d[keep]


def windows(d, fing, win, stride, hor, overload):
    sig = []
    for f in fing:
        j = int(f[1:])
        sig += [f"id{j}_act_deg", f"id{j}_load_Nm", f"{f}_track_err_deg", f"{f}_depth_hat_mm",
                f"{f}_s_depth_max_mm", f"{f}_s_area_frac", f"{f}_s_cx", f"{f}_s_cy", f"{f}_s_shear_x",
                f"{f}_s_shear_y", f"{f}_s_twist", f"{f}_s_slip_score", f"{f}_s_E", f"{f}_s_dE"]
    sig += [f"id{j}_act_deg" for j in range(1, 7) if f"id{j}_act_deg" in d and
            not any(f"id{j}_act_deg" == s for s in sig)]                       # base joints
    sig = [s for s in sig if s in d]
    rows = []
    for (run, att), x in d[d["attempt"] >= 0].groupby(["run", "attempt"]):
        t = x["t_s"].to_numpy()
        X = x[sig].to_numpy(float)
        con = x[[f"{f}_contact" for f in fing]].fillna(0).to_numpy(int)
        slip = x["slip_any"].to_numpy(int)
        load = x[[f"id{int(f[1:])}_load_Nm" for f in fing]].abs().to_numpy(float)
        valid = x["tactile_valid"].to_numpy(int)
        te = t[0] + win
        while te <= t[-1] - hor:
            w = (t > te - win) & (t <= te)
            fu = (t > te) & (t <= te + hor)
            if w.sum() < 3 or fu.sum() < 1:
                te += stride
                continue
            tw = t[w] - t[w].mean()
            r = dict(run=run, attempt_uid=f"{run}_{att}", t_end_s=te, t_in_attempt_s=te - t[0],
                     valid_frac=valid[w].mean())
            Xw = X[w]
            with np.errstate(all="ignore"):
                for k, s in enumerate(sig):
                    col = Xw[:, k]
                    ok = ~np.isnan(col)
                    r[f"{s}__mean"] = np.nanmean(col) if ok.any() else np.nan
                    r[f"{s}__std"] = np.nanstd(col) if ok.any() else np.nan
                    r[f"{s}__min"] = np.nanmin(col) if ok.any() else np.nan
                    r[f"{s}__max"] = np.nanmax(col) if ok.any() else np.nan
                    r[f"{s}__last"] = col[ok][-1] if ok.any() else np.nan
                    r[f"{s}__slope"] = (np.polyfit(tw[ok], col[ok], 1)[0] if ok.sum() >= 3 and np.ptp(tw[ok]) > 0
                                        else np.nan)
            now = con[w][-1]
            for i, f in enumerate(fing):
                r[f"{f}_contact_now"] = int(now[i])
            r["n_contact_now"] = int(now.sum())
            fc = con[fu]
            r["y_contact_soon"] = int(any((now[i] == 0) and fc[:, i].any() for i in range(len(fing))))
            r["y_lose_contact_soon"] = int(any((now[i] == 1) and (fc[:, i] == 0).any() for i in range(len(fing))))
            r["y_slip_soon"] = int(slip[fu].any())
            r["y_overload_soon"] = int((load[fu] > overload).any())
            r["y_all_contact_soon"] = int((fc.sum(axis=1) == len(fing)).any())
            r["y_n_contact_end"] = int(fc[-1].sum())
            rows.append(r)
            te += stride
    return pd.DataFrame(rows)


def attempts(d, fing, overload):
    out = []
    for (run, att), x in d[d["attempt"] >= 0].groupby(["run", "attempt"]):
        con = x[[f"{f}_contact" for f in fing]].fillna(0)
        nc = con.sum(axis=1)
        first = {f: (x.loc[con[f"{f}_contact"] > 0, "t_in_attempt_s"].min()) for f in fing}
        loads = x[[f"id{int(f[1:])}_load_Nm" for f in fing]].abs()
        r = dict(run=run, file=x["file"].iloc[0], attempt_uid=f"{run}_{att}",
                 label=x["label"].dropna().iloc[-1] if x["label"].notna().any() else np.nan,
                 object=x["object"].dropna().iloc[-1] if x["object"].notna().any() else "",
                 duration_s=x["t_in_attempt_s"].max(), samples=len(x),
                 max_n_contact=int(nc.max()), n_contact_end=int(nc.iloc[-1]),
                 time_all_contact_s=float((x["all_contact"] * x["t_s"].diff().fillna(0)).sum()),
                 first_all_contact_s=x.loc[x["all_contact"] > 0, "t_in_attempt_s"].min(),
                 max_load_Nm=float(loads.max().max()), overload=int((loads > overload).any().any()),
                 slip_frac=float(x["slip_any"].mean()), tactile_valid_frac=float(x["tactile_valid"].mean()))
        for f in fing:
            j = int(f[1:])
            r[f"{f}_first_contact_s"] = first[f]
            r[f"{f}_contact_frac"] = float(con[f"{f}_contact"].mean())
            r[f"{f}_depth_end_mm"] = x[f"{f}_depth_hat_mm"].iloc[-1]
            r[f"id{j}_end_deg"] = x[f"id{j}_act_deg"].iloc[-1]
            r[f"{f}_fps_median"] = x[f"{f}_s_fps"].median()
        r["outcome_auto"] = ("all_contact" if r["n_contact_end"] == len(fing) else
                             "partial" if r["n_contact_end"] > 0 else "none")
        out.append(r)
    return pd.DataFrame(out)


def main():
    args = [a for a in sys.argv[1:] if a.startswith("--")]
    files = sorted({p for a in sys.argv[1:] if not a.startswith("--") for p in glob.glob(a)})
    if not files:
        sys.exit(__doc__)
    outd = opt(args, "out", "ml_data")
    win, stride, hor = opt(args, "win", 0.5), opt(args, "stride", 0.1), opt(args, "horizon", 0.5)
    stale, overload = opt(args, "stale", 0.5), opt(args, "overload", 0.6)
    os.makedirs(outd, exist_ok=True)
    S, W, A = [], [], []
    for i, p in enumerate(files):
        d, fing = load(p, i, stale)
        if d is None:
            print(f"{os.path.basename(p)}: skipped - {fing}")
            continue
        S.append(numeric(d))
        W.append(windows(d, fing, win, stride, hor, overload))
        A.append(attempts(d, fing, overload))
        print(f"{os.path.basename(p)}: {len(d)} rows, {d['attempt'].max() + 1} grasp attempts, {len(W[-1])} windows")
    if not S:
        sys.exit("no usable log")
    S, W, A = (pd.concat(x, ignore_index=True) for x in (S, W, A))
    S.to_csv(f"{outd}/samples.csv", index=False)
    W.to_csv(f"{outd}/windows.csv", index=False)
    A.to_csv(f"{outd}/attempts.csv", index=False)
    with open(f"{outd}/README.txt", "w") as fh:
        fh.write(__doc__ + f"\nsettings: win={win}s stride={stride}s horizon={hor}s stale={stale}s overload={overload}Nm\n"
                 f"files: {', '.join(os.path.basename(p) for p in files)}\n"
                 "Split train/test BY attempt_uid (windows of one attempt overlap -> never mix them).\n")
    print(f"-> {outd}/samples.csv {S.shape}, windows.csv {W.shape}, attempts.csv {A.shape}")
    if len(W):
        print("window targets (positive rate):", {c: round(float(W[c].mean()), 3) for c in W if c.startswith("y_")})


if __name__ == "__main__":
    main()
