#!/usr/bin/env python3
"""Live predictor for servo_grasp.py --ml (runs in the ML Python, e.g. ~/mlenv/bin/python).

servo_grasp.py starts it and writes one JSON line per log row to its stdin:
    {"t": <s>, "row": {<CSV column>: <value>, ...}}
It keeps the last --win seconds of the CURRENT grasp attempt (mode GRASP), makes the same window
features as grasp_to_ml.py (shared function), and every --every s prints one JSON line:
    {"t": <s of the last row>, "p": {"y_slip_soon": 0.12, ...}}
Models: <dir>/*.joblib written by grasp_train.py --save.

    grasp_ml_live.py --models=ml_data/models [--win=0.5] [--every=0.1]
"""
import glob
import json
import os
import sys
import warnings
from collections import deque

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grasp_to_ml as g2m  # noqa: E402


def opt(args, k, d):
    for a in args:
        if a.startswith(f"--{k}="):
            return type(d)(a.split("=", 1)[1])
    return d


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    args = sys.argv[1:]
    mdir = opt(args, "models", "ml_data/models")
    win, every = opt(args, "win", 0.5), opt(args, "every", 0.1)
    import joblib
    models = []
    for p in sorted(glob.glob(os.path.join(mdir, "*.joblib"))):
        m = joblib.load(p)
        models.append(m)
    if not models:
        print(json.dumps(dict(error=f"no models in {mdir} (grasp_train.py --save)")), flush=True)
        return
    print(json.dumps(dict(ready=[m["target"] for m in models])), flush=True)
    buf = deque()
    fing, sig, t_att, last_pred = None, None, None, -1e9
    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        row, t = msg.get("row", {}), float(msg.get("t", 0.0))
        if row.get("mode") != "GRASP":
            buf.clear(); t_att = None
            continue
        if fing is None:
            fing = sorted({k.split("_")[0] for k in row if k[:1] == "f" and k[1:2].isdigit() and "_" in k})
        for f in fing:                                        # derived like grasp_to_ml.load()
            j = int(f[1:])
            row[f"{f}_track_err_deg"] = num(row.get(f"id{j}_sent_deg")) - num(row.get(f"id{j}_act_deg"))
        if sig is None:
            sig = g2m.signals(list(row.keys()), fing)
        if t_att is None:
            t_att = t
        buf.append((t, [num(row.get(s)) for s in sig], [int(num(row.get(f"{f}_contact")) == 1) for f in fing]))
        while buf and buf[0][0] < t - win:
            buf.popleft()
        if t - last_pred < every or len(buf) < 3:
            continue
        last_pred = t
        tw = np.array([b[0] for b in buf])
        Xw = np.array([b[1] for b in buf], float)
        feat = g2m.window_features(Xw, tw, sig, np.array(buf[-1][2]), fing, t - t_att)
        p = {}
        for m in models:
            x = np.array([[feat.get(c, np.nan) for c in m["features"]]], float)
            p[m["target"]] = round(float(m["model"].predict_proba(x)[0, 1]), 3)
        print(json.dumps(dict(t=t, p=p)), flush=True)


if __name__ == "__main__":
    main()
