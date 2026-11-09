#!/usr/bin/env python3
"""Baseline models on the ML data made by grasp_to_ml.py (one classifier per future target).

    python3 grasp_train.py [--data=ml_data] [--cap=600] [--min-valid=0.8] [--folds=5] [--save]

- windows.csv rows are sliding windows; ONE very long hold would dominate the data, so at most
  --cap windows are kept per grasp attempt (random, seed 0).
- Windows with too little valid tactile data (valid_frac < --min-valid) are dropped.
- Cross-validation is GROUPED BY attempt_uid: windows of one attempt never are in train AND test.
- For every y_* target with enough positives: ROC-AUC and PR-AUC (vs the base rate) per fold,
  the top features, and with --save the model fitted on all data -> <data>/models/<target>.joblib
Needs scikit-learn (pip install scikit-learn).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:                                             # scikit-learn < 1.0
    try:
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        sys.exit("[ERROR] scikit-learn missing:  pip install scikit-learn")
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def opt(args, k, d):
    for a in args:
        if a.startswith(f"--{k}="):
            return type(d)(a.split("=", 1)[1])
    return d


def main():
    args = sys.argv[1:]
    data = opt(args, "data", "ml_data")
    cap, min_valid, folds = opt(args, "cap", 600), opt(args, "min-valid", 0.8), opt(args, "folds", 5)
    W = pd.read_csv(os.path.join(data, "windows.csv"), low_memory=False)
    n0 = len(W)
    W = W[W["valid_frac"] >= min_valid]
    keep = []
    for _, idx in W.groupby("attempt_uid").groups.items():     # same on old and new pandas
        idx = np.array(idx)
        keep.extend(np.random.default_rng(0).choice(idx, min(len(idx), cap), replace=False) if len(idx) > cap else idx)
    W = W.loc[sorted(keep)].reset_index(drop=True)
    groups = W["attempt_uid"].astype(str).to_numpy()
    n_att = len(np.unique(groups))
    print(f"[DATA] {n0} windows -> {len(W)} after valid >= {min_valid} and <= {cap} per attempt; {n_att} attempts")
    if n_att < 3:
        sys.exit("[DATA] fewer than 3 usable grasp attempts - record more grasps first")
    feats = [c for c in W.columns if "__" in c or c.endswith("_contact_now") or c in ("n_contact_now", "t_in_attempt_s")]
    feats = [c for c in feats if W[c].notna().any() and W[c].nunique(dropna=True) > 1]
    X = W[feats].to_numpy(float)
    targets = [c for c in W.columns if c.startswith("y_") and set(W[c].dropna().unique()) <= {0, 1}]
    k = min(folds, n_att)
    res = []
    for t in targets:
        y = W[t].to_numpy(int)
        pos_att = len(np.unique(groups[y == 1]))
        if y.sum() < 20 or pos_att < 2 or (1 - y).sum() < 20:
            print(f"[{t}] skipped: {y.sum()} positives in {pos_att} attempts (need >= 20 in >= 2 attempts)")
            continue
        aucs, aps = [], []
        for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
                continue
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p)); aps.append(average_precision_score(y[te], p))
        if not aucs:
            print(f"[{t}] skipped: no fold has both classes in train and test")
            continue
        base = y.mean()
        r = dict(target=t, windows=len(y), positives=int(y.sum()), pos_attempts=pos_att, base_rate=round(base, 3),
                 roc_auc=round(float(np.mean(aucs)), 3), roc_auc_sd=round(float(np.std(aucs)), 3),
                 pr_auc=round(float(np.mean(aps)), 3), folds=len(aucs))
        res.append(r)
        print(f"[{t}] ROC-AUC {r['roc_auc']:.3f} +- {r['roc_auc_sd']:.3f}   PR-AUC {r['pr_auc']:.3f} "
              f"(base rate {base:.3f})   {len(aucs)} grouped folds")
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(X, y)
        idx = np.random.default_rng(0).choice(len(y), min(len(y), 3000), replace=False)
        pi = permutation_importance(m, X[idx], y[idx], n_repeats=3, random_state=0, scoring="roc_auc")
        top = np.argsort(pi.importances_mean)[::-1][:8]
        print("      top features: " + ", ".join(f"{feats[i]} ({pi.importances_mean[i]:+.3f})" for i in top))
        if "--save" in args:
            import joblib
            os.makedirs(os.path.join(data, "models"), exist_ok=True)
            joblib.dump(dict(model=m, features=feats, target=t), os.path.join(data, "models", f"{t}.joblib"))
    if res:
        out = os.path.join(data, "baseline_results.csv")
        pd.DataFrame(res).to_csv(out, index=False)
        print(f"-> {out}")
        print("ROC-AUC 0.5 = guessing, > 0.8 useful. PR-AUC compares with the base rate (= guessing).")


if __name__ == "__main__":
    main()
