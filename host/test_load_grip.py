"""Offline test of load_estimator.py with a REAL-like gel (no hardware).

Every pad: frame noise 0.05 px + its own wander (Ornstein-Uhlenbeck, fitted to the log of 17 Nov:
0.38 px in 1 s, 0.83 px in 3 s, 1.16 px in 10 s), 15 / 15 / 12 fps. A lifted weight pulls every pad along its
own (random) direction, size = gain_k x weight. Scenario: rest 90 s, lift 2 px and hold 60 s, put down,
rest 30 s, lift 4 px and hold 30 s, put down, rest 20 s; one pad stalls for 1 s during the first hold;
the base moves at 60 s; the GRIP changes the depth (and with it the shear, 15 px/mm) at 40, 90 (together
with the lift), 110 and 200 s - the estimator must compensate that, it is no load.

Checks: no false load at rest (< 2 % of the time), each lift felt (latched) within 1.3 s, the load within 35 % of the
truth (measured: 2 px load +-16 %, 4 px +-10 %, 1 sigma, from the gel's wander) during the holds and still there at the end of the 60 s hold (fades < 15 %), released within 1.5 s
after it is put down, the needed grip follows the load, a slip under load raises rho.

    python3 test_load_grip.py        (30 random hands)
    python3 test_load_grip.py 7 -v
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_estimator import LoadEstimator, LoadGrip  # noqa: E402


def trial(seed, verbose=False, s_ou=2.1, T_ou=60.0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, 3)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    gain = rng.uniform(0.8, 1.2, 3)
    base = rng.normal(0, 1.0, (3, 2))
    walk = rng.normal(0, s_ou, (3, 2))                      # Ornstein-Uhlenbeck wander fitted to the log:
                                                             # 0.38 px / 1 s, 0.83 px / 3 s, 1.16 px / 10 s
    FPS = [15, 15, 12]
    E = LoadEstimator(log=lambda m: None)
    frames = {}
    t, dt = 0.0, 0.025
    plan = [(90, 0.0), (60, 2.0), (30, 0.0), (30, 4.0), (20, 0.0)]
    squeeze_dir = rng.normal(size=(3, 2))                   # what moving the base does to the shear
    grip_dir = rng.normal(size=(3, 2)) * 15.0               # px of shear per mm of depth the grip adds
    def depth_at(tt):
        """the grip: closes 0 -> 0.30 mm (0-3 s), +0.10 mm at rest (40 s), +0.15 mm together with the first
        lift (90 s, the feed-forward), +0.08 mm during the hold (110 s), back -0.1 at 200 s"""
        d = 0.30 * min(1.0, tt / 3.0)
        for t0, dd in ((40.0, 0.10), (90.1, 0.15), (110.0, 0.08), (200.0, -0.10)):
            d += dd * min(1.0, max(0.0, (tt - t0) / 0.5))
        return d
    sq_t = (60.0, 61.0)                                      # the hand squeezes harder at rest (hand moving)
    edges, acc = [], 0.0
    for dur, w in plan:
        edges.append((acc, acc + dur, w))
        acc += dur

    def truth(tt):
        for a, b, w in edges:
            if a <= tt < b:
                ramp = min(1.0, (tt - a) / 0.6)
                prev = [e for e in edges if e[1] <= a]
                w0 = prev[-1][2] if prev else 0.0
                return w0 + (w - w0) * ramp
        return 0.0
    rec = []
    last = 0.0
    while t < acc:
        dtw = t - last
        last = t
        if dtw > 0:
            a_ = math.exp(-dtw / T_ou)
            walk = a_ * walk + rng.normal(0, s_ou * math.sqrt(1 - a_ * a_), (3, 2))
        W = truth(t)
        pads = {}
        for k in range(3):
            if 120.0 <= t < 121.0 and k == 2:
                continue                                        # camera 3 stalls 1 s during the hold
            ft = math.floor(t * FPS[k]) / FPS[k]
            if frames.get(k, (None,))[0] != ft:
                sq = 1.5 * min(1.0, max(0.0, (t - sq_t[0]) / (sq_t[1] - sq_t[0])))
                dk = depth_at(t) * (0.9 + 0.1 * k) + rng.normal(0, 0.004)
                v = (base[k] + walk[k] + gain[k] * W * dirs[k] + sq * squeeze_dir[k] + dk * grip_dir[k]
                     + rng.normal(0, 0.05, 2))
                frames[k] = (ft, np.array([v[0], v[1], dk]))
            pads[k + 1] = tuple(frames[k][1])
        s = E.update(t, pads, hand_moving=sq_t[0] <= t < sq_t[1] + 0.1)
        if abs(t - 5.0) < 0.012 or abs(t - 62.0) < 0.012:
            E.zero_now(pads)                                    # the settled, unloaded grasp (HOLD)
        rec.append((t, W, s["W"], s["loaded"]))
        t += dt
    R = np.array([(a, b, c, float(d)) for a, b, c, d in rec])
    tt, Wt, We, Ld = R.T

    def win(a, b):
        return (tt >= a) & (tt < b)
    rest = win(5, 90) | win(152, 180) | win(212, 230)
    false_frac = float(Ld[rest].mean())
    on1 = tt[(tt >= 90) & (Ld > 0.5)]
    on2 = tt[(tt >= 180) & (Ld > 0.5)]
    det1 = (on1[0] - 90) if len(on1) else 9.0
    det2 = (on2[0] - 180) if len(on2) else 9.0
    m_true = float(np.mean(gain))                          # W = the mean pad shear
    err1 = float(np.median(We[win(92, 150)]) / (2.0 * m_true) - 1.0)
    err2 = float(np.median(We[win(182, 210)]) / (4.0 * m_true) - 1.0)
    fade = float(np.mean(We[win(146, 150)]) / max(1e-6, np.mean(We[win(92, 96)])) - 1.0)
    sq_false = float(Ld[win(59, 70)].mean())
    grip_false = float(Ld[win(39, 50)].mean())
    off1 = tt[(tt >= 150) & (Ld < 0.5)]
    off2 = tt[(tt >= 210) & (Ld < 0.5)]
    rel1 = (off1[0] - 150) if len(off1) else 9.0
    rel2 = (off2[0] - 210) if len(off2) else 9.0
    # LoadGrip: need follows the load; a slip at load 2 px with too little grip raises rho
    G = LoadGrip(os.path.join(tempfile.mkdtemp(), "lg.json"), "obj", log=lambda m: None)
    n2, n4 = G.need(2.0 * m_true), G.need(4.0 * m_true)
    r0 = G.rho
    G.slip(2.0 * m_true, n2)
    checks = {
        "rest: false load < 2 %": false_frac < 0.02,
        "the base moving is no load": sq_false < 0.05,
        "a grip change (squeeze) is no load": grip_false < 0.05,
        "lifts felt within 1.3 s": max(det1, det2) <= 1.3,
        "load within 35 %": abs(err1) < 0.35 and abs(err2) < 0.35,
        "60 s hold fades < 15 %": fade > -0.15,
        "released within 1.5 s": max(rel1, rel2) <= 1.5,
        "grip follows the load": n4 > 1.8 * n2 > 0,
        "slip raises rho": G.rho > r0,
    }
    if verbose:
        print(f"  false {100 * false_frac:.2f} %  detect {det1:.2f} / {det2:.2f} s  error {100 * err1:+.0f} / "
              f"{100 * err2:+.0f} %  fade {100 * fade:+.0f} %  release {rel1:.2f} / {rel2:.2f} s  "
              f"need {n2:.2f} / {n4:.2f}  rho {r0:.3f} -> {G.rho:.3f}")
    bad = [k for k, v in checks.items() if not v]
    return not bad, ", ".join(bad)


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    seeds = [int(a[0])] if a else range(30)
    fails = []
    for s in seeds:
        ok, why = trial(s, verbose="-v" in sys.argv)
        if not ok:
            fails.append(f"seed {s}: {why}")
    print(f"{len(list(seeds)) - len(fails)}/{len(list(seeds))} hands passed")
    for f in fails:
        print("  FAIL", f)
    sys.exit(1 if fails else 0)
