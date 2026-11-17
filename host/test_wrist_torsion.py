"""Offline test of wrist_torsion.py with synthetic gel data (no hardware).

Every trial draws a random hand: per-pad tangential direction, gain, twist, a squeeze direction, a pull
direction, a grip shear offset; camera 3 runs at half rate. It runs the full 4-phase calibration, then
checks (x nf = the noise factor the calibration reports when a pull pattern resembles torsion):
  zero at rest |tau| < 0.1 px x nf and not active     sign of +1 / -2      -2 / +1 = -2 +- 0.25 x nf
  scale within 15 % x nf of the true mean tangential shear     a grip change of 1.5x the calibration pulse
  reads < 15 % of a unit torsion     torsion during a grip change within 20 %     a stalled camera: within
  25 % or reported invalid     a taught pull reads < 15 %     rotational slip detected, never on a pull.

    python3 test_wrist_torsion.py          # 40 random hands
    python3 test_wrist_torsion.py 7 -v     # one hand, verbose
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wrist_torsion import WristTorsion  # noqa: E402

IDS = [1, 2, 3]


def trial(seed, verbose=False):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, 3)
    D = np.stack([np.cos(ang), np.sin(ang)], 1)          # tangential direction per pad (image frame)
    G = rng.uniform(0.6, 1.4, 3)                          # px per unit torque per pad
    TW = rng.uniform(-0.02, 0.02, 3)                      # twist (rad) per unit torque
    angr = ang + rng.uniform(1.2, 1.9, 3)
    R = np.stack([np.cos(angr), np.sin(angr)], 1)
    Q = rng.uniform(0.3, 0.8, 3)                          # squeeze shear per unit grip change
    PULL = rng.normal(size=(3, 2))
    PULL /= np.linalg.norm(PULL, axis=1, keepdims=True)
    B0 = rng.normal(0, 1.0, (3, 3))                       # shear / twist of the grip itself
    FPS = [12, 12, 6]
    DG = rng.uniform(0.7, 1.3, 3)                         # depth step per pad for a unit grip change
    logs = []
    cal_path = os.path.join(tempfile.mkdtemp(), "wrist_cal.json")
    W = WristTorsion(IDS, cal_path, log=logs.append)
    frames = {fid: None for fid in IDS}
    clock = [0.0]

    def obs(t, tau, sq, pull=0.0, slip=(), stall=()):
        out = {}
        for k, fid in enumerate(IDS):
            ft = math.floor(t * FPS[k]) / FPS[k]
            if fid not in stall and (frames[fid] is None or frames[fid]["rx"] != ft):
                sh = (B0[k, :2] + G[k] * tau * D[k] + Q[k] * sq * R[k] + pull * PULL[k]
                      + rng.normal(0, 0.05, 2))
                tw = B0[k, 2] + TW[k] * tau + rng.normal(0, 0.003)
                frames[fid] = dict(sx=sh[0], sy=sh[1], tw=tw, rx=ft, contact=True, slip=fid in slip,
                                   d=0.3 + 0.1 * DG[k] * sq + rng.normal(0, 0.005))
            out[fid] = frames[fid]
        return out

    def run(T, f):
        res, end = [], clock[0] + T
        while clock[0] < end:
            t = clock[0]
            res.append(W.update(t, obs(t, *f(t))))
            clock[0] += 0.025
        return res

    def mean_last(r):
        S = r[-16:]
        d = dict(S[-1])
        d["tau_px"] = float(np.mean([x["tau_px"] for x in S]))
        return d

    W.start_cal(0.0, {1: "camera1", 2: "camera2", 3: "camera3"})
    run(4.1, lambda t: (0, 0))
    t1 = clock[0]
    run(3.05, lambda t: (0, 1.0 if t - t1 < 1.5 else 0.0))
    t2 = clock[0]
    run(9.0, lambda t: (3.0 * math.sin(2 * math.pi * (t - t2) / 9.0), 0))
    W.finish_turn()
    run(0.1, lambda t: (0, 0))
    t3 = clock[0]
    run(6.0, lambda t: (0, 0, 3.0 * math.sin(2 * math.pi * (t - t3) / 6.0)))
    W.finish_turn()
    run(0.1, lambda t: (0, 0))
    if W.cal is None:
        return False, "calibration failed: " + " | ".join(logs[-2:])
    B0 += rng.normal(0, 0.3, (3, 3))                      # a new grasp: new grip shear
    run(0.6, lambda t: (0, 0))
    W.zero(clock[0], obs(clock[0], 0, 0))
    r = {}
    r["rest"] = mean_last(run(1.0, lambda t: (0, 0)))
    r["+1"] = mean_last(run(1.0, lambda t: (1.0, 0)))
    r["-2"] = mean_last(run(1.0, lambda t: (-2.0, 0)))
    r["squeeze"] = mean_last(run(1.0, lambda t: (0, 1.5)))
    r["+1 squeeze"] = mean_last(run(1.0, lambda t: (1.0, 1.5)))
    r["+1 cam3 stalled"] = mean_last(run(1.0, lambda t: (1.0, 0, 0, (), (3,))))
    o = run(1.0, lambda t: (0, 0, 3.0))
    r["pull"] = mean_last(o)
    slip_on_pull = any(x["rot_slip"] for x in o)
    run(1.0, lambda t: (0, 0))
    ts = clock[0]
    o = run(1.0, lambda t: (2.0 + 4.0 * (t - ts), 0, 0, (1, 2) if t - ts > 0.3 else ()))
    slip_seen = any(x["rot_slip"] for x in o)
    p1 = r["+1"]["tau_px"]
    nf = max(1.0, W.cal.get("noise_factor", 1.0))        # the calibration reports how much a pull pattern
    checks = {                                            # that resembles torsion costs in accuracy
        "zero at rest": abs(r["rest"]["tau_px"]) < 0.1 * nf and not r["rest"]["active"],
        "sign": p1 > 0 and r["-2"]["tau_px"] < 0 and r["+1"]["dir"] == 1 and r["-2"]["dir"] == -1,
        "linear": abs(r["-2"]["tau_px"] / p1 + 2) < 0.25 * nf,
        "scale": abs(p1 / (np.mean(G)) - 1) < 0.15 * nf,
        "squeeze ignored": abs(r["squeeze"]["tau_px"]) < 0.15 * nf * abs(p1),
        "torsion during squeeze": abs(r["+1 squeeze"]["tau_px"] - p1) < 0.2 * abs(p1),
        "stalled camera": ((r["+1 cam3 stalled"]["valid"] and abs(r["+1 cam3 stalled"]["tau_px"] - p1) < 0.25 * abs(p1))
                           or (not r["+1 cam3 stalled"]["valid"] and r["+1 cam3 stalled"]["why"])),
        "pull ignored": abs(r["pull"]["tau_px"]) < 0.15 * nf * abs(p1) or any("looks like torsion" in m for m in logs),
        "rot-slip detected": slip_seen,
        "no rot-slip on pull": not slip_on_pull,
    }
    if verbose:
        for k, x in r.items():
            coh = "-" if x["coh"] is None else f"{x['coh']:.2f}"
            print(f"  {k:16s} valid={x['valid']!s:5} active={x['active']!s:5} tau={x['tau_px']:+.3f} px "
                  f"snr={x['snr']:5.1f} coh={coh}")
        print("  true px for +1:", round(float(np.mean(G)), 3))
        print("  " + "  ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in checks.items()))
    # ---- zero drift: the gel creeps 0.2 px over 40 s (0.3 px/min) with no load -> still zero, and a torsion after
    # it reads right; a torsion HELD for 20 s is not drifted away
    drift = rng.normal(size=(3, 3)) * np.array([1, 1, 0.02])
    drift /= np.abs(drift[:, :2]).max()
    run(1.0, lambda t: (0, 0))
    W.zero(clock[0], obs(clock[0], 0, 0))
    B_start, t_d = B0.copy(), clock[0]

    def creep(t):
        B0[:] = B_start + 0.2 * drift * min(1.0, (t - t_d) / 40.0)
        return (0, 0)
    o = run(40.0, creep)
    d_rest = mean_last(o)
    d_false = any(x["active"] for x in o)
    d_p1 = mean_last(run(1.0, lambda t: (1.0, 0)))
    o = run(20.0, lambda t: (1.0, 0))
    d_hold = mean_last(o)
    run(1.0, lambda t: (0, 0))
    # ---- a pad loses contact for 1 s and comes back with a new shear: re-zeroed while quiet
    def lost(t, tau=0.0):
        ob = obs(t, tau, 0)
        ob[2] = dict(ob[2], contact=False)
        return ob
    end = clock[0] + 1.0
    while clock[0] < end:
        W.update(clock[0], lost(clock[0]))
        clock[0] += 0.025
    two = W.state()
    B0[1] += rng.normal(0, 0.5, 3)
    c_rest = mean_last(run(1.5, lambda t: (0, 0)))
    c_p1 = mean_last(run(1.0, lambda t: (1.0, 0)))
    # ---- save / load: a new object gives the same reading
    W2 = WristTorsion(IDS, cal_path, log=logs.append)
    run(1.0, lambda t: (0, 0))
    W2.update(clock[0], obs(clock[0], 0, 0))
    for _ in range(30):
        clock[0] += 0.025
        W2.update(clock[0], obs(clock[0], 0, 0))
    W2.zero(clock[0], obs(clock[0], 0, 0))
    end, l2 = clock[0] + 1.0, []
    while clock[0] < end:
        ob = obs(clock[0], 1.0, 0)
        l2.append(W2.update(clock[0], ob)["tau_px"])
        clock[0] += 0.025
    checks.update({
        "drift: stays zero": abs(d_rest["tau_px"]) < 0.1 * nf and not d_false,
        "drift: torsion after": abs(d_p1["tau_px"] - p1) < 0.2 * nf * abs(p1),
        "held torsion kept (20 s: fades <= 25 %)": abs(d_hold["tau_px"] - p1) < 0.25 * nf * abs(p1),
        "2 pads still valid": two["valid"] or bool(two["why"]),
        "pad back: zero": abs(c_rest["tau_px"]) < 0.12 * nf,
        "pad back: torsion": abs(c_p1["tau_px"] - p1) < 0.25 * nf * abs(p1),
        "save/load": W2.cal is not None and abs(float(np.mean(l2[-16:])) - p1) < 0.2 * nf * abs(p1),
    })
    if verbose:
        print("  " + "  ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in list(checks.items())[10:]))
    bad = [k for k, v in checks.items() if not v]
    return not bad, ", ".join(bad)


def wander_trial(seed, verbose=False, D=0.144, amp=3.0):
    """A REAL gel: every shear channel wanders like the log of 17 Nov (0.38 px in 1 s, 0.8 px in 3 s,
    random walk D = 0.144 px^2/s) on top of 0.05 px frame noise. Checks: the calibration works, 60 s
    without a turn give (almost) no false torsion, five turns of ~3 px are all felt within 0.6 s with the
    right sign and size, and each one ends within 2.5 s after the release."""
    rng = np.random.default_rng(1000 + seed)
    ang = rng.uniform(0, 2 * np.pi, 3)
    Dm = np.stack([np.cos(ang), np.sin(ang)], 1)
    G = rng.uniform(0.7, 1.3, 3)
    TW = rng.uniform(-0.01, 0.01, 3)
    angr = ang + rng.uniform(1.2, 1.9, 3)
    Rm = np.stack([np.cos(angr), np.sin(angr)], 1)
    Q = rng.uniform(0.5, 1.0, 3)
    DG = rng.uniform(0.7, 1.3, 3)
    walk = np.zeros((3, 3))
    logs = []
    W = WristTorsion(IDS, os.path.join(tempfile.mkdtemp(), "w.json"), log=logs.append)
    frames = {fid: None for fid in IDS}
    clock, last = [0.0], [0.0]
    FPS = [15, 15, 12]

    def obs(t, tau, sq):
        dtw = t - last[0]
        last[0] = t
        if dtw > 0:
            walk[:, :2] += rng.normal(0, math.sqrt(D * dtw), (3, 2))
            walk[:, 2] += rng.normal(0, math.sqrt(1e-6 * dtw), 3)
        out = {}
        for k, fid in enumerate(IDS):
            ft = math.floor(t * FPS[k]) / FPS[k]
            if frames[fid] is None or frames[fid]["rx"] != ft:
                sh = walk[k, :2] + G[k] * tau * Dm[k] + Q[k] * sq * Rm[k] + rng.normal(0, 0.05, 2)
                frames[fid] = dict(sx=sh[0], sy=sh[1], tw=walk[k, 2] + TW[k] * tau + rng.normal(0, 0.003), rx=ft,
                                   contact=True, slip=False, d=0.3 + 0.1 * DG[k] * sq + rng.normal(0, 0.005))
            out[fid] = frames[fid]
        return out

    def run(T_, f):
        res, end = [], clock[0] + T_
        while clock[0] < end:
            res.append((clock[0], W.update(clock[0], obs(clock[0], *f(clock[0])))))
            clock[0] += 0.025
        return res

    W.start_cal(0.0, {i: f"camera{i}" for i in IDS})
    run(4.1, lambda t: (0, 0))
    t1 = clock[0]
    run(3.05, lambda t: (0, 1.0 if t - t1 < 1.5 else 0.0))
    t2 = clock[0]
    run(8.0, lambda t: (1.3 * amp * math.sin(2 * math.pi * (t - t2) / 8.0), 0))
    W.finish_turn()
    run(0.1, lambda t: (0, 0))
    W.finish_turn()
    run(0.1, lambda t: (0, 0))
    if W.cal is None:
        return False, "calibration failed: " + (logs[-1] if logs else "")
    run(1.0, lambda t: (0, 0))
    W.zero(clock[0], obs(clock[0], 0, 0))
    r = run(60.0, lambda t: (0, 0))
    false_frac = float(np.mean([x["active"] for _, x in r[40:]]))
    truth = amp * float(np.mean(G))
    det, size, sign_ok, rel = [], [], [], []
    for i in range(5):
        sgn = 1.0 if i % 2 == 0 else -1.0
        t0 = clock[0]
        r = run(3.0, lambda t: (sgn * amp * min(1.0, (t - t0) / 0.5), 0))          # a turn building up in 0.5 s
        on = [t for t, x in r if x["active"] and x["dir"] == sgn]
        det.append((on[0] - t0) if on else 9.0)
        late = [x["tau_px"] for t, x in r if t - t0 > 1.0]
        size.append(float(np.mean(late)) / (sgn * truth))
        sign_ok.append(float(np.mean(late)) * sgn > 0)
        t0 = clock[0]
        r = run(6.0, lambda t: (0, 0))
        off = [t for t, x in r if not x["active"] and t - t0 > 0.3]
        still = float(np.mean([x["active"] for t, x in r if t - t0 > 2.5]))
        rel.append(still)
    checks = {
        "calibrated": True,
        "60 s rest: false torsion < 3 % of the time": false_frac < 0.03,
        "5 turns felt within 0.9 s": max(det) <= 0.9,
        "sign": all(sign_ok),
        "size: median error < 15 %, 4 of 5 within 35 %": (float(np.median([abs(x - 1.0) for x in size])) < 0.15
                                                          and sum(abs(x - 1.0) < 0.35 for x in size) >= 4),
        "ends after the release": max(rel) < 0.1,
    }
    if verbose:
        print(f"  wander: false {100 * false_frac:.1f} %  detect {[round(x, 2) for x in det]} s  "
              f"size {[round(x, 2) for x in size]}  still-active-after-release {[round(x, 2) for x in rel]}")
        print("  " + [m for m in logs if "noise while still" in m][0][:140])
        print("  " + [m for m in logs if "calibrated" in m][0][:170])
    bad = [k for k, v in checks.items() if not v]
    return not bad, ", ".join(bad)


def failure_paths():
    """Calibrations that must NOT produce a calibration."""
    out = {}
    rng = np.random.default_rng(1)

    def mk(tau_amp, n_touch=3):
        logs = []
        W = WristTorsion(IDS, os.path.join(tempfile.mkdtemp(), "c.json"), log=logs.append)
        W.start_cal(0.0, {i: f"camera{i}" for i in IDS})
        t = 0.0
        while t < 22.0 and W.calib:
            tau = tau_amp * math.sin(2 * math.pi * max(0.0, t - 7.5) / 8.0) if t > 7.5 else 0.0
            ob = {}
            for k, fid in enumerate(IDS):
                ft = math.floor(t * 12) / 12
                r_ = np.random.default_rng(int(ft * 1000) + fid)
                ob[fid] = dict(sx=tau * (k + 1) * 0.5 + r_.normal(0, 0.05), sy=r_.normal(0, 0.05),
                               tw=r_.normal(0, 0.003), rx=ft, contact=k < n_touch, slip=False, d=0.3)
            W.update(t, ob)
            if t > 19.0:
                W.finish_turn()
            t += 0.025
        return W, logs
    W, logs = mk(0.0)
    out["no turn -> FAILED"] = W.cal is None and any("FAILED" in m for m in logs)
    W, logs = mk(1.0, n_touch=1)
    out["1 pad -> cancelled"] = W.cal is None and any("cancelled" in m for m in logs)
    W, logs = mk(1.0)
    out["good turn -> calibrated"] = W.cal is not None
    W.unzero("test")
    out["unzero -> invalid"] = not W.update(30.0, {})["valid"]
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("-")]
    seeds = [int(a[0])] if a else range(40)
    fails = []
    for s in seeds:
        ok, why = trial(s, verbose="-v" in sys.argv)
        if not ok:
            fails.append(f"seed {s}: {why}")
    print(f"{len(list(seeds)) - len(fails)}/{len(list(seeds))} random hands passed")
    for f in fails:
        print("  FAIL", f)
    wf = []
    for sd in (range(10) if not a else [int(a[0])]):
        ok, why = wander_trial(sd, verbose="-v" in sys.argv)
        if not ok:
            wf.append(f"seed {sd}: {why}")
    print(f"{(10 if not a else 1) - len(wf)}/{10 if not a else 1} wandering-gel hands passed")
    for f in wf:
        print("  FAIL", f)
    fails += wf
    fp = failure_paths()
    print("failure paths: " + "  ".join(f"{k}: {'ok' if v else 'FAIL'}" for k, v in fp.items()))
    sys.exit(1 if fails or not all(fp.values()) else 0)
