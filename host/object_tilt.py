"""OBJECT TILT from the gel contact patches ("kemiringan benda").

An elongated object (pen, bottle, bar, box edge) presses an elongated patch into every gel; the long axis of
that patch is the object's axis on that pad. Its angle is measured from the depth grid the sensor sends
(weighted second moments, cells above 25 % of the peak), per pad:

    theta   angle of the patch's long axis in the pad image, deg (-90 .. 90]
    elong   long / short axis (sqrt of the eigenvalue ratio); < 1.4 = round patch -> no direction, ignored

TILT = how far the object turned in the grasp (about the axis normal to the pads):
    a ZERO taken with the object held straight (key k in servo_grasp, or automatically when HOLD is
    reached the first time with an elongated patch) -> tilt_k = theta_k - zero_k, wrapped to +-90.
    Pads that face each other see the same turn MIRRORED -> sign -1 for the pads on the other side
    (parallel grip: the lone finger; circular grip: all +1, read per pad).
    Object tilt = elongation-weighted mean of the signed pad tilts, smoothed (EMA 0.3 s); the spread
    between pads tells whether the pads agree (a bent / soft object, or a pad that slipped).

    python3 object_tilt.py           (self-test with synthetic patches)
"""
import math

import numpy as np


def patch_axis(d, rel=0.25):
    """depth grid (rows x cols, mm) -> (theta deg, elong, n cells) or None. theta: 0 = along the image
    columns (x), + = counter-clockwise in the image (y up)."""
    if d is None:
        return None
    d = np.asarray(d, np.float32)
    top = float(d.max()) if d.size else 0.0
    if top <= 1e-4:
        return None
    w = np.where(d >= rel * top, d - rel * top, 0.0)
    n = int((w > 0).sum())
    if n < 6:
        return None
    rows, cols = d.shape
    ys, xs = np.mgrid[0:rows, 0:cols].astype(np.float32)
    ys = -ys                                                   # y up
    s = float(w.sum())
    mx, my = float((w * xs).sum() / s), float((w * ys).sum() / s)
    cxx = float((w * (xs - mx) ** 2).sum() / s)
    cyy = float((w * (ys - my) ** 2).sum() / s)
    cxy = float((w * (xs - mx) * (ys - my)).sum() / s)
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = math.sqrt(max(0.0, tr * tr / 4 - det))
    l1, l2 = tr / 2 + disc, max(1e-6, tr / 2 - disc)
    th = 0.5 * math.degrees(math.atan2(2 * cxy, cxx - cyy))
    return th, math.sqrt(l1 / l2), n


def wrap90(a):
    return (a + 90.0) % 180.0 - 90.0


class ObjectTilt:
    def __init__(self, min_elong=1.4, tau=0.1, log=print, tau_pad=0.15, on_elong=1.5, off_elong=1.25, hold_s=0.8):
        self.min_elong, self.tau, self.log = min_elong, tau, log
        # STABLE per-pad axis (log 18 Nov 15:5x: the bar jumped): the axis is smoothed as a doubled-angle
        # vector (an axis has no head), the elongation too; a pad becomes valid at elong 1.5 and stays
        # valid down to 1.25 (hysteresis); a pad that drops out for < 0.8 s keeps its last axis
        self.tau_pad, self.on_e, self.off_e, self.hold_s = tau_pad, on_elong, off_elong, hold_s
        self.trk = {}
        self.A0 = self.n0 = self.A_last = self.n_last = None
        self.zero = {}                # pad id -> theta at the zero
        self.sign = {}                # pad id -> +1 / -1
        self.tilt, self.t, self.pads = None, None, {}
        # PAD ORIENTATION per camera: how the camera image sits on the finger is not known (mounting:
        # rotated 90 deg / mirrored). theta_on_pad = s * theta_image + o, s in +-1, o in 0 / 90.
        # LEARNED: while >= 2 pads feel an elongated contact and the object takes different angles, every
        # combination is scored by how well the pads agree on ONE 3-D axis; the clearly best one is kept
        # (map_path json). Manual: set_map({1: (1, 0), 2: (-1, 90), ...}).
        self.map, self.map_path, self.map_src = {}, None, "default"
        self.cal = None

    COMBOS = [(1.0, 0.0), (-1.0, 0.0), (1.0, 90.0), (-1.0, 90.0)]
    N_VIEWS, VIEW_STEP, RATIO = 12, 3.0, 0.5            # views needed / deg the pads must change between views

    def load_map(self, path):
        import json
        import os
        self.map_path = path
        try:
            if path and os.path.exists(path):
                m = json.load(open(path))
                self.map = {int(k): (float(v[0]), float(v[1])) for k, v in m.get("map", {}).items()}
                self.map_src = f"learned {m.get('date', '')}"
        except (OSError, ValueError):
            pass

    def set_map(self, m, src="manual"):
        self.map, self.map_src = dict(m), src

    def _save_map(self):
        import json
        import time as _t
        if not self.map_path:
            return
        try:
            json.dump(dict(map={str(k): list(v) for k, v in self.map.items()}, date=_t.strftime("%Y-%m-%d %H:%M"),
                           note="theta_pad = s * theta_image + o per camera / finger id"),
                      open(self.map_path, "w"), indent=1)
        except OSError:
            pass

    def _learn_map(self, S):
        """score every orientation combination on this frame (only when the pads' angles changed)"""
        import itertools
        ids = [i for i, p in self.pads.items() if p.get("on")]
        if len(ids) < 2 or self.map_src.startswith("manual") or self.map_src.startswith("learned"):
            return                                            # LOCKED once learned (--tilt-relearn = again)
            return
        # only where the pads face DIFFERENT ways (circular grip): pads in one plane (parallel grip) cannot tell
        # a turned camera from a turned object. The learned map holds for every grip type (camera mounting).
        import twin_view as tv
        lay = S.get("layout") or [90, 210, 330]
        nn = []
        for k, f in enumerate(S.get("fingers") or []):
            if int(f.get("id") or k + 1) in ids:
                nn.append(tv.pad_corners(f, lay[k % len(lay)], 24, 24, "")[4])
        ev_n = np.linalg.eigvalsh(np.sum([np.outer(n, n) for n in nn], 0))
        if ev_n[-2] / max(1e-9, ev_n.sum()) < 0.15:
            return
        C = self.cal
        if C is None or set(C["ids"]) != set(ids):
            C = self.cal = dict(ids=sorted(ids), score={}, n=0, last=None)
        th = np.array([self.pads[i]["theta"] for i in C["ids"]])
        if C["last"] is not None and float(np.max(np.abs(((th - C["last"]) + 90) % 180 - 90))) < self.VIEW_STEP:
            return                                            # the same view again: no new information
        C["last"] = th
        base = {i: self.pads[i]["theta"] for i in C["ids"]}
        for combo in itertools.product(range(4), repeat=len(C["ids"])):
            mp = {i: self.COMBOS[c] for i, c in zip(C["ids"], combo)}
            ax = self._axes3d(S, mp, base)
            M = np.zeros((3, 3))
            for d, n, w in ax.values():
                m = np.cross(n, d)
                m /= max(1e-9, float(np.linalg.norm(m)))
                M += np.outer(m, m)
            ev = np.linalg.eigvalsh(M)
            C["score"][combo] = C["score"].get(combo, 0.0) + float(ev[0] / max(1e-9, ev.sum()))
        C["n"] += 1
        if C["n"] >= self.N_VIEWS:
            # mirroring EVERY pad gives an equally consistent (mirrored) axis -> mirror pairs are one hypothesis;
            # decide between pairs, inside a pair take the one with fewer mirrored pads
            mirror = {0: 1, 1: 0, 2: 3, 3: 2}
            pairs = {}
            for combo, v in C["score"].items():
                key = min(combo, tuple(mirror[c] for c in combo))
                pairs[key] = min(pairs.get(key, 1e9), v)
            sc = sorted(pairs.items(), key=lambda kv: kv[1])
            best, second = sc[0][1], sc[1][1]
            cand = [sc[0][0], tuple(mirror[c] for c in sc[0][0])]
            pick = min(cand, key=lambda cb: (sum(1 for c in cb if c in (1, 3)), C["score"].get(cb, 1e9)))
            new = {i: self.COMBOS[c] for i, c in zip(C["ids"], pick)}
            if best >= self.RATIO * second:
                if C["n"] >= 6 * self.N_VIEWS:                  # no clear winner after many views: start over
                    self.cal = None
                return
            if C.get("cand") != new:                          # the SAME winner twice in a row (two windows)
                self.cal = dict(ids=C["ids"], score={}, n=0, last=None, cand=new)
                return
            if True:
                self.map.update(new)
                self.map_src = "learned"
                self._save_map()
                self.log("[TILT] pad orientation learned from " + str(C["n"]) + " views: " + "  ".join(
                    f"ID{i} {'mirror ' if v[0] < 0 else ''}{'+90' if v[1] else '0'}" for i, v in new.items())
                         + f" (fit {best / C['n']:.4f} vs next {second / C['n']:.4f}, confirmed twice) -> locked")
                self.cal = None
                if self.zero:                                   # keep the zero: its axis again, new orientation
                    ax0 = self._axes3d(S, None, {i: a for i, a in self.zero.items() if i in self.pads})
                    A0, _ = self._fuse(ax0)
                    if A0 is not None:
                        self.A0, self.tilt = A0, None

    def set_sides(self, lone=None, ids=(1, 2, 3)):
        """parallel grip: the lone finger faces the pair -> mirrored (-1)"""
        self.sign = {i: (-1.0 if lone is not None and i == lone else 1.0) for i in ids}

    # ---- ONE 3-D frame for all pads (step 1): every pad's patch axis is mapped into the hand frame with the
    # finger kinematics (layout angle, rotation joint, flexion: twin_view.pad_corners), the three axes are
    # fused with an axis structure tensor (sign-free, weighted by how elongated each patch is), and the tilt
    # is the angle between the fused axis and the axis at the zero, in that one frame.
    def _axes3d(self, S, mp=None, th_in=None):
        import twin_view as tv
        lay = S.get("layout") or [90, 210, 330]
        mp = self.map if mp is None else mp
        out = {}
        for k, f in enumerate(S.get("fingers") or []):
            i = int(f.get("id") or k + 1)
            p = self.pads.get(i)
            if not p or not p.get("on") or (th_in is not None and i not in th_in):
                continue
            _, pc, av, au, pn, _, _ = tv.pad_corners(f, lay[k % len(lay)], 24, 24, S.get("pad_map") or "")
            s_, o_ = mp.get(i, (1.0, 0.0))
            a = math.radians(s_ * (p["theta"] if th_in is None else th_in[i]) + o_)
            d = math.cos(a) * au + math.sin(a) * av
            out[i] = (d / max(1e-9, float(np.linalg.norm(d))), pn, min(4.0, p["elong"]) - 1.0)
        return out

    @staticmethod
    def _fuse(ax):
        """a pad sees only the object axis PROJECTED on its own plane: pad i says A . (n_i x d_i) = 0
        (the axis lies in the plane through its normal and its patch axis). A = the direction that best
        meets all pads' constraints (smallest eigenvector of sum w m m^T), with a small pull into the pads'
        planes for the case all pads lie in one plane (parallel grip: pair + lone face each other).
        conf: 1 = the pads agree on one axis, 0 = no direction."""
        if not ax:
            return None, 0.0
        M = np.zeros((3, 3))
        for d, n, w in ax.values():
            m = np.cross(n, d)
            m /= max(1e-9, float(np.linalg.norm(m)))
            M += max(w, 0.05) * np.outer(m, m) + 0.05 * max(w, 0.05) * np.outer(n, n)
        ev, V = np.linalg.eigh(M)
        A = V[:, 0]
        conf = float(np.clip((ev[1] - ev[0]) / max(1e-9, ev[1] + ev[0]), 0.0, 1.0))
        return A, conf

    @staticmethod
    def _resid(A, d, n):
        """deg between the fused axis projected on the pad and the pad's own patch axis"""
        p = A - (A @ n) * n
        nr = float(np.linalg.norm(p))
        if nr < 1e-6:
            return 90.0
        return math.degrees(math.acos(float(np.clip(abs((p / nr) @ d), 0.0, 1.0))))

    def zero_now(self, why="key k"):
        ok = {i: p["theta"] for i, p in self.pads.items() if p.get("on")}
        if getattr(self, "used3d", False) and (len(ok) < 2 or self.A_last is None):
            self.log(f"[TILT] cannot zero: {len(ok)} pad(s) with a clear line contact, needs >= 2 (pads at their"
                     " edge / load-limited / round contacts do not count) -> hold the object more centred")
            return False
        self.zero = dict(ok)
        self.tilt = 0.0
        self.A0 = self.A_last                               # the fused 3-D axis at the zero (None: per pad)
        self.n0 = self.n_last
        self.log(f"[TILT] zero ({why}): " + "  ".join(f"ID{i} patch {a:+.1f} deg" for i, a in ok.items())
                 + " = straight")
        return True

    def reset(self):
        self.zero, self.tilt, self.t, self.pads, self.trk = {}, None, None, {}, {}
        self.A0 = self.n0 = self.A_last = self.n_last = None

    def update(self, now, grids, S=None):
        """grids: {pad id: depth grid (mm) or None} of the TOUCHING pads; S: the twin state (finger angles,
        rotations, layout) -> the pads are fused in ONE 3-D hand frame. -> state dict"""
        for i, d in grids.items():
            a = patch_axis(d)
            if a is None:
                continue
            th, el, n = a
            T = self.trk.get(i)
            c2, s2 = math.cos(math.radians(2 * th)), math.sin(math.radians(2 * th))
            w_ = min(1.0, max(0.0, (el - 1.0) / 1.5))            # a rounder patch moves the axis less
            if T is None or now - T["t"] > self.hold_s:
                T = self.trk[i] = dict(c2=c2, s2=s2, el=el, t=now, on=el >= self.on_e)
            else:
                k = 1.0 - math.exp(-max(0.0, now - T["t"]) / self.tau_pad)
                T["c2"] += k * w_ * (c2 - T["c2"])
                T["s2"] += k * w_ * (s2 - T["s2"])
                T["el"] += k * (el - T["el"])
                T["t"] = now
                T["on"] = T["el"] >= (self.off_e if T["on"] else self.on_e)
        self.pads = {}
        for i, T in list(self.trk.items()):
            if now - T["t"] > self.hold_s:
                del self.trk[i]
                continue
            self.pads[i] = dict(theta=0.5 * math.degrees(math.atan2(T["s2"], T["c2"])), elong=T["el"], on=T["on"])
        if S is not None:
            self._learn_map(S)
        self.used3d = S is not None
        ax = self._axes3d(S) if S is not None else {}
        if S is not None and len(ax) < 2:
            # ONE pad sees only the axis projected on its own plane: no 3-D angle from it (log 18 Nov 17:34:
            # a single edge pad gave +26 / -60 deg). Unknown is better than wrong.
            self.A_last = None
            return dict(valid=False, deg=None, spread=0.0, zeroed=bool(self.zero), conf=0.0, frame="3d",
                        axis=None, axis0=[round(float(x), 4) for x in self.A0] if self.A0 is not None else None,
                        n_good=len(ax), zero={i: round(a, 1) for i, a in self.zero.items()},
                        pads={i: dict(theta=round(p["theta"], 1), elong=round(p["elong"], 2), on=bool(p["on"]),
                                      tilt=None) for i, p in self.pads.items()})
        A, conf = self._fuse(ax)
        self.A_last = A
        if ax:
            nn = np.sum([n * w for _, n, w in ax.values()], 0)
            self.n_last = nn / max(1e-9, float(np.linalg.norm(nn)))
        if A is not None and self.A0 is not None:
            # tilt in the ONE frame: angle between the fused axis and the zero axis, signed about the pads'
            # mean normal at the zero (the grasp direction); per-pad residuals = how far each pad is off
            A0 = self.A0
            c_ = float(np.clip(abs(A @ A0), 0.0, 1.0))
            Ae = A if A @ A0 >= 0 else -A
            sg = 1.0 if self.n0 is None or float(np.cross(A0, Ae) @ self.n0) >= 0 else -1.0
            v = sg * math.degrees(math.acos(c_))
            if self.tilt is None or self.t is None:
                self.tilt = v
            else:
                k = 1.0 - math.exp(-max(0.0, now - self.t) / self.tau)
                self.tilt += k * (v - self.tilt)
            self.t = now
            for i, (d, n, _) in ax.items():
                self.pads[i]["tilt"] = round(self._resid(A, d, n), 1)          # per pad: how far off the fused axis
            spread = max((self.pads[i]["tilt"] for i in ax), default=0.0)
            return dict(valid=True, deg=self.tilt, spread=spread, zeroed=True, conf=round(conf, 2), frame="3d",
                        axis=[round(float(x), 4) for x in Ae], axis0=[round(float(x), 4) for x in A0],
                        zero={i: round(a, 1) for i, a in self.zero.items()},
                        pads={i: dict(theta=round(p["theta"], 1), elong=round(p["elong"], 2), on=bool(p["on"]),
                                      tilt=p.get("tilt")) for i, p in self.pads.items()})
        if A is not None:
            return dict(valid=False, deg=None, spread=0.0, zeroed=bool(self.zero), conf=round(conf, 2), frame="3d",
                        axis=[round(float(x), 4) for x in A], axis0=None, zero={},
                        pads={i: dict(theta=round(p["theta"], 1), elong=round(p["elong"], 2), on=bool(p["on"]),
                                      tilt=None) for i, p in self.pads.items()})
        vals = []
        for i, p in self.pads.items():
            if not p["on"] or i not in self.zero:
                p["tilt"] = None
                continue
            p["tilt"] = self.sign.get(i, 1.0) * wrap90(p["theta"] - self.zero[i])
            vals.append((p["tilt"], min(4.0, p["elong"]) - 1.0))
        if vals:
            v = sum(a * w for a, w in vals) / max(1e-6, sum(w for _, w in vals))
            if self.tilt is None or self.t is None:
                self.tilt = v
            else:
                k = 1.0 - math.exp(-max(0.0, now - self.t) / self.tau)
                self.tilt += k * (v - self.tilt)
            self.t = now
        spread = (max(a for a, _ in vals) - min(a for a, _ in vals)) if len(vals) >= 2 else 0.0
        return dict(valid=bool(vals), deg=self.tilt if vals else None, spread=spread, zeroed=bool(self.zero),
                    zero={i: round(a, 1) for i, a in self.zero.items()},
                    pads={i: dict(theta=round(p["theta"], 1), elong=round(p["elong"], 2), on=bool(p["on"]),
                                  tilt=None if p.get("tilt") is None else round(p["tilt"], 1))
                          for i, p in self.pads.items()})


def _bar(angle, rows=24, cols=24, length=14, width=2.2, depth=0.4, cx=0.0, cy=0.0):
    """synthetic elongated contact patch at angle (deg, image y up)"""
    ys, xs = np.mgrid[0:rows, 0:cols].astype(np.float32)
    x = xs - (cols - 1) / 2 - cx
    y = -(ys - (rows - 1) / 2) - cy
    a = math.radians(angle)
    u = x * math.cos(a) + y * math.sin(a)
    v = -x * math.sin(a) + y * math.cos(a)
    return (depth * np.exp(-(v / width) ** 2) * np.clip(1 - (u / length) ** 2, 0, 1)).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    bad = 0
    for true in (-60, -30, -10, 0, 5, 20, 45, 80):
        d = _bar(true) + rng.normal(0, 0.01, (24, 24)).astype(np.float32)
        th, el, n = patch_axis(np.clip(d, 0, None))
        err = abs(wrap90(th - true))
        bad += err > 3.0
        print(f"patch {true:+4d} deg -> {th:+6.1f} deg  elong {el:.1f}  err {err:.1f}")
    # round patch: no direction
    ys, xs = np.mgrid[0:24, 0:24]
    blob = 0.4 * np.exp(-((xs - 11.5) ** 2 + (ys - 11.5) ** 2) / 20.0)
    print("round patch elong", round(patch_axis(blob)[1], 2), "(< 1.4 -> ignored)")
    # tracker: parallel grip, pair 1 + 2, lone 3 (mirrored); object turns 0 -> 15 deg
    T = ObjectTilt(log=print)
    T.set_sides(lone=3)
    base = {1: 10.0, 2: 12.0, 3: -8.0}
    T.update(0.0, {i: _bar(a) for i, a in base.items()})
    T.zero_now("test")
    for k in range(40):
        turn = 15.0 * min(1.0, k / 10)
        s = T.update(0.05 * (k + 1), {1: _bar(base[1] + turn), 2: _bar(base[2] + turn), 3: _bar(base[3] - turn)})
    print(f"object turned 15 deg -> tilt {s['deg']:+.1f} deg, pad spread {s['spread']:.1f}")
    bad += abs(s["deg"] - 15.0) > 2.0
    print("OK" if not bad else f"{bad} FAIL")
