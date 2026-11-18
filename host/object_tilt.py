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
    def __init__(self, min_elong=1.4, tau=0.3, log=print, tau_pad=0.5, on_elong=1.5, off_elong=1.25, hold_s=0.8):
        self.min_elong, self.tau, self.log = min_elong, tau, log
        # STABLE per-pad axis (log 18 Nov 15:5x: the bar jumped): the axis is smoothed as a doubled-angle
        # vector (an axis has no head), the elongation too; a pad becomes valid at elong 1.5 and stays
        # valid down to 1.25 (hysteresis); a pad that drops out for < 0.8 s keeps its last axis
        self.tau_pad, self.on_e, self.off_e, self.hold_s = tau_pad, on_elong, off_elong, hold_s
        self.trk = {}
        self.zero = {}                # pad id -> theta at the zero
        self.sign = {}                # pad id -> +1 / -1
        self.tilt, self.t, self.pads = None, None, {}

    def set_sides(self, lone=None, ids=(1, 2, 3)):
        """parallel grip: the lone finger faces the pair -> mirrored (-1)"""
        self.sign = {i: (-1.0 if lone is not None and i == lone else 1.0) for i in ids}

    def zero_now(self, why="key k"):
        ok = {i: p["theta"] for i, p in self.pads.items() if p.get("on")}
        if not ok:
            self.log("[TILT] cannot zero: no elongated contact patch (round contact has no direction)")
            return False
        self.zero = dict(ok)
        self.tilt = 0.0
        self.log(f"[TILT] zero ({why}): " + "  ".join(f"ID{i} patch {a:+.1f} deg" for i, a in ok.items())
                 + " = straight")
        return True

    def reset(self):
        self.zero, self.tilt, self.t, self.pads, self.trk = {}, None, None, {}, {}

    def update(self, now, grids):
        """grids: {pad id: depth grid (mm) or None} of the TOUCHING pads -> state dict"""
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
