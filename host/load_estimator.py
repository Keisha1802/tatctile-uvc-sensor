"""LOAD (weight) on the held object from the shear of the gel pads, and the grip it needs.

Why events, not a level
  The shear of a real gel wanders by itself (log of 17 Nov: 0.4 px in 1 s, 0.8 px in 3 s, 1.2 px in 10 s
  per channel, a few px over a minute) - as big as the load of a light object. A load measured as a LEVEL
  against a zero drifts with it. But a load arrives as a STEP: when the object is lifted / hung on, every
  pad is pulled at the same time, within a second. So:
    - every pad keeps its own short history; the change over the last --step-win (0.6 s) is compared with
      what that change is while calm (learned noise, per pad)
    - ONSET of a load = >= 2 pads (all if fewer touch) jump together beyond their noise for 0.15 s while
      the hand itself holds still. The load is LATCHED when the step has settled (0.5 s later):
      W = mean over the pads of |shear - shear before the step| (px), a pad far off the others left out
    - while loaded: a further step in the same direction = more load (re-latched), a step back by more
      than half of W = released (object put down), --> the zero follows the gel again
    - the slow wander during the hold does not change the latched W (it is not a step)
    - shear the HAND makes itself (fingers / base moving: hand_moving=True) is never a load: the
      history is re-anchored after it
  W_live (|shear - zero| now) is reported too, for the display; the grip uses the latched W.

The grip it needs (friction, no ML)
  To hold a weight the friction must carry it: grip ~ W / mu. The controller asks for
      need = sf * rho * max(0, W - w_dead)        (fraction of the base grip, capped by --tg-max)
  rho = grip fraction per px of load (~ 1/mu). It starts from the value learned for this object (saved
  per object) or the prior (0.15 / px = the old feed-forward gain). A slip under load teaches it: the
  grip was too low for that load -> rho >= (grip at the slip + step) / (sf * (W - w_dead)). It only goes
  up during a grasp (the safe side) and is saved at the end of the grasp.
"""
import json
import math
import os
import time
from collections import deque

import numpy as np


def _robust_mean(v):
    """Mean of the pads, but a pad far from the others (> 2 x / < 0.5 x their median) is left out."""
    v = np.asarray(v, float)
    if len(v) < 3:
        return float(v.mean()) if len(v) else 0.0
    m = float(np.median(v))
    keep = v[(v <= 2.0 * m + 0.3) & (v >= 0.5 * m - 0.3)]
    return float(keep.mean()) if len(keep) else m


class LoadEstimator:
    def __init__(self, win=0.6, on_k=2.6, min_step=0.5, settle=0.5, sig_floor=0.06, log=print):
        self.win, self.on_k, self.min_step, self.settle = win, on_k, min_step, settle
        self.sig_floor, self.log = sig_floor, log
        self.reset()

    def reset(self):
        self.hist = {}              # pad id -> deque[(t, (2,) shear)] of its own frames
        self.sig = {}               # pad id -> RMS of its calm 'win' change (px)
        self.raw, self.last_raw, self.prev_raw, self.sc = {}, {}, {}, {}
        self.c, self.c_n = {}, {}                 # squeeze compensation per pad (px per mm)
        self.t_now, self.slip_now = 0.0, False
        self.n_sig = {}
        self.zero = {}              # pad id -> shear before the load (valid while loaded)
        self.dirs = {}              # pad id -> unit direction of its load step
        self.loaded = False
        self.W = 0.0                # latched load (px)
        self.W_live = 0.0
        self.t_on = None            # a step is building up since
        self.t_step = None          # a step was accepted at (latching after 'settle')
        self.pre = {}               # shear before the step being latched
        self.t_quiet = -1e9         # last hand motion / slip
        self.t_latch = -1e9         # last latched step
        self.n_pads = 0
        self.events = []


    def zero_now(self, pads=None):
        """The current shear = no load (key n)."""
        self.loaded, self.W, self.zero, self.dirs, self.t_on, self.t_step = False, 0.0, {}, {}, None, None

    def _at(self, i, t):
        """Shear of pad i at time t (its last frame not newer than t)."""
        h = self.hist[i]
        best = None
        for tt, v in h:
            if tt <= t:
                best = v
            else:
                break
        return best

    def _squeeze_comp(self, i, v):
        """Shear the GRIP itself makes, taken out INCREMENTALLY: per pad c (2,) = px of shear per mm of
        depth, learned from frames where the depth changed (the grip moved) - first from the closing of
        the grasp, later slowly and only when a change fits it (a load step never teaches it).
        compensated shear += d(shear) - c * d(depth): a later change of c never makes a jump."""
        if len(v) < 3:
            return v[:2]
        prev = self.prev_raw.get(i)
        self.prev_raw[i] = v.copy()
        if prev is None:
            self.sc[i] = v[:2].copy()
            return self.sc[i].copy()
        ds, dd = v[:2] - prev[:2], float(v[2] - prev[2])
        c = self.c.get(i, np.zeros(2))
        self.sc[i] = self.sc[i] + ds - c * dd
        # learn c from the change over ~0.3 s
        h = self.raw.setdefault(i, deque())
        h.append((self.t_now, v.copy()))
        while h and self.t_now - h[0][0] > 0.35:
            h.popleft()
        if len(h) >= 3 and self.t_now - h[0][0] >= 0.25 and not self.slip_now and not self.loaded \
                and self.t_on is None and self.t_step is None:
            DD = float(v[2] - h[0][1][2])
            if abs(DD) > 0.012:
                DS = v[:2] - h[0][1][:2]
                n_ = self.c_n.get(i, 0)
                res = DS - c * DD
                fits = n_ < 30 or float(np.hypot(*res)) < 3.0 * max(self.sig.get(i, 0.25), self.sig_floor) + 0.2
                if fits:
                    k = 1.0 / (n_ + 1) if n_ < 30 else 0.02
                    self.c[i] = c + k * (DS / DD - c) * min(1.0, abs(DD) / 0.03)
                    self.c_n[i] = n_ + 1
        return self.sc[i].copy()

    def update(self, now, pads, hand_moving=False, slipping=False):
        """pads: {id: (sx, sy[, depth_mm])} of the pads in contact (newest frame of each). With the depth,
        the shear the grip itself makes (squeeze) is compensated. Returns the state dict."""
        self.t_now, self.slip_now = now, slipping
        for i, v in pads.items():
            h = self.hist.setdefault(i, deque())
            v = np.asarray(v, float)
            if h and len(v) >= 3 and i in self.last_raw and np.all(self.last_raw[i] == v):
                continue
            self.last_raw[i] = v.copy()
            v = self._squeeze_comp(i, v)
            if not h or np.any(h[-1][1] != v):
                h.append((now, v))
            while h and now - h[0][0] > 4.0:
                h.popleft()
        for i in list(self.hist):
            if i not in pads and self.hist[i] and now - self.hist[i][-1][0] > 0.5:
                self.hist.pop(i)
                self.sig.pop(i, None)
                self.n_sig.pop(i, None)
                self.zero.pop(i, None)
                self.dirs.pop(i, None)
        ids = [i for i in pads if i in self.hist and len(self.hist[i]) >= 2]
        self.n_pads = len(ids)
        if hand_moving or slipping:
            self.t_quiet = now
        if not ids:
            self.loaded, self.W, self.W_live = False, 0.0, 0.0
            return self.state()
        cur = {i: self.hist[i][-1][1] for i in ids}
        # the change over the window, per pad
        dlt = {}
        for i in ids:
            p = self._at(i, now - self.win)
            if p is not None and now - self.hist[i][0][0] >= self.win:
                dlt[i] = cur[i] - p
        calm = not hand_moving and not slipping and now - self.t_quiet > self.win + 0.2
        # learn the calm noise of the window change (not during / right after steps and hand motion)
        mg = {i: float(np.hypot(*d)) for i, d in dlt.items()}
        ready = {i for i in mg if self.n_sig.get(i, 0) >= 80}
        busy = sum(1 for i in mg if i in ready and mg[i] > 1.5 * max(self.sig.get(i, 0.25), self.sig_floor)) >= min(2, len(mg))
        if calm and self.t_on is None and self.t_step is None and not busy:
            for i, m in mg.items():
                # first 2 s: a plain average (it must start right); then slow (~8 s) and clipped, so a step
                # never teaches it
                n_ = self.n_sig.get(i, 0) + 1
                self.n_sig[i] = n_
                if n_ <= 80:
                    s0 = self.sig.get(i, 0.0)
                    self.sig[i] = math.sqrt(((n_ - 1) * s0 * s0 + m * m) / n_)
                else:
                    s0 = self.sig[i]
                    k = 0.003
                    self.sig[i] = math.sqrt((1 - k) * s0 * s0 + k * min(m * m, (2.0 * max(s0, self.sig_floor)) ** 2))
        if len(ready) < min(2, len(mg)):
            self.t_on = None                                     # noise not known yet: no load detection
            mg = {}
        thr = {i: max(self.on_k * max(self.sig.get(i, 0.25), self.sig_floor), self.min_step) for i in dlt}
        # hysteresis: once a step is building it only has to stay above 0.7 x the threshold
        hold_k = 0.7 if self.t_on is not None else 1.0
        jump = [i for i in mg if mg[i] > hold_k * thr[i]]
        # a load pulls ALL pads: besides >= 2 pads over their threshold, the pads together must be far out
        # of their noise (sum of squared z >= 3 x on_k^2) - two pads wandering by chance are not enough
        z2 = sum((mg[i] / max(self.sig.get(i, 0.25), self.sig_floor)) ** 2 for i in mg)
        if z2 < hold_k * 3.0 * self.on_k ** 2:
            jump = []
        need_n = min(2, len(ids))
        # hand moving: what changes now is the hand's own doing -> re-anchor the zero (keep the latched W)
        if hand_moving:
            self.t_on = None
            if self.t_step is not None:
                self.t_step = None
            if self.loaded:
                for i in ids:
                    p = self._at(i, now - 0.05)
                    if p is not None and i in self.zero:
                        self.zero[i] = self.zero[i] + (cur[i] - p)
        elif self.t_step is None:
            # refractory after a latched step: its own tail must not count as a new step
            fresh_ = now - self.t_latch > self.win + 0.2
            self.t_on = (self.t_on if self.t_on is not None else now) if (len(jump) >= need_n and calm and fresh_) else None
            if self.t_on is not None and now - self.t_on >= 0.15:
                # a step: remember the shear BEFORE it began (window + the 0.15 s it took to confirm)
                self.pre = {}
                for i in ids:
                    p = self._at(i, max(self.t_on - self.win, self.t_latch))
                    if p is not None:
                        self.pre[i] = p
                self.t_step, self.t_on = now, None
        if self.t_step is not None and now - self.t_step >= self.settle:
            # the step has settled: what is it?
            steps = {i: cur[i] - self.pre[i] for i in ids if i in self.pre}
            mags = {i: float(np.hypot(*s)) for i, s in steps.items()}
            self.t_step = None
            self.t_latch = now
            z_ok = [i for i in steps if mags[i] > 0.6 * thr.get(i, self.min_step)]
            if len(steps) >= need_n and len(z_ok) >= need_n:             # the step has stayed (no wander blip)
                if not self.loaded:
                    self.zero = dict(self.pre)
                    self.W = _robust_mean([float(np.hypot(*(cur[i] - self.zero[i]))) for i in self.zero if i in cur])
                    self.dirs = {i: steps[i] / max(mags[i], 1e-9) for i in steps}
                    self.loaded = True
                    self.events.append((now, "load", self.W))
                else:
                    # projection of the step on the load direction of every pad
                    # the STEP itself decides (the level wanders during a long hold)
                    pr = [float(steps[i] @ self.dirs[i]) for i in steps if i in self.dirs]
                    cs = [float(steps[i] @ self.dirs[i]) / max(mags[i], 1e-9) for i in steps if i in self.dirs]
                    p_ = float(np.median(pr)) if pr else 0.0
                    back = cs and float(np.median(cs)) < -0.5 and float(np.median(list(mags.values()))) > 0.35 * self.W
                    if p_ < -0.5 * self.W or back:
                        self.loaded, self.W, self.zero, self.dirs = False, 0.0, {}, {}
                        self.events.append((now, "release", 0.0))
                    elif abs(p_) > 0.15 * self.W:
                        self.W = max(0.0, self.W + p_)               # more / less load: W changes by the step
                        self.events.append((now, "change", self.W))
        # live level (display only; wanders)
        if self.loaded and self.zero:
            self.W_live = _robust_mean([float(np.hypot(*(cur[i] - self.zero[i]))) for i in self.zero if i in cur])
        else:
            self.W_live = 0.0
        return self.state()

    def state(self):
        return dict(loaded=self.loaded, W=self.W if self.loaded else 0.0, W_live=self.W_live, n_pads=self.n_pads,
                    building=self.t_on is not None or self.t_step is not None,
                    sig=float(np.median(list(self.sig.values()))) if self.sig else 0.0)


class LoadGrip:
    """need = sf * rho * max(0, W - w_dead); rho learned from slips under load, saved per object."""

    def __init__(self, path, obj="", rho0=0.15, sf=1.25, w_dead=0.3, step=0.12, log=print):
        self.path, self.obj = path, obj or "_default"
        self.rho0, self.sf, self.w_dead, self.step, self.log = rho0, sf, w_dead, step, log
        self.rho, self.src, self.W_last = rho0, "prior", 0.0
        try:
            with open(path) as f:
                d = json.load(f).get(self.obj)
            if d and d.get("rho"):
                self.rho, self.src = float(d["rho"]), f"learned {d.get('date', '?')} ({d.get('n', 0)} slips)"
            if d:
                self.W_last = float(d.get("W_last", 0.0))       # the load this object had last time
        except (OSError, ValueError):
            pass
        self.n_learn = 0
        self.W_seen = 0.0

    def need(self, W):
        return self.sf * self.rho * max(0.0, W - self.w_dead)

    def slip(self, W, grip_now):
        """A slip under load W while the grip was grip_now (fraction above base): it was too low."""
        if W <= self.w_dead + 0.2:
            return False
        r_new = (grip_now + self.step) / (self.sf * (W - self.w_dead))
        if r_new > self.rho * 1.02:
            old, self.rho = self.rho, r_new
            self.n_learn += 1
            self.log(f"[LOAD-GRIP] slip at load {W:.2f} px with grip +{100 * grip_now:.0f} % -> this object needs "
                     f"more grip per load: rho {old:.3f} -> {self.rho:.3f} per px (its friction is lower)")
            return True
        return False

    def seen(self, W):
        self.W_seen = max(self.W_seen, W)

    def save(self):
        if self.n_learn == 0 and self.W_seen <= 0.0:
            return
        try:
            with open(self.path) as f:
                allc = json.load(f)
        except (OSError, ValueError):
            allc = {}
        prev = allc.get(self.obj) or {}
        allc[self.obj] = dict(rho=round(self.rho, 4), n=int(prev.get("n", 0)) + self.n_learn,
                              date=time.strftime("%Y-%m-%d %H:%M"),
                              W_last=round(self.W_seen if self.W_seen > 0 else float(prev.get("W_last", 0.0)), 3))
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(allc, f, indent=1)
        os.replace(tmp, self.path)
        self.log(f"[LOAD-GRIP] '{self.obj}': rho {self.rho:.3f} per px"
                 + (f", load {self.W_seen:.2f} px" if self.W_seen > 0 else "") + " saved -> the next grasp of it starts there")
        self.n_learn, self.W_seen = 0, 0.0
