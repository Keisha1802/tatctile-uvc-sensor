"""WRIST TORSION from the three gel fingers: feel the wrist turning against the held object.

What it measures
  The torque the hand transmits to the object around the wrist axis (and the rotational slip when the
  grip cannot transmit it). A wrist turn that the object follows freely creates no load on the gels and
  cannot be felt - the wrist ANGLE comes from the arm encoder; this module gives the torque, its
  direction, how fast it changes, and whether the object starts to turn in the hand.

How (no geometry needed)
  Every finger sends shear x/y (mean gel flow in the contact) and twist (in-plane rotation of the
  contact). A torque around ANY wrist axis moves these 9 channels in one fixed pattern. A short
  calibration learns that pattern:
    1. REST    (2 s)    baseline + noise of every channel (the shear of the grip itself is the zero)
    2. SQUEEZE (3 s)    the hand squeezes +40 % for 1.5 s and relaxes (done by servo_grasp): the pattern a grip
                        change makes is learned and later projected OUT, so a grip change is no torque
    3. TURN    (<= 12 s) turn the wrist (or the object) one way, back, the other way, back.
                        The first principal component of the noise-whitened channels is the torsion
                        pattern u; + = the direction turned first.
    4. OTHER   (optional, <= 10 s) pull / push / lift without turning: up to 3 more load patterns.
  Live: amplitude a along u by least squares on the fresh cameras only (a stalled camera just drops out),
  the squeeze pattern estimated jointly and removed. Units:
    tau_z   amplitude in noise sigmas (detection: |tau_z| * |u_avail| > --wrist-on)
    tau_px  mean tangential shear per pad, px (or mm with --gel-mm in the sensor)
    tau_nmm torque in N*mm if --wrist-nmm-per-px is given (fit it once against a force/torque sensor)
  coherence = how much of the deviation is torsion (1) vs something else, e.g. a pull (0).

Robustness
  - the squeeze pattern comes from each pad's shear regressed on its OWN depth of the same frame
  - the noise is measured at the timescale that matters: frame noise + the gel's own wander (a real gel
    moves 0.4 px in 1 s, 0.8 px in 3 s by itself - 5-10 x its frame noise); the thresholds also follow
    what the estimate really fluctuates by while nothing is felt
  - the zero follows the gel (time constant drift_s = 2 s) while no torsion is felt; at the onset of a
    torsion it goes back to its value of 0.6 s before; while a torsion is felt it leaks with hold_s =
    120 s, so a held torsion fades slowly (-15 % in 20 s) and a wrong zero cannot stay for ever
  - torsion counts as "active" after 0.15 s above the threshold (one noisy frame is no torsion)
  - a pad that loses contact drops out; back in contact it gets a new zero once it has settled (>= 3 of its
    own frames) and the other pads feel no load; a stalled camera keeps its zero and just drops out meanwhile

Rotational slip (ROT-SLIP)
  >= 2 pads report slip and their recent shear change follows the torsion pattern, or the torsion drops
  sharply from a high level while a pad slips. The direction of the slip is reported too.
"""
import json
import math
import os
import time
from collections import deque

import numpy as np

CH = ("shear_x", "shear_y", "twist")          # channels per finger
NCH = len(CH)


def _robust_sigma(x):
    x = np.asarray(x, float)
    if x.size < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


class WristTorsion:
    REST_S, SQ_S, SQ_UP_S, SQ_GAIN, TURN_MAX_S, OTHER_MAX_S = 4.0, 3.0, 1.5, 0.4, 12.0, 10.0
    REWIND_S = 0.6                  # s: at the onset of a torsion the zero goes back to what it was this long ago

    def __init__(self, finger_ids, cal_path, obj="", on=5.0, off=3.0, tau_s=0.12, rate_s=0.25,
                 nmm_per_px=0.0, names=("+", "-"), age_max=0.3, drift_s=2.0, hold_s=120.0, log=print):
        self.ids = list(finger_ids)
        self.n = len(self.ids) * NCH
        self.cal_path, self.obj = cal_path, obj or "default"
        self.on, self.off, self.tau_s, self.rate_s = on, off, tau_s, rate_s
        self.nmm = nmm_per_px
        self.drift_s = drift_s          # s: the zero follows the gel's own wander while NO torsion is felt (0 = off)
        self.hold_s = hold_s            # s: while a torsion / other load IS felt the zero still leaks this slowly
        self.t_quiet = None             #    (0 = frozen), so a wrong zero can never stay for ever
        self.sd_run = 0.0               # running fluctuation of the torsion estimate while nothing is felt
        self.bhist = deque()            # (t, zero) of the last 3 s, for the rewind at an onset
        self.names = names
        self.age_max = age_max
        self.log = log
        self.cal = None                 # dict(u, v, sigma, c_px, cams, ...)
        self.b = None                   # baseline (raw units) after zero()
        self.calib = None               # running calibration state
        self.bhist = deque()
        self.hist = deque()             # (t, z) for slip analysis
        self.lost = set()
        self.fbuf = {fid: deque() for fid in self.ids}   # per finger: (t, rx, [sx, sy, tw]) of DISTINCT frames
        self.reset_live()
        self.load()

    # ------------------------------------------------------------------ persistence
    def load(self):
        try:
            with open(self.cal_path) as f:
                allc = json.load(f)
        except (OSError, ValueError):
            return
        c = allc.get(self.obj) or allc.get("default")
        if not c or c.get("ids") != self.ids:
            return
        self.cal = dict(c, u=np.array(c["u"]), v=None if c.get("v") is None else np.array(c["v"]),
                        sigma=np.array(c["sigma"]), P=c.get("P") or [])
        self.log(f"[WRIST] calibration loaded ({c.get('object', 'default')}, {c.get('date', '?')}, "
                 f"quality {c.get('quality', 0):.2f}, SNR {c.get('snr', 0):.0f}) - press z (or HOLD) to zero")

    def save(self):
        try:
            with open(self.cal_path) as f:
                allc = json.load(f)
        except (OSError, ValueError):
            allc = {}
        c = dict(self.cal)
        c["u"] = c["u"].tolist()
        c["v"] = None if c.get("v") is None else c["v"].tolist()
        c["sigma"] = c["sigma"].tolist()
        allc[self.obj] = c
        if self.obj != "default":
            allc.setdefault("default", c)
        tmp = self.cal_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(allc, f, indent=1)
        os.replace(tmp, self.cal_path)

    # ------------------------------------------------------------------ helpers
    def reset_live(self):
        self.a = self.a_f = self.rate = 0.0
        self.t_prev = None
        self.valid = False
        self.active = False
        self.coh = None
        self.tau_px = self.tau_nmm = 0.0
        self.snr = 0.0
        self.n_cams = 0
        self.rot_slip = False
        self.slip_dir = 0
        self.t_slip = -1e9
        self.n_slip_events = 0
        self.peak = deque()
        self.t_quiet = None
        self.n_on = 0.0
        self.a_peak = 0.0
        self.t_rel = -1e9
        self.t_evt = -1e9
        self.leak = np.zeros(self.n)
        self.a_peak_last = 0.0
        self.ep_sign = 0.0
        self.sd_run = 0.0
        self.bhist.clear()
        self.why = "not calibrated (w)" if self.cal is None else "not zeroed (z / HOLD)"
        self.hist.clear()

    def _vector(self, obs, now):
        """obs: {finger id: dict(sx, sy, tw, rx, contact, slip, cam)} -> raw x (nan where missing),
        availability mask per channel, number of fresh touching cameras, slipping fingers."""
        x = np.full(self.n, np.nan)
        n_ok, slips = 0, []
        self.lost = set()                # fresh frame that says: no contact (a stale camera is NOT lost)
        for k, fid in enumerate(self.ids):
            o = obs.get(fid)
            if o and o.get("rx") is not None and now - o["rx"] <= self.age_max and not o.get("contact"):
                self.lost.add(fid)
            if not o or not o.get("contact") or o.get("rx") is None or now - o["rx"] > self.age_max:
                continue
            x[k * NCH:(k + 1) * NCH] = (o["sx"], o["sy"], o["tw"])
            n_ok += 1
            if o.get("slip"):
                slips.append(fid)
        return x, ~np.isnan(x), n_ok, slips

    # ------------------------------------------------------------------ calibration
    def start_cal(self, now, cams):
        self.calib = dict(phase="rest", t0=now, cams=dict(cams), rest=[], sq=[], turn=[], last_rx={},
                          slip=False, msg="WRIST CAL 1/4: keep still")
        self.reset_live()
        self.log("[WRIST] calibration 1/4: keep the hand and the object still (4 s)")

    def cancel_cal(self, why):
        if self.calib:
            self.log(f"[WRIST] calibration cancelled: {why}")
        self.calib = None

    def finish_turn(self):
        """Key w during the TURN phase: enough turning, compute now."""
        if self.calib and self.calib["phase"] in ("turn", "other"):
            self.calib["force_end"] = True

    def cal_phase(self):
        return self.calib["phase"] if self.calib else ""

    def _cal_step(self, now, x, ok, n_ok, slips, obs):
        C = self.calib
        if n_ok < 2:
            self.cancel_cal("fewer than 2 fingers touch")
            return
        new = any(o.get("rx") is not None and o["rx"] != C["last_rx"].get(fid) for fid, o in obs.items())
        for fid, o in obs.items():
            C["last_rx"][fid] = o.get("rx")
        if slips:
            C["slip"] = True
        t = now - C["t0"]
        if C["phase"] == "rest":
            if t >= self.REST_S:
                # baseline + noise from every camera's OWN distinct frames of the rest phase
                b, sig, n_cam = np.full(self.n, np.nan), np.full(self.n, np.nan), 0
                for k, fid in enumerate(self.ids):
                    fr = np.array([v_ for t_, _, v_ in self.fbuf[fid] if t_ >= C["t0"]])
                    tt = np.array([t_ for t_, _, _ in self.fbuf[fid] if t_ >= C["t0"]])
                    if len(fr) >= 5:
                        n_cam += 1
                        b[k * NCH:(k + 1) * NCH] = fr[-max(5, len(fr) // 4):].mean(0)    # the newest quarter
                        for c in range(NCH):
                            # white noise (frame to frame) + the gel's own WANDER (random walk, measured from
                            # the change over 1 s): what the zero follower (time constant drift_s) leaves over
                            # is sigma^2 = white^2 + D * drift_s / 2. A real gel wanders 5-10 x its frame noise
                            # within a second - a threshold from the frame noise alone fires all the time.
                            d1 = np.diff(fr[:, c])
                            s_w = 1.4826 * float(np.median(np.abs(d1))) / math.sqrt(2.0) if len(d1) >= 3 else 0.0
                            j = np.searchsorted(tt, tt - 1.0)
                            m = (tt - tt[0]) >= 1.0
                            D = 0.0
                            if m.sum() >= 4:
                                dl = fr[m, c] - fr[j[m], c]
                                lag = float(np.median(tt[m] - tt[j[m]]))
                                D = max(0.0, (1.4826 ** 2) * float(np.median(dl ** 2)) - 2.0 * s_w ** 2) / max(lag, 0.3)
                            C.setdefault("s_w", {})[k * NCH + c] = s_w
                            C.setdefault("D", {})[k * NCH + c] = D
                            sig[k * NCH + c] = s_w
                if n_cam < 2:
                    self.cancel_cal("not enough sensor frames while still (camera stalled?)")
                    return
                # the wander is pooled over the pads (4 s give only a few independent 1 s steps per
                # channel - one channel alone would be far too noisy an estimate): one D for the shear
                # channels, one for the twist
                Dv = np.array([C.get("D", {}).get(i, np.nan) for i in range(self.n)])
                for cs_ in ((0, 1), (2,)):
                    idx = [k_ * NCH + c_ for k_ in range(len(self.ids)) for c_ in cs_]
                    Dp = float(np.nanmean(Dv[idx])) if np.isfinite(Dv[idx]).any() else 0.0
                    for i in idx:
                        if np.isfinite(sig[i]):
                            sig[i] = math.sqrt(sig[i] ** 2 + Dp * max(self.drift_s, 0.5) / 2.0)
                for c in range(NCH):                       # floors: never trust a too-quiet channel
                    cs = sig[c::NCH]
                    med = np.nanmedian(cs) if np.isfinite(cs).any() else 1e-3
                    floor = max(0.6 * med, 0.02 if c < 2 else 0.002)
                    cs = np.where(np.isfinite(cs), np.maximum(cs, floor), floor)
                    sig[c::NCH] = cs
                sw = np.array([C.get("s_w", {}).get(i, np.nan) for i in range(self.n)])
                C["wander"] = float(np.nanmedian(sig[0::NCH].tolist() + sig[1::NCH].tolist())
                                    / max(1e-6, np.nanmedian(sw[0::NCH].tolist() + sw[1::NCH].tolist())))
                self.log(f"[WRIST] noise while still: frame {np.nanmedian(sw[0::NCH].tolist() + sw[1::NCH].tolist()):.3f} px, "
                         f"with the gel's wander {np.nanmedian(sig[0::NCH].tolist() + sig[1::NCH].tolist()):.3f} px "
                         f"(x{C['wander']:.1f}) - the thresholds use the second number")
                C.update(b=b, sigma=sig, phase="squeeze", t1=now,
                         msg="WRIST CAL 2/4: the hand squeezes a little - keep still")
                self.log("[WRIST] calibration 2/4: the hand squeezes a little and relaxes - keep still (3 s)")
            return
        z = (x - C["b"]) / C["sigma"]
        if C["phase"] == "squeeze":
            if new:
                C["sq"].append(z)
            for k, fid in enumerate(self.ids):           # each camera's own frames with its own depth
                o = obs.get(fid)
                if o and ok[k * NCH] and o.get("d") is not None:
                    L = C.setdefault("sqf", {}).setdefault(fid, [])
                    if not L or L[-1][0] != o["rx"]:
                        L.append((o["rx"], z[k * NCH:(k + 1) * NCH].copy(), float(o["d"])))
            if now - C["t1"] >= self.SQ_S:
                C.update(phase="turn", t2=now, msg="WRIST CAL 3/4: TURN one way, back, the other way, back (w = done)")
                self.log("[WRIST] calibration 3/4: now TURN the wrist (or the object) slowly ~5 deg one way, back, "
                         "the other way, back (<= 12 s, press w when done). The first direction = +")
            return
        if C["phase"] == "turn":
            if new:
                C["turn"].append(z)
            if C.get("force_end") or now - C["t2"] >= self.TURN_MAX_S:
                C["force_end"] = False
                if self._cal_turn(C):
                    C.update(phase="other", t3=now, oth=[],
                             msg="WRIST CAL 4/4 (optional): PULL / PUSH / lift the object, NO turning (w = done/skip)")
                    self.log("[WRIST] calibration 4/4 (optional, <= 10 s, w = done / skip): pull, push, lift the object "
                             "(the loads the task also puts on it, e.g. pulling a door) WITHOUT turning - these "
                             "patterns are then never read as torsion")
                else:
                    self.calib = None
            return
        if C["phase"] == "other":
            if new:
                C["oth"].append(z)
            if C.get("force_end") or now - C["t3"] >= self.OTHER_MAX_S:
                self._cal_finish(C)

    def _cal_turn(self, C):
        """Torsion pattern u from the TURN phase (squeeze pattern v projected out)."""
        S = np.array(C["sq"]) if C["sq"] else np.zeros((0, self.n))
        Tm = np.array(C["turn"]) if C["turn"] else np.zeros((0, self.n))
        C["keep"] = int((~np.isnan(Tm).any(axis=0)).sum()) if len(Tm) else 0
        # channels missing in a frame (stalled camera) count as 0 = "no information"
        Tm = np.where(np.isnan(Tm), 0.0, Tm)
        S = np.where(np.isnan(S), 0.0, S)
        if len(Tm) < 15:
            self.log("[WRIST] calibration FAILED: too few sensor frames while turning")
            return False
        v = None
        if len(S) >= 5:
            Ws, Vs = np.linalg.eigh(S.T @ S / len(S))
            if Ws[-1] > 4.0:                            # a real squeeze response (> 2 sigma rms)
                # refine: squeezed frames minus relaxed frames (the transitions, where the cameras see
                # the step at different times, are left out) -> the pattern without camera skew
                q = S @ Vs[:, -1]
                lo, hi = np.percentile(q, 30), np.percentile(q, 70)
                if (q >= hi).sum() >= 2 and (q <= lo).sum() >= 2:
                    d_ = S[q >= hi].mean(0) - S[q <= lo].mean(0)
                    v = d_ / max(np.linalg.norm(d_), 1e-9)
                else:
                    v = Vs[:, -1]
        # better when the pads report their depth: per pad, its channels regressed on its OWN depth of the
        # same frame (perfectly synchronous, uses every frame incl. the ramps) x its depth step
        vd, n_fit = np.zeros(self.n), 0
        for k, fid in enumerate(self.ids):
            L = (C.get("sqf") or {}).get(fid) or []
            if len(L) < 6:
                continue
            Zk = np.array([q_[1] for q_ in L])
            dk = np.array([q_[2] for q_ in L])
            step = np.percentile(dk, 90) - np.percentile(dk, 10)
            dc = dk - dk.mean()
            if step < 0.01 or dc @ dc < 1e-9:
                continue
            vd[k * NCH:(k + 1) * NCH] = (dc @ (Zk - Zk.mean(0))) / (dc @ dc) * step
            n_fit += 1
        C["sq_method"] = "pattern"
        if n_fit >= 2 and np.linalg.norm(vd) > 3.0 and (v is None or abs(float(vd @ v)) / np.linalg.norm(vd) > 0.5):
            v = vd / np.linalg.norm(vd)
            C["sq_method"] = "depth regression"
        # the RAW turning pattern (the grip is frozen while turning, so no squeeze is in it): projecting
        # the squeeze out here would bend u and scale the torsion wrongly; the live joint least squares
        # with u AND v separates the two instead
        X = Tm
        W, V = np.linalg.eigh(X.T @ X / len(X))
        u = V[:, -1]
        p = X @ u
        # + = the direction turned first: the first frame clearly away from rest
        big = np.nonzero(np.abs(p) > max(4.0, 0.3 * np.abs(p).max()))[0]
        if len(big) and p[big[0]] < 0:
            u, p = -u, -p
        C["quality"] = float(W[-1] / max(W.sum(), 1e-9))   # share of the turning motion in the pattern
        C["snr"] = float(np.percentile(np.abs(p), 95))      # in noise sigmas
        C["both"] = bool(p.max() > 4.0 and p.min() < -4.0)
        if v is not None and abs(float(u @ v)) > 0.9:
            self.log(f"[WRIST] the turn looks like a squeeze ({abs(float(u @ v)):.2f}) - squeeze pattern not used")
            v = None
        C["u"], C["v"] = u, v
        if C["snr"] < 6.0:
            self.log(f"[WRIST] calibration FAILED: the turn hardly moved the gels (SNR {C['snr']:.1f} < 6). Turn "
                     f"further / the object must resist (mounted, or hold it) - a freely following object gives "
                     f"no torque")
            return False
        self.log(f"[WRIST] torsion pattern learned (quality {C['quality']:.2f}, SNR {C['snr']:.0f})")
        return True

    def _cal_finish(self, C):
        """Other-load patterns P (pull / push / lift) from the optional phase 4, then save."""
        self.calib = None
        u, v = C["u"], C["v"]
        P = []
        O = np.array(C.get("oth") or [])
        if len(O) >= 10:
            O = np.where(np.isnan(O), 0.0, O)
            # the RAW load directions (not orthogonalised against u): the joint least squares then gives
            # a pure pull zero torsion even where its pattern overlaps the torsion pattern
            if v is not None:
                O = O - np.outer(O @ v, v)                  # squeeze is modelled already
            Wo, Vo = np.linalg.eigh(O.T @ O / len(O))
            for j in range(1, 4):
                q = Vo[:, -j]
                if Wo[-j] <= 9.0:                           # <= 3 sigma rms: no real load pattern
                    break
                if abs(q @ u) > 0.9:                        # looks like the torsion itself: the user turned
                    self.log(f"[WRIST] an other-load pattern looks like torsion ({abs(q @ u):.2f}) - not used "
                             f"(pull without turning)")
                    continue
                P.append(q)
        sig = C["sigma"]
        c_px = float(np.mean([np.linalg.norm((sig * u)[k * NCH:k * NCH + 2]) for k in range(len(self.ids))]))
        self.cal = dict(ids=self.ids, cams=C["cams"], u=u, v=v, P=[q.tolist() for q in P], sigma=sig, c_px=c_px,
                        quality=C["quality"], snr=C["snr"], both=C["both"], object=self.obj,
                        date=time.strftime("%Y-%m-%d %H:%M"), squeeze=v is not None, slip_during=bool(C["slip"]),
                        live_channels=C.get("keep", 0))
        self.save()
        b = C["b"].copy()                               # zero = the rest pose of this calibration
        self.zero_ok = ~np.isnan(b)
        self.b = np.where(self.zero_ok, b, 0.0)
        self.t_seen = {}
        self.reset_live()
        self.why = ""
        warn = []
        if C["quality"] < 0.5:
            warn.append(f"pattern unclear (quality {C['quality']:.2f} < 0.5: turn around ONE axis, steadily)")
        if not C["both"]:
            warn.append("only one direction seen (the other is assumed symmetric)")
        if C["slip"]:
            warn.append("a pad slipped during the calibration (turn / pull less)")
        if v is None:
            warn.append("no squeeze response learned (grip changes may show as a little torsion)")
        if P:
            A = np.stack([u] + ([v] if v is not None else []) + P, 1)
            infl = float(np.sqrt(np.linalg.inv(A.T @ A)[0, 0]))   # noise factor of the torsion estimate
            self.cal["noise_factor"] = infl
            if infl > 1.6:
                warn.append(f"pull and torsion look partly alike: torsion noise x{infl:.1f} (pull more "
                            f"differently / in fewer directions)")
            self.save()
        per = " ".join(f"ID{fid} {np.linalg.norm(u[k * NCH:(k + 1) * NCH]) ** 2 * 100:.0f}%"
                       for k, fid in enumerate(self.ids))
        self.log(f"[WRIST] calibrated: quality {C['quality']:.2f}, SNR {C['snr']:.0f}, 1 unit = {c_px:.3f} px "
                 f"tangential shear per pad; share per finger {per}; {len(P)} other-load pattern(s) excluded"
                 + (" | WARNING " + "; ".join(warn) if warn else "")
                 + f" -> saved {os.path.basename(self.cal_path)} [{self.obj}]")

    # ------------------------------------------------------------------ live
    def zero(self, now, obs, why=""):
        """The current shear of every touching pad = zero torque (call it once the grasp is settled)."""
        if self.cal is None:
            return False
        x, ok, n_ok, _ = self._vector(obs, now)
        if n_ok < 2:
            return False
        # zero = per camera the mean of its own frames of the last 0.5 s (at least 3 frames: a slow
        # camera looks further back, <= 1 s): one noisy frame would leave a bias of ~1 sigma
        for k, fid in enumerate(self.ids):
            fr = [v_ for t_, _, v_ in self.fbuf[fid] if now - t_ <= 1.0]
            n_ = max(3, sum(1 for t_, _, _ in self.fbuf[fid] if now - t_ <= 0.5))
            if fr and ok[k * NCH]:
                x[k * NCH:(k + 1) * NCH] = np.mean(fr[-n_:], 0)
        b = self.b.copy() if self.b is not None else np.where(ok, x, 0.0)
        b[ok] = x[ok]
        self.b = b
        self.zero_ok = ok.copy()
        self.t_seen = {}
        self.reset_live()
        self.why = ""
        self.log(f"[WRIST] zero{(' (' + why + ')') if why else ''}: torsion measured from here on")
        return True

    def unzero(self, why):
        if self.b is not None:
            self.log(f"[WRIST] {why} -> torsion off until the next zero")
        self.b = None
        self.reset_live()

    def update(self, now, obs):
        """Call every control tick. Returns the state dict."""
        x, ok, n_ok, slips = self._vector(obs, now)
        for k, fid in enumerate(self.ids):              # each camera's own frames, no sample-and-hold copies
            fb = self.fbuf[fid]
            if ok[k * NCH] and (not fb or fb[-1][1] != obs[fid]["rx"]):
                fb.append((now, obs[fid]["rx"], x[k * NCH:(k + 1) * NCH].copy()))
            while fb and now - fb[0][0] > 6.0:
                fb.popleft()
        if self.calib:
            self._cal_step(now, x, ok, n_ok, slips, obs)
            return self.state()
        dt = 0.0 if self.t_prev is None else min(0.2, max(0.0, now - self.t_prev))
        self.t_prev = now
        self.n_cams = n_ok
        if self.cal is None or self.b is None:
            self.valid = False
            return self.state()
        # a pad that was away > 0.5 s lost its zero (its contact changed); a pad that was not zeroed
        # (stalled camera / touched later) gets its zero the first time it is fresh while no torsion is felt
        quiet = None
        for k, fid in enumerate(self.ids):
            sl = slice(k * NCH, (k + 1) * NCH)
            if fid not in self.lost:
                self.t_seen[fid] = now
            elif now - self.t_seen.get(fid, now) > 0.5:
                if self.zero_ok[sl].any():
                    self.fbuf[fid].clear()                   # its old frames belong to the old contact
                self.zero_ok[sl] = False                     # contact really lost: its zero is gone
            if ok[sl].all() and not self.zero_ok[sl].all():
                # back in contact: wait until it has settled (>= 3 own frames over >= 0.4 s, the first
                # 0.15 s of landing left out), then its mean = its zero - but only while the pads that
                # still have their zero feel no load at all
                fr = [(t_, v_) for t_, _, v_ in self.fbuf[fid]]
                use = [v_ for t_, v_ in fr if t_ - fr[0][0] >= 0.15][-6:] if fr else []
                if len(use) < 3 or now - fr[0][0] < 0.4:
                    continue
                if quiet is None:                            # no load on the zeroed pads at all?
                    m = ok & self.zero_ok
                    zq = (x[m] - self.b[m]) / self.cal["sigma"][m]
                    quiet = m.sum() >= NCH and float(zq @ zq) < 3.0 * m.sum() and not self.active
                if quiet:
                    self.b[sl], self.zero_ok[sl] = np.mean(use, 0), True
        okz = ok & self.zero_ok
        u, v, sig = self.cal["u"], self.cal["v"], self.cal["sigma"]
        ua = u[okz]
        if n_ok < 2 or ua @ ua < 0.35:
            self.valid, self.why = False, (f"{n_ok} pad(s) in contact" if n_ok < 2 else "the pads that carry the "
                                           "torsion pattern are not fresh (camera stall?)")
            self.active = False
            return self.state()
        z = np.where(okz, (np.where(ok, x, 0.0) - self.b) / sig, 0.0)
        za = z[okz]
        # torsion amplitude by least squares together with the nuisance patterns (squeeze, pulls)
        nus = ([v] if v is not None else []) + [np.asarray(q) for q in self.cal.get("P", [])]
        nus = [q[okz] for q in nus if q[okz] @ q[okz] > 0.05]
        nus = nus[:max(0, len(za) - 3)]                  # keep the fit well determined
        if nus:
            A = np.stack([ua] + nus, 1)
            coef = np.linalg.lstsq(A, za, rcond=None)[0]
            a = float(coef[0])
            rest = za - A[:, 1:] @ coef[1:]
            try:
                sd_a = float(np.sqrt(np.linalg.inv(A.T @ A)[0, 0]))   # noise std of a (z units)
            except np.linalg.LinAlgError:
                sd_a = 1.0 / math.sqrt(ua @ ua)
        else:
            a = float(ua @ za / (ua @ ua))
            rest = za
            sd_a = 1.0 / math.sqrt(ua @ ua)
        self.valid, self.why = True, ""
        k = min(1.0, dt / self.tau_s) if dt > 0 else 1.0
        a_prev = self.a_f
        self.a, self.a_f = a, self.a_f + k * (a - self.a_f)
        if dt > 0:
            kr = min(1.0, dt / self.rate_s)
            self.rate += kr * ((self.a_f - a_prev) / dt - self.rate)
        # noise of the estimate: the theoretical one, or what it really fluctuates by while nothing is
        # felt (running, 10 s) if that is larger - the threshold tunes itself to this gel and this grasp
        sd_eff = max(sd_a, self.sd_run)
        self.snr = abs(self.a_f) / sd_eff                  # how many noise sigmas the torsion is
        e_rest = float(rest @ rest)
        sig_part = self.a_f ** 2 * float(ua @ ua)
        self.coh = (min(1.0, sig_part / e_rest) if e_rest > 2.0 * len(za) else None)
        # on: above --on for 0.15 s without a break (one noisy camera frame is no torsion); off: below --off
        # after a release the same direction is not taken as a new torsion for 1.5 s: what is left then is
        # the gel's wander during the hold (the zero was frozen) - the follower takes it out first
        blocked = now - self.t_rel < 1.5 and self.a_f * self.ep_sign > 0
        self.n_on = (self.n_on + dt) if (self.snr > self.on and not blocked) else 0.0
        was = self.active
        if self.active:
            # off: below --off, or back under 40 % of the peak of THIS torsion = it was released (the
            # rest is wander / leak collected while the zero was frozen, not a torsion)
            self.a_peak = max(self.a_peak, abs(self.a_f))
            self.active = self.snr > self.off and abs(self.a_f) > 0.4 * self.a_peak
            if not self.active:
                self.n_on = 0.0                              # a new torsion needs its own 0.15 s
                self.t_rel = now
                self.a_peak_last = self.a_peak
        else:
            self.active = self.n_on >= 0.15
            self.a_peak = abs(self.a_f) if self.active else 0.0
            if self.active:
                self.ep_sign = 1.0 if self.a_f > 0 else -1.0
        if self.active and not was and self.bhist:
            # ONSET: the follower has already eaten the first part of this torsion - go back to the zero
            # of REWIND_S ago (before it began) and measure from there
            old = [bb for tb, bb in self.bhist if now - tb >= self.REWIND_S]
            b_old = old[-1] if old else self.bhist[0][1]
            m = okz & np.isfinite(b_old)
            self.b[m] = b_old[m]
        if self.active or self.n_on > 0 or now - self.t_rel < 1.5:
            self.t_evt = now                                 # an onset / torsion / release is going on
        if dt > 0 and now - self.t_evt > 1.5:
            # only the quiet stretches (>= 1.5 s away from any torsion) teach the noise level
            kk = min(1.0, dt / 10.0)
            self.sd_run = math.sqrt((1 - kk) * self.sd_run ** 2 + kk * min(self.a_f ** 2, (3.0 * sd_a) ** 2))
        if not self.bhist or now - self.bhist[-1][0] >= 0.1:
            self.bhist.append((now, np.where(self.zero_ok, self.b, np.nan)))
            while self.bhist and now - self.bhist[0][0] > 3.0:
                self.bhist.popleft()
        # THE ZERO FOLLOWS THE GEL: while no torsion is felt (and none is building up)
        # the zero follows the reading fast (drift_s) - the gel's own wander never builds up. While a
        # torsion or another load is felt it still leaks very slowly (hold_s): a held torsion is kept
        # for a minute or two, but a wrong zero cannot stay for ever.
        calm = (not self.active) and not slips and self.n_on == 0.0
        tc = self.drift_s if calm else self.hold_s
        if tc > 0 and dt > 0:
            inc = min(1.0, dt / tc) * (x[okz] - self.b[okz])
            self.b[okz] += inc
            if self.active:
                self.leak[okz] += inc                        # what the slow leak has taken from this torsion
        if was and not self.active:
            # RELEASE: give the slow leak of this torsion back to the zero - unless the leak itself is
            # what ended it (then it was a wrong zero that the leak has corrected: keep it)
            a_leak = float(ua @ (self.leak[okz] / sig[okz]) / (ua @ ua))
            if abs(a_leak) < 0.5 * self.a_peak_last:
                self.b[okz] -= self.leak[okz]
                self.a_f += a_leak
                self.a = self.a_f
            self.leak[:] = 0.0
        self.tau_px = self.a_f * self.cal["c_px"]
        self.tau_nmm = self.tau_px * self.nmm if self.nmm else 0.0
        self._slip(now, z, okz, slips)
        return self.state()

    def _slip(self, now, z, okz, slips):
        self.hist.append((now, z, okz))
        while self.hist and now - self.hist[0][0] > 0.35:
            self.hist.popleft()
        self.peak.append((now, abs(self.a_f)))
        while self.peak and now - self.peak[0][0] > 0.5:
            self.peak.popleft()
        u = self.cal["u"]
        t0, z0, ok0 = self.hist[0]
        m = okz & ok0
        dz = (z - z0)[m]
        um = u[m]
        d_a = float(um @ dz / max(um @ um, 1e-9))
        e = float(dz @ dz)
        coh_d = (d_a ** 2 * float(um @ um) / e) if e > 1e-9 else 0.0
        big = abs(d_a) * math.sqrt(um @ um) > 3.0 and coh_d > 0.5
        pk = max(p for _, p in self.peak)
        drop = pk * math.sqrt(um @ um) > 2 * self.on and abs(self.a_f) < 0.6 * pk
        hit = (len(slips) >= 2 and big) or (len(slips) >= 1 and drop and coh_d > 0.5)
        if hit:
            if not self.rot_slip:
                self.n_slip_events += 1
                self.slip_dir = 1 if d_a > 0 else -1
                self.log(f"[WRIST] ROT-SLIP: the object turns in the hand ({self.names[0 if d_a > 0 else 1]}), "
                         f"pads slipping {slips}, torsion {self.tau_px:+.2f} px")
            self.rot_slip, self.t_slip = True, now
        elif self.rot_slip and now - self.t_slip > 0.8:
            self.rot_slip = False

    def state(self):
        C = self.calib
        d = 0 if not (self.valid and self.active) else (1 if self.a_f > 0 else -1)
        return dict(cal=self.cal is not None, cal_phase=C["phase"] if C else "", cal_msg=C["msg"] if C else "",
                    cal_t=(time.monotonic() - C.get("t2", C.get("t1", C["t0"]))) if C else 0.0,
                    zeroed=self.b is not None, valid=self.valid, active=self.active, tau_z=self.a_f,
                    snr=self.snr, tau_px=self.tau_px, tau_nmm=self.tau_nmm, rate=self.rate * (self.cal["c_px"] if self.cal else 0.0),
                    dir=d, dir_name=self.names[0] if d > 0 else self.names[1] if d < 0 else "",
                    coh=self.coh, pure=self.coh is None or self.coh >= 0.85, rot_slip=self.rot_slip, slip_dir=self.slip_dir, n_slip=self.n_slip_events,
                    n_cams=self.n_cams, why=self.why,
                    quality=self.cal.get("quality", 0.0) if self.cal else 0.0)
