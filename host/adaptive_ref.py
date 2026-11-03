#!/usr/bin/env python3
"""
adaptive_ref.py  —  contact-aware adaptive referencing (automatic re-baseline)
================================================================================
Flow/depth are measured against a reference frame. After a press the gel creeps back
for a while (viscoelastic), and without contact the reference slowly goes stale
(temperature, the Pi Zero's auto exposure, cable/mount). This module decides WHEN to
take a new reference.

KEY RULE: a new reference is only taken when the gel is truly STILL, judged from the
IMAGE itself: the current frame is compared with the frame ~1 s earlier. If they differ
by no more than camera noise, nothing has moved for a second -> safe to re-reference.
This does not depend on the flow (a flow deadzone/EMA can hide slow creep), so the
reference is never taken in the middle of the creep that follows a release.

When still:
  after a contact, once the contact is gone                  -> "released"
  after a contact, if the flow fell below leftover_frac of
  its peak (residue the depth still calls "contact")         -> "release leftover"
  no contact: residual flow p90 > resid_px                   -> "residual"
  no contact: brightness moved > bright_pct                  -> "brightness"
  no contact: every period_s seconds                         -> "periodic"
  for follow_s seconds after a release reference: no contact, image still for
  follow_still_s and residual p98 > follow_px                 -> "release follow-up"
    (the gel keeps creeping back slower than the camera noise, so the first reference
    after a release can land in the middle of the creep; the rest of the creep then
    stays on screen as a ~0.2 px residual below resid_px. Follow-up references chase it
    down in small steps until it is gone.)
Never while an object is held (flow near its peak), never while recording.
Optional stuck_s > 0: a "contact" still for that long is re-referenced (OFF by default,
it would also erase an object held perfectly still).

Arguments for run_sensor_fast.py / run_sensor_dual.py:
  --aref=off              disable (only 'r')
  --aref-resid=0.35       residual flow threshold when idle, px (p90 of |flow|)
  --aref-period=20        periodic re-reference while idle, s (0 = off)
  --aref-still=1.0        how long the image must be unchanged before re-referencing, s
  --aref-leftover=0.4     residue fraction of the peak flow treated as "released"
  --aref-stuck=0          stuck-contact rescue, s (0 = off)
  --aref-release-still=0.4  stillness before the first reference after a release, s
                          (> 0.4 = wait the full --aref-still instead)
  --aref-follow=6         follow-up window after a release, s (0 = off)
  --aref-follow-px=0.15   residual that triggers a follow-up reference, px (p98)
"""
import collections
import time

import cv2
import numpy as np


class AdaptiveReference:
    def __init__(self, resid_px=0.35, bright_pct=4.0, period_s=20.0, still_s=1.0,
                 leftover_frac=0.4, leftover_min_peak=1.0, cooldown_s=0.5,
                 stuck_s=0.0, enabled=True, settle_n=6, settle_px=0.02,
                 follow_s=6.0, follow_px=0.15, follow_still_s=0.4, release_still_s=0.4):
        self.enabled = enabled
        self.resid_px, self.bright_pct, self.period_s = resid_px, bright_pct, period_s
        self.still_s, self.cooldown_s, self.stuck_s = still_s, cooldown_s, stuck_s
        self.leftover_frac, self.leftover_min_peak = leftover_frac, leftover_min_peak
        self.settle_n, self.settle_px = settle_n, settle_px       # fallback without images
        self.follow_s, self.follow_px, self.follow_still_s = follow_s, follow_px, follow_still_s
        self._follow_until = -1.0
        # the first reference after a release only needs a SHORT stillness: the follow-up
        # references clean up whatever creep is left, so no need to wait a full still_s
        self.release_still_s = release_still_s
        self.follow_still_s = release_still_s        # one short stillness window for both
        self.follow_every_s = 1.0
        self.release_now = True                      # reference at the release itself
        self._nc_n = 0
        self.n_resets = 0
        self.last_reason = ""
        # image history (persists across references)
        self._hist = collections.deque()             # (t, small grey)
        self._cdiff = collections.deque(maxlen=300)  # consecutive-frame differences
        self._prev_small = None
        self._prev_med = None
        self._settled = 0
        self.still = False
        self._still_n = 0                            # consecutive "still" frames
        self._still_short_n = 0
        self.still_short = False
        self.still_frames = 6
        self.on_reference(None, None)

    # ── call after ANY new reference (also manual 'r') ──
    def on_reference(self, t=None, brightness=None):
        self._t_ref = t                     # None = time of the next update()
        self._bright_ref = brightness
        self._had_contact = False
        self._peak_flow = 0.0
        self._t_contact0 = None

    # ── image stillness: frame now vs frame still_s ago, against the camera noise ──
    def _image_still(self, gray, t):
        """-> (still over still_s, still over follow_still_s)"""
        h, w = gray.shape[:2]
        g = cv2.resize(gray.astype(np.float32), (max(8, w // 2), max(8, h // 2)),
                       interpolation=cv2.INTER_AREA)            # float: no re-quantisation
        if self._prev_small is not None and self._prev_small.shape == g.shape:
            self._cdiff.append(float(np.percentile(np.abs(g - self._prev_small), 95)))
        self._prev_small = g
        self._hist.append((t, g))
        win = max(self.still_s, self.follow_still_s)
        while len(self._hist) > 2 and self._hist[1][0] <= t - win:
            self._hist.popleft()                       # keep the newest frame >= win old
        if len(self._cdiff) < 10:
            return False, False
        nf = float(np.median(self._cdiff))               # camera noise floor (grey levels)
        thr = max(1.25 * nf, nf + 0.5)

        def still_over(window):
            old = None
            for t_i, g_i in self._hist:                  # newest frame at least window old
                if t_i <= t - window:
                    old = g_i
                else:
                    break
            if old is None or old.shape != g.shape:
                return False
            # p95, not the mean: creep is LOCAL (around the contact) and a mean over the
            # whole gel would dilute it below the noise
            return float(np.percentile(np.abs(g - old), 95)) < thr
        return still_over(self.still_s), still_over(self.follow_still_s)

    def update(self, flow, contact, brightness=None, t=None, blocked=False, gray=None):
        """flow HxWx2 (px vs reference), contact bool, brightness 0..1, gray = processed
        grey frame. Returns a reason string when a new reference should be taken now."""
        t = time.monotonic() if t is None else t
        if self._t_ref is None:
            self._t_ref = t
        if self._bright_ref is None and brightness is not None:
            self._bright_ref = brightness
        mag = np.sqrt(flow[::4, ::4, 0] ** 2 + flow[::4, ::4, 1] ** 2)
        med = float(np.median(mag))
        p90 = float(np.percentile(mag, 90))
        p98 = float(np.percentile(mag, 98))     # local residue (creep is around the dent)
        rad = self._radial(flow[::4, ::4], mag, p98)   # >0 outward (press), <0 inward (creep back)

        if gray is not None:
            s_long, s_short = self._image_still(gray, t)
            self._still_n = self._still_n + 1 if s_long else 0
            self._still_short_n = self._still_short_n + 1 if s_short else 0
            self.still = self._still_n >= self.still_frames   # not one lucky noisy frame
            self.still_short = self._still_short_n >= 3
        else:                                            # fallback: flow-based settling
            self._settled = (self._settled + 1 if self._prev_med is not None
                             and abs(med - self._prev_med) < self.settle_px else 0)
            self.still = self.still_short = self._settled >= self.settle_n
        self._prev_med = med

        self._peak_flow = max(self._peak_flow, p90)
        if contact:
            self._had_contact = True
            if self._t_contact0 is None:
                self._t_contact0 = t
        else:
            self._t_contact0 = None
        self._nc_n = 0 if contact else self._nc_n + 1   # frames without contact

        if not self.enabled or blocked:
            return None
        if t - self._t_ref < self.cooldown_s:
            return None
        # RELEASE: contact gone (2 frames) and the flow clearly dropped from its peak ->
        # new reference NOW, so the picture is clean when 'release' is reported. The gel
        # keeps creeping back afterwards; the follow-up below removes that.
        if (self.release_now and self._had_contact and self._nc_n >= 2
                and p90 < self.leftover_frac * max(self._peak_flow, 1e-6)):
            return self._fire(t, brightness, f"released (residual p90 {p90:.2f}px)",
                              start_follow=True)
        if contact or self._had_contact:
            self._follow_until = -1.0                    # a new press ends the follow-up
        elif t < self._follow_until:
            # creep back = INWARD flow: re-reference right away (no stillness needed);
            # outward flow without contact may be a new light press: leave it alone.
            if rad < 0 and p98 > self.follow_px:
                return self._fire(t, brightness, f"release follow-up (creep p98 {p98:.2f}px)",
                                  follow=True)
            if self.still_short and rad <= 0.5 * self.follow_px and (
                    p98 > self.follow_px or t - self._t_ref > self.follow_every_s):
                return self._fire(t, brightness, f"release follow-up (residual p98 {p98:.2f}px)",
                                  follow=True)
        pressed = self._had_contact or self._peak_flow > self.leftover_min_peak
        use_short = self.follow_s > 0 and self.release_still_s < self.still_s
        rel_still = self.still_short if use_short else self.still
        if pressed and not contact and rel_still:
            win = self.release_still_s if use_short else self.still_s
            return self._fire(t, brightness, f"released, gel still {win:.1f}s "
                                             f"(residual p90 {p90:.2f}px)", start_follow=True)
        if not self.still:
            return None

        if pressed:
            if not contact:
                return None
            if (self._peak_flow > self.leftover_min_peak
                    and p90 < self.leftover_frac * self._peak_flow):
                return self._fire(t, brightness, f"release leftover (p90 {p90:.2f}px, "
                                                 f"peak {self._peak_flow:.2f}px)", start_follow=True)
            if self.stuck_s > 0 and self._t_contact0 is not None \
                    and t - self._t_contact0 > self.stuck_s:
                return self._fire(t, brightness, f"stuck contact ({self.stuck_s:.0f}s still)")
            return None                                  # object held: keep the reference

        if contact:
            return None
        if p90 > self.resid_px:
            return self._fire(t, brightness, f"residual flow p90 {p90:.2f}px")
        if (brightness is not None and self._bright_ref
                and abs(brightness - self._bright_ref) / max(self._bright_ref, 1e-3) * 100
                > self.bright_pct):
            return self._fire(t, brightness, "brightness changed "
                              f"{(brightness / self._bright_ref - 1) * 100:+.1f}%")
        if self.period_s > 0 and t - self._t_ref > self.period_s:
            return self._fire(t, brightness, f"periodic ({self.period_s:.0f}s idle)")
        return None

    @staticmethod
    def _radial(f, mag, p98):
        """Mean radial flow (px) around the moving area: >0 outward, <0 inward."""
        sel = mag > max(0.5 * p98, 0.05)
        if sel.sum() < 4:
            return 0.0
        ys, xs = np.nonzero(sel)
        wgt = mag[sel]
        cy, cx = float((ys * wgt).sum() / wgt.sum()), float((xs * wgt).sum() / wgt.sum())
        ry, rx = ys - cy, xs - cx
        rn = np.sqrt(rx * rx + ry * ry) + 1e-6
        return float(((f[..., 0][sel] * rx + f[..., 1][sel] * ry) / rn).mean())

    def _fire(self, t, brightness, reason, start_follow=False, follow=False):
        self.n_resets += 1
        self.last_reason = reason
        self.on_reference(t, brightness)
        if start_follow and self.follow_s > 0:
            self._follow_until = t + self.follow_s       # keep watching the creep
        return reason

    def status(self):
        return f"AREF {'on' if self.enabled else 'off'} ({self.n_resets})"


def parse_args(argv):
    o = dict(enabled=True, resid_px=0.35, period_s=20.0, stuck_s=0.0, still_s=1.0,
             leftover_frac=0.4, follow_s=6.0, follow_px=0.15)
    for a in argv:
        al = a.lower()
        if al.startswith("--aref="):
            o["enabled"] = al.split("=", 1)[1] not in ("off", "0", "no", "none")
        elif al.startswith("--aref-resid="):
            o["resid_px"] = float(al.split("=", 1)[1])
        elif al.startswith("--aref-period="):
            o["period_s"] = float(al.split("=", 1)[1])
        elif al.startswith("--aref-stuck="):
            o["stuck_s"] = float(al.split("=", 1)[1])
        elif al.startswith("--aref-still="):
            o["still_s"] = float(al.split("=", 1)[1])
        elif al.startswith("--aref-release-still="):
            o["release_still_s"] = float(al.split("=", 1)[1])
        elif al.startswith("--aref-follow-px="):
            o["follow_px"] = float(al.split("=", 1)[1])
        elif al.startswith("--aref-follow="):
            o["follow_s"] = float(al.split("=", 1)[1])
        elif al.startswith("--aref-leftover="):
            o["leftover_frac"] = float(al.split("=", 1)[1])
    return o


def make(o):
    return AdaptiveReference(resid_px=o["resid_px"], period_s=o["period_s"],
                             stuck_s=o["stuck_s"], still_s=o["still_s"],
                             leftover_frac=o["leftover_frac"], enabled=o["enabled"],
                             follow_s=o.get("follow_s", 6.0), follow_px=o.get("follow_px", 0.15),
                             release_still_s=o.get("release_still_s", 0.4))
