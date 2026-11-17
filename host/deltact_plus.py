"""
deltact_plus.py  (FINAL - bersih & konsisten)
=============================================
Tactile sensing engine (Du et al. 2021, IEEE RA-L) untuk IMX219-200 + Jetson.

Pipeline: frame -> undistort ROI (1 remap pra-hitung) -> DIS optical flow
          -> deadzone -> mask -> Gaussian density -> guided filter -> depth bersih

Fitur: fold guard undistort (29 Sep 2026), single/adaptive referencing, variational refinement (flow halus),
denoise + temporal averaging (anti-noise), asymmetric EMA (hilang cepat saat
dilepas), mask kotak/ellips, kontrol ROI live.

30 Sep 2026: kalibrasi distorsi checkerboard (cx, cy, roi_undist). Bila cx, cy dan
roi_undist diisi, undistort dihitung full-frame dengan newK = K (skala seragam) lalu
dipotong ke roi_undist. Profil full (820x616): focal 356.5,
dist (-0.857, 1.1134, 0, 0, 0), cx 419.7, cy 307.3, roi_undist (270, 123, 271, 361).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
import json
import numpy as np
import cv2


# ====================== KONFIGURASI ======================
@dataclass
class SensorConfig:
    # undistortion (hasil tuning user)
    focal: float = 1206.0
    dist: Tuple[float, float, float, float, float] = (-0.43, -0.40, 0.0, 0.0, 0.0)
    roi_thresh: int = 70
    roi_min_area: int = 5000
    # guard lipatan undistort: piksel output di mana r_d(r_u) tidak lagi monoton
    # (1 + 3 k1 r^2 + 5 k2 r^4 <= fold_margin) berisi PANTULAN, bukan citra asli
    # -> di-nol-kan (citra) dan dikeluarkan dari flow/depth (mask).
    fold_guard: bool = True
    # ROI tetap (x, y, w, h) di frame mentah; None = deteksi otomatis via roi_thresh.
    # Dipakai saat jendela gel di-mask (profil full): area hitam hasil masking membuat
    # deteksi ambang memilih SELURUH frame sebagai ROI.
    roi_fixed: Optional[Tuple[int, int, int, int]] = None
    fold_margin: float = 0.05
    # Kalibrasi checkerboard 30 Sep 2026: principal point di frame MENTAH (px).
    # None = pusat ROI (perilaku lama).
    cx: Optional[float] = None
    cy: Optional[float] = None
    # ROI (x, y, w, h) di citra TER-UNDISTORT full-frame (newK = K). Bila diisi
    # bersama cx/cy, dipakai menggantikan roi_fixed/roi_thresh.
    roi_undist: Optional[Tuple[int, int, int, int]] = None

    # resolusi kerja (3:4 = 18x24mm)
    work_w: int = 240
    work_h: int = 320
    edge_crop: int = 4

    # mask area valid
    valid_margin: int = 3           # 0 = ROI penuh tanpa potong
    valid_ellipse: bool = False     # False = KOTAK, True = ellips

    # tracking
    tracking_mode: str = "single"   # "single" | "adaptive"
    flow_scale: int = 1
    dis_preset: str = "ultrafast"
    dis_finest_scale: int = 1
    dis_grad_iters: int = 16
    dis_var_iters: int = 5          # variational refinement -> flow HALUS (paper [31])
    dis_use_mean_norm: bool = True
    dis_use_spatial: bool = True

    # penghalusan input & temporal
    denoise_ksize: int = 5
    temporal_avg: int = 2
    flow_deadzone: float = 0.5      # flow < ini (px) -> 0 (buang noise)
    flow_ema: float = 0.3           # smoothing saat MENEKAN (flow naik)
    flow_ema_release: float = 0.0   # smoothing saat MELEPAS (flow turun). 0 = hilang instan

    # auto-rebaseline: saat TANPA kontak, baseline menyesuaikan -> flow balik ke 0 sendiri
    auto_rebaseline: bool = True
    rebaseline_flow_thresh: float = 1.2  # dianggap "diam" bila mean|flow| < ini (px)
    rebaseline_hold: int = 8             # snap baseline stlh diam N frame (0=off)

    # adaptive only
    enable_loop_closure: bool = False
    loop_check_interval: int = 15
    add_threshold: float = 12.0
    loop_threshold: float = 6.0
    err_percentile: float = 99.0
    max_stack: int = 12

    # depth
    enable_depth: bool = True
    sigma: float = 2.5
    density_subsample: int = 1
    guided_radius: int = 6
    guided_eps: float = 1e-3
    depth_slope: float = 1.0
    depth_intercept: float = 0.0
    depth_min: float = 0.03         # depth < ini -> 0 (region bersih)
    depth_open: int = 3             # morphological open (buang bintik). 0=off

    # Fisher-information / trackability (lihat draft T-RO Sec. IV-VII)
    # noise_sigma WAJIB diisi dari kalibrasi temporal sensor sendiri
    # (statistik no-contact, lihat Sec. IX-A/E), BUKAN nilai provisional
    # paper (8/255 di Tabel I hanya placeholder sebelum kalibrasi).
    noise_sigma: float = 8.0 / 255.0   # std noise sensor, satuan intensitas [0,1]
    fisher_window: int = 8             # wf: jendela analisis flow, FIXED (Sec. VII)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        import dataclasses
        with open(path) as f:
            d = json.load(f)
        valid = {fld.name for fld in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in valid}
        if "dist" in d:
            d["dist"] = tuple(d["dist"])
        if d.get("roi_fixed") is not None:
            d["roi_fixed"] = tuple(d["roi_fixed"])
        if d.get("roi_undist") is not None:
            d["roi_undist"] = tuple(d["roi_undist"])
        return cls(**d)

    @classmethod
    def fast_profile(cls):
        return cls(work_w=192, work_h=256, dis_preset="fast", dis_finest_scale=1,
                   dis_grad_iters=12, dis_var_iters=3, density_subsample=2,
                   guided_radius=4, temporal_avg=2)

    @classmethod
    def quality_profile(cls):
        return cls(work_w=300, work_h=400, tracking_mode="adaptive",
                   enable_loop_closure=True, dis_preset="medium",
                   dis_finest_scale=0, dis_grad_iters=20, dis_var_iters=8,
                   density_subsample=1, guided_radius=8, temporal_avg=3)


# ====================== MESHGRID CACHE ======================
_GRID_CACHE = {}


def _grid(h, w):
    g = _GRID_CACHE.get((h, w))
    if g is None:
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        g = (gx, gy)
        _GRID_CACHE[(h, w)] = g
    return g


# ====================== UNDISTORT + ROI ======================
class RoiUndistorter:
    def __init__(self, cfg, full_w, full_h):
        self.cfg = cfg
        self.full_w, self.full_h = full_w, full_h
        self.roi = None
        self.map_x = self.map_y = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.fold_mask = None      # (work_h, work_w) bool, True = valid; None = semua valid
        self.fold_frac = 0.0       # fraksi output di zona lipatan

    def detect_roi(self, frame):
        if self.cfg.roi_fixed is not None:
            return tuple(int(v) for v in self.cfg.roi_fixed)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, self.cfg.roi_thresh, 255, cv2.THRESH_BINARY_INV)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._kernel)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cx, cy = self.full_w // 2, self.full_h // 2
        best, ba = None, 0
        for c in cnts:
            a = cv2.contourArea(c)
            if a < self.cfg.roi_min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if not (bx < cx < bx + bw and by < cy < by + bh):
                continue
            if a > ba:
                ba, best = a, (bx, by, bw, bh)
        return best

    def _build_maps_undist(self):
        """Undistort full-frame (newK = K, skala seragam) lalu potong ke cfg.roi_undist."""
        cfg = self.cfg
        K = np.array([[cfg.focal, 0, cfg.cx], [0, cfg.focal, cfg.cy], [0, 0, 1.0]], np.float64)
        D = np.array(cfg.dist, np.float64)
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (self.full_w, self.full_h),
                                             cv2.CV_32FC1)
        x, y, w, h = (int(v) for v in cfg.roi_undist)
        x, y = max(0, x), max(0, y)
        w, h = min(w, self.full_w - x), min(h, self.full_h - y)
        m1, m2 = m1[y:y + h, x:x + w], m2[y:y + h, x:x + w]
        self.fold_mask, self.fold_frac = None, 0.0
        if cfg.fold_guard:
            k1 = float(D[0]); k2 = float(D[1]) if len(D) > 1 else 0.0
            ys, xs = np.mgrid[y:y + h, x:x + w].astype(np.float32)
            ru2 = ((xs - cfg.cx) / cfg.focal) ** 2 + ((ys - cfg.cy) / cfg.focal) ** 2
            fold = (1.0 + 3.0 * k1 * ru2 + 5.0 * k2 * ru2 ** 2) > cfg.fold_margin
            if not fold.all():
                self.fold_mask = cv2.resize(fold.astype(np.uint8), (cfg.work_w, cfg.work_h),
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
                self.fold_frac = float(1.0 - self.fold_mask.mean())
                print(f"[deltact_plus] PERINGATAN fold guard: {100 * self.fold_frac:.1f}% ROI "
                      f"di zona lipatan (k1={k1}, k2={k2}).")
        self.map_x = cv2.resize(m1, (cfg.work_w, cfg.work_h)).astype(np.float32)
        self.map_y = cv2.resize(m2, (cfg.work_w, cfg.work_h)).astype(np.float32)
        rx, ry = int(np.floor(m1.min())), int(np.floor(m2.min()))
        self.roi = (rx, ry, int(np.ceil(m1.max())) - rx, int(np.ceil(m2.max())) - ry)
        print(f"[deltact_plus] ROI undistort {tuple(cfg.roi_undist)} (pusat {cfg.cx}, {cfg.cy}) "
              f"-> area mentah {self.roi}, fold {100 * self.fold_frac:.1f}%")
        return True

    def build_maps(self, frame):
        if self.cfg.roi_undist is not None and self.cfg.cx is not None \
                and self.cfg.cy is not None:
            return self._build_maps_undist()
        roi = self.detect_roi(frame)
        if roi is None:
            return False
        rx, ry, rw, rh = roi
        rx, ry = max(0, rx), max(0, ry)
        rw, rh = min(rw, self.full_w - rx), min(rh, self.full_h - ry)
        K = np.array([[self.cfg.focal, 0, rw / 2.0],
                      [0, self.cfg.focal, rh / 2.0], [0, 0, 1.0]], np.float64)
        D = np.array(self.cfg.dist, np.float64)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (rw, rh), 0, (rw, rh))
        m1, m2 = cv2.initUndistortRectifyMap(K, D, None, newK, (rw, rh), cv2.CV_32FC1)
        fold = None
        if self.cfg.fold_guard:
            # koordinat output ter-normalisasi (newK) = radius TAK-terdistorsi r_u
            k1 = float(D[0]); k2 = float(D[1]) if len(D) > 1 else 0.0
            ys, xs = np.mgrid[0:rh, 0:rw].astype(np.float32)
            ru2 = ((xs - newK[0, 2]) / newK[0, 0]) ** 2 + ((ys - newK[1, 2]) / newK[1, 1]) ** 2
            fold = (1.0 + 3.0 * k1 * ru2 + 5.0 * k2 * ru2 ** 2) > self.cfg.fold_margin
        c = self.cfg.edge_crop
        if c > 0 and m1.shape[0] > 2 * c and m1.shape[1] > 2 * c:
            m1, m2 = m1[c:-c, c:-c], m2[c:-c, c:-c]
            if fold is not None:
                fold = fold[c:-c, c:-c]
        m1 = cv2.resize(m1, (self.cfg.work_w, self.cfg.work_h))
        m2 = cv2.resize(m2, (self.cfg.work_w, self.cfg.work_h))
        if fold is not None and not fold.all():
            self.fold_mask = cv2.resize(fold.astype(np.uint8), (self.cfg.work_w, self.cfg.work_h),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
            self.fold_frac = float(1.0 - self.fold_mask.mean())
            print(f"[deltact_plus] PERINGATAN fold guard: {100 * self.fold_frac:.1f}% ROI di zona "
                  f"lipatan undistort (k1={k1}, k2={k2}) -> di-nol-kan. Fit ulang distorsi "
                  f"untuk mode kamera ini.")
        else:
            self.fold_mask, self.fold_frac = None, 0.0
        self.map_x = (m1 + rx).astype(np.float32)
        self.map_y = (m2 + ry).astype(np.float32)
        self.roi = (rx, ry, rw, rh)
        return True

    def process(self, frame, redetect=False):
        if self.map_x is None or redetect:
            if not self.build_maps(frame):
                return None
        out = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        if self.fold_mask is not None:
            out[~self.fold_mask] = 0
        return out


# ====================== FLOW UTIL ======================
def make_dis(cfg):
    pm = {"ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
          "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
          "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM}
    d = cv2.DISOpticalFlow_create(pm.get(cfg.dis_preset,
                                  cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST))
    d.setFinestScale(cfg.dis_finest_scale)
    d.setGradientDescentIterations(cfg.dis_grad_iters)
    d.setVariationalRefinementIterations(cfg.dis_var_iters)
    d.setUseMeanNormalization(cfg.dis_use_mean_norm)
    d.setUseSpatialPropagation(cfg.dis_use_spatial)
    return d


def compose_flows(flow_ab, flow_bc):
    h, w = flow_ab.shape[:2]
    gx, gy = _grid(h, w)
    s = cv2.remap(flow_bc, gx + flow_ab[..., 0], gy + flow_ab[..., 1],
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return flow_ab + s


def warp_by_flow(img, flow):
    h, w = flow.shape[:2]
    gx, gy = _grid(h, w)
    return cv2.remap(img, gx + flow[..., 0], gy + flow[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def make_valid_mask(cfg):
    m = cfg.valid_margin
    if m <= 0:
        return None
    h, w = cfg.work_h, cfg.work_w
    mask = np.zeros((h, w), np.float32)
    if cfg.valid_ellipse:
        cv2.ellipse(mask, (w // 2, h // 2),
                    ((w - 2 * m) // 2, (h - 2 * m) // 2), 0, 0, 360, 1.0, -1)
    else:
        mask[m:h - m, m:w - m] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, m / 3.0))


# ====================== TRACKER ======================
@dataclass
class _Ref:
    gray: np.ndarray
    flow_base: np.ndarray


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dis = make_dis(cfg)
        self.baseline = None
        self.stack: List[_Ref] = []
        self._idx = 0
        self._prev_flow = None
        self._rest = 0

    def reset(self, gray):
        self.baseline = gray.copy()
        zero = np.zeros((*gray.shape, 2), np.float32)
        self.stack = [_Ref(gray.copy(), zero)]
        self._idx = 0
        self._prev_flow = None

    def _smooth(self, flow):
        pf = self._prev_flow
        if pf is not None and pf.shape == flow.shape \
                and (self.cfg.flow_ema > 0 or self.cfg.flow_ema_release > 0):
            nm = np.linalg.norm(flow, axis=2)
            pm = np.linalg.norm(pf, axis=2)
            # pixel flow MENGECIL (dilepas) -> ema_release; membesar (ditekan) -> ema
            e = np.where(nm < pm, self.cfg.flow_ema_release,
                         self.cfg.flow_ema).astype(np.float32)[..., None]
            flow = e * pf + (1.0 - e) * flow
        self._prev_flow = flow
        return flow

    def _maybe_rebaseline(self, cur, flow):
        # deteksi "diam": bila flow kecil selama N frame -> snap baseline = frame kini
        # (snap, bukan blend, supaya tidak ada bayangan ganda). Hilangkan drift/sisa.
        if not self.cfg.auto_rebaseline or self.cfg.rebaseline_hold <= 0 \
                or self.baseline is None:
            return
        if float(np.linalg.norm(flow, axis=2).mean()) < self.cfg.rebaseline_flow_thresh:
            self._rest += 1
            if self._rest >= self.cfg.rebaseline_hold:
                self.baseline = cur.copy()      # snap ke kondisi diam terkini
                self._prev_flow = None
                self._rest = 0
        else:
            self._rest = 0                       # ada kontak aktif -> bekukan baseline

    def _calc(self, a, b):
        s = self.cfg.flow_scale
        if s > 1:
            h, w = a.shape
            f = self.dis.calc(cv2.resize(a, (w // s, h // s)),
                              cv2.resize(b, (w // s, h // s)), None)
            return cv2.resize(f, (w, h)) * float(s)
        return self.dis.calc(a, b, None)

    def update(self, cur):
        if self.baseline is None:
            self.reset(cur)
            return np.zeros((*cur.shape, 2), np.float32)

        if self.cfg.tracking_mode == "single":
            flow = self._smooth(self._calc(self.baseline, cur))
            self._maybe_rebaseline(cur, flow)
            return flow

        # adaptive
        self._idx += 1
        top = self.stack[-1]
        ft = self._calc(top.gray, cur)
        out = compose_flows(top.flow_base, ft)
        warped = warp_by_flow(cur, ft)
        err = float(np.percentile(
            np.abs(warped.astype(np.float32) - top.gray.astype(np.float32)),
            self.cfg.err_percentile))
        closed = False
        if (self.cfg.enable_loop_closure and len(self.stack) >= 2
                and self._idx % self.cfg.loop_check_interval == 0):
            prev = self.stack[-2]
            fp = self._calc(prev.gray, cur)
            wp = warp_by_flow(cur, fp)
            ep = float(np.percentile(
                np.abs(wp.astype(np.float32) - prev.gray.astype(np.float32)),
                self.cfg.err_percentile))
            if ep < self.cfg.loop_threshold:
                self.stack.pop()
                out = compose_flows(prev.flow_base, fp)
                closed = True
        if (not closed) and err > self.cfg.add_threshold \
                and len(self.stack) < self.cfg.max_stack:
            self.stack.append(_Ref(cur.copy(), out.copy()))
        return self._smooth(out)

    @property
    def stack_size(self):
        return len(self.stack)


# ====================== DEPTH ======================
_HAS_XIMG = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")


def _gf_np(guide, src, r, eps):
    g = guide.astype(np.float32); p = src.astype(np.float32); k = (r, r)
    mI = cv2.boxFilter(g, -1, k); mp = cv2.boxFilter(p, -1, k)
    cov = cv2.boxFilter(g * p, -1, k) - mI * mp
    var = cv2.boxFilter(g * g, -1, k) - mI * mI
    a = cov / (var + eps); b = mp - a * mI
    return cv2.boxFilter(a, -1, k) * g + cv2.boxFilter(b, -1, k)


def guided_filter(guide, src, r, eps):
    if _HAS_XIMG:
        return cv2.ximgproc.guidedFilter(guide.astype(np.float32),
                                         src.astype(np.float32), r, eps)
    return _gf_np(guide, src, r, eps)


def gaussian_density(flow, sigma, subsample=1):
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h:subsample, 0:w:subsample]
    xs = xs.astype(np.float32); ys = ys.astype(np.float32)
    wx = xs + flow[::subsample, ::subsample, 0]
    wy = ys + flow[::subsample, ::subsample, 1]
    acc = np.zeros(h * w, np.float32)
    x0 = np.floor(wx).astype(np.int32); y0 = np.floor(wy).astype(np.int32)
    fx = wx - x0; fy = wy - y0
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = np.clip(x0 + dx, 0, w - 1); yi = np.clip(y0 + dy, 0, h - 1)
        acc += np.bincount((yi * w + xi).ravel(), weights=wgt.ravel(),
                           minlength=h * w)
    return cv2.GaussianBlur(acc.reshape(h, w), (0, 0), sigma)


class DepthEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.baseline = None

    def set_baseline(self, flow_zero):
        self.baseline = gaussian_density(flow_zero, self.cfg.sigma,
                                         self.cfg.density_subsample)

    def compute(self, flow):
        dens = gaussian_density(flow, self.cfg.sigma, self.cfg.density_subsample)
        if self.baseline is None:
            self.baseline = gaussian_density(np.zeros_like(flow),
                                             self.cfg.sigma, self.cfg.density_subsample)
        feat = np.clip(self.baseline - dens, 0, None)
        feat = guided_filter(feat, feat, self.cfg.guided_radius, self.cfg.guided_eps)
        return np.clip(self.cfg.depth_slope * feat + self.cfg.depth_intercept, 0, None)


# ====================== FISHER INFORMATION / TRACKABILITY ======================
# Implements the chain pattern -> Fisher information -> CRB (T-RO draft, Sec. IV).
#
#   J(x)      = structure tensor, windowed over `win` px (Eq. 2 / Eq. 7)
#   lambda2   = smaller eigenvalue of J = lambda_min = worst-case trackability
#   fisher    = lambda_min / sigma_n^2  (Eq. 2: I(d) = J / sigma^2)
#   crb_flow  = sigma_n^2 / lambda_min  (Eq. 3: CRB lower bound on displacement var, px^2)
#
# IMPORTANT: this bounds DISPLACEMENT variance, not depth error. Eq. (4) of the
# paper requires propagating through the flow->depth Jacobian J_Phi, which is
# NOT computed here. Do not relabel crb_flow as a depth bound.
def fisher_trackability(
    gray: np.ndarray,
    win: int = 8,
    sigma_noise: float = 8.0 / 255.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pixel structure tensor smaller eigenvalue (lambda_min),
    Fisher information, and CRB displacement-variance bound.

    gray: grayscale image, any dtype; internally normalized to [0,1] so that
          sigma_noise (also in [0,1] intensity units) is on the same scale as
          the image gradients. Passing a uint8 [0,255] image with a [0,1]
          sigma_noise silently makes the Fisher numbers meaningless -- this
          function normalizes for you so callers don't have to remember it.
    win:  fixed analysis window wf (px), independent of pattern patch size s
          (Sec. VII) -- pass cfg.fisher_window, don't hardcode separately.
    sigma_noise: noise std in [0,1] intensity units. MUST come from your own
          temporal noise calibration (Sec. IX-A/E), not the paper's
          provisional Table I default, if results are meant to support a
          quantitative claim.

    Returns: (lambda_min, fisher_info, crb_flow_var) -- all same shape as gray.
    """
    g = gray.astype(np.float32)
    if g.max() > 1.5:  # heuristically detect uint8-scale input
        g = g / 255.0

    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    k = (win, win)
    Jxx = cv2.boxFilter(Ix * Ix, -1, k)
    Jyy = cv2.boxFilter(Iy * Iy, -1, k)
    Jxy = cv2.boxFilter(Ix * Iy, -1, k)

    tr = Jxx + Jyy
    det = Jxx * Jyy - Jxy * Jxy
    disc = np.maximum(tr * tr - 4.0 * det, 0.0)
    sqrt_disc = np.sqrt(disc)

    lambda_min = np.maximum((tr - sqrt_disc) / 2.0, 0.0)

    sigma2 = max(sigma_noise, 1e-6) ** 2
    fisher_info = lambda_min / sigma2
    crb_flow_var = sigma2 / np.maximum(lambda_min, 1e-12)

    return lambda_min, fisher_info, crb_flow_var


# ====================== PIPELINE ======================
class TactileSensor:
    def __init__(self, cfg, full_w, full_h):
        self.cfg = cfg
        self.undist = RoiUndistorter(cfg, full_w, full_h)
        self.tracker = Tracker(cfg)
        self.depth = DepthEstimator(cfg)
        self.mask = make_valid_mask(cfg)
        self._init = False
        self._buf = []

    def reset(self):
        self._init = False
        self.tracker.baseline = None
        self.tracker.stack = []
        self.tracker._prev_flow = None
        self.depth.baseline = None

    def redetect_roi(self):
        self.undist.map_x = None

    def rebuild_mask(self):
        self.mask = make_valid_mask(self.cfg)

    def process(self, frame):
        roi = self.undist.process(frame)
        if roi is None:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.cfg.denoise_ksize > 0:
            k = self.cfg.denoise_ksize | 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        if self.cfg.temporal_avg > 1:
            self._buf.append(gray.astype(np.float32))
            if len(self._buf) > self.cfg.temporal_avg:
                self._buf.pop(0)
            gray = (sum(self._buf) / len(self._buf)).astype(np.uint8)

        if not self._init:
            self.tracker.reset(gray)
            self.depth.set_baseline(np.zeros((*gray.shape, 2), np.float32))
            self._init = True
            flow = np.zeros((*gray.shape, 2), np.float32)
        else:
            flow = self.tracker.update(gray)

        if self.cfg.flow_deadzone > 0:
            mag = np.linalg.norm(flow, axis=2)
            flow[mag < self.cfg.flow_deadzone] = 0.0
        if self.mask is not None:
            flow = flow * self.mask[..., None]
        fm = self.undist.fold_mask
        if fm is not None:
            flow = flow * fm[..., None]

        depth = None
        if self.cfg.enable_depth:
            depth = self.depth.compute(flow)
            if self.mask is not None:
                depth = depth * self.mask
            if fm is not None:
                depth = depth * fm
            if self.cfg.depth_min > 0:
                depth[depth < self.cfg.depth_min] = 0.0
            if self.cfg.depth_open > 1:
                kk = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self.cfg.depth_open, self.cfg.depth_open))
                depth = cv2.morphologyEx(depth, cv2.MORPH_OPEN, kk)
        return {"roi": roi, "gray": gray, "flow": flow,
                "depth": depth, "stack": self.tracker.stack_size}


# ====================== VISUALISASI ======================
def flow_to_color(flow):
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def depth_to_color(depth, vmax=None):
    if vmax is None:
        vmax = max(1e-6, float(np.percentile(depth, 99)))
    u8 = (np.clip(depth / vmax, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
