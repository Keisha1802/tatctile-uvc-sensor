#!/usr/bin/env python3
"""
gpu_depth.py  —  depth (Gaussian density -> blur -> guided filter) on the GPU with CuPy
=========================================================================================
Same steps and numbers as deltact_plus.DepthEstimator, computed on an NVIDIA GPU
(Jetson Orin: CUDA 12, `pip install cupy-cuda12x`). Used by the depth process of
run_sensor_fast.py / run_sensor_dual.py with the argument  gpu .

Safety: make() runs the CPU and the GPU version on the same test flow and compares them.
If they differ by more than 1 % of the depth range it refuses (the caller falls back to
the CPU), so the GPU can never silently change the measurement.
"""
import numpy as np

import deltact_plus as dp

try:
    import cupy as cp
    import cupyx
    from cupyx.scipy import ndimage as cnd
    _ERR = None
except Exception as e:                       # no CUDA / no CuPy (Raspberry Pi, PC without GPU)
    cp = cupyx = cnd = None
    _ERR = e


class GpuDepthEstimator:
    """Drop-in for dp.DepthEstimator: .cfg, .baseline, .compute(flow) -> numpy depth."""

    def __init__(self, cfg, box_mode="mirror"):
        if cp is None:
            raise RuntimeError(f"CuPy not available ({_ERR})")
        self.cfg = cfg
        self.baseline = None
        self.box_mode = box_mode
        self._grid = {}
        # deltact_plus uses cv2.ximgproc.guidedFilter (window 2r+1) when available,
        # otherwise its NumPy fallback (window r): use the same window as the CPU path
        self.box_from_radius = (lambda r: 2 * r + 1) if dp._HAS_XIMG else (lambda r: r)

    # ── Gaussian density: bilinear splat of the displaced grid points, then blur ──
    def _density(self, flow_g, sigma, sub):
        h, w = flow_g.shape[:2]
        key = (h, w, sub)
        g = self._grid.get(key)
        if g is None:
            ys, xs = cp.mgrid[0:h:sub, 0:w:sub]
            g = (xs.astype(cp.float32), ys.astype(cp.float32))
            self._grid[key] = g
        xs, ys = g
        wx = xs + flow_g[::sub, ::sub, 0]
        wy = ys + flow_g[::sub, ::sub, 1]
        x0f, y0f = cp.floor(wx), cp.floor(wy)
        fx, fy = wx - x0f, wy - y0f
        x0, y0 = x0f.astype(cp.int32), y0f.astype(cp.int32)
        acc = cp.zeros(h * w, cp.float32)
        for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                            (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi = cp.clip(x0 + dx, 0, w - 1)
            yi = cp.clip(y0 + dy, 0, h - 1)
            cupyx.scatter_add(acc, (yi * w + xi).ravel(), wgt.ravel().astype(cp.float32))
        # cv2.GaussianBlur(ksize=(0,0)) on float: radius 4 sigma, BORDER_REFLECT_101 (= mirror)
        return cnd.gaussian_filter(acc.reshape(h, w), sigma, mode="mirror", truncate=4.0)

    # ── self-guided filter (I = p = feat) ──
    def _guided(self, x, r, eps):
        k = self.box_from_radius(r)

        def box(a):
            return cnd.uniform_filter(a, size=k, mode=self.box_mode)
        m = box(x)
        var = box(x * x) - m * m
        a = var / (var + eps)
        b = m - a * m
        return box(a) * x + box(b)

    def compute(self, flow):
        c = self.cfg
        fg = cp.asarray(flow, dtype=cp.float32)
        dens = self._density(fg, c.sigma, c.density_subsample)
        if self.baseline is None:
            self.baseline = self._density(cp.zeros_like(fg), c.sigma, c.density_subsample)
        feat = cp.clip(self.baseline - dens, 0, None)
        feat = self._guided(feat, c.guided_radius, c.guided_eps)
        out = cp.clip(c.depth_slope * feat + c.depth_intercept, 0, None)
        return cp.asnumpy(out).astype(np.float32)


def _test_flow(cfg):
    h, w = cfg.work_h, cfg.work_w
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    g = np.exp(-((xx - w * 0.45) ** 2 + (yy - h * 0.55) ** 2) / (2 * (0.12 * w) ** 2))
    fl = np.stack([3.0 * g * (xx - w * 0.45) / (0.12 * w),
                   3.0 * g * (yy - h * 0.55) / (0.12 * w)], -1).astype(np.float32)
    return fl


def make(cfg, tol=0.01, verbose=True):
    """-> GpuDepthEstimator checked against the CPU, or raises RuntimeError."""
    fl = _test_flow(cfg)
    ref = dp.DepthEstimator(cfg).compute(fl)
    rng = float(ref.max()) or 1.0
    best = None
    for mode in ("reflect", "mirror", "nearest"):        # box-filter border of the CPU path
        est = GpuDepthEstimator(cfg, box_mode=mode)
        err = float(np.abs(est.compute(fl) - ref).max()) / rng
        if best is None or err < best[1]:
            best = (mode, err)
    mode, err = best
    if err > tol:
        raise RuntimeError(f"GPU depth differs from CPU by {100 * err:.2f} % (> {100 * tol:.0f} %)")
    if verbose:
        print(f"[GPU] depth on the GPU (CuPy {cp.__version__}): matches CPU within "
              f"{100 * err:.3f} % (box border '{mode}')", flush=True)
    return GpuDepthEstimator(cfg, box_mode=mode)
