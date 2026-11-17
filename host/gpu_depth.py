#!/usr/bin/env python3
"""
gpu_depth.py  —  depth (Gaussian density -> blur -> guided filter) on the GPU with CuPy
=========================================================================================
Same steps and numbers as deltact_plus.DepthEstimator, computed on an NVIDIA GPU
(Jetson Orin: CUDA 12, `pip install cupy-cuda12x`). Used by the depth process of
run_sensor_fast.py / run_sensor_dual.py with the argument  gpu .

Only core CuPy is used (no cupyx / SciPy): on JetPack the system SciPy is built for
NumPy 1.x and importing it from a NumPy 2 environment crashes ("_ARRAY_API not found").
Blur and box filters are written here as separable / cumulative-sum filters with the
same borders as OpenCV.

Safety: make() runs the CPU and the GPU version on the same test flow and compares them.
If they differ by more than 1 % of the depth range it refuses (the caller falls back to
the CPU), so the GPU can never silently change the measurement.
"""
import math
import sys

import numpy as np

import deltact_plus as dp


def _hide_broken_scipy():
    """CuPy itself loads cupyx lazily (e.g. inside cp.clip), and cupyx imports SciPy if it
    exists. On JetPack the system SciPy (/usr/lib/python3/dist-packages, 1.8) is built for
    NumPy 1.x: with NumPy 2 it fails with "_ARRAY_API not found" instead of an ImportError,
    which cupyx does not expect. Mark such a SciPy as absent in THIS process (depth worker
    only), so cupyx sees "no SciPy" and keeps working. A NumPy-2 SciPy is left alone."""
    if int(np.__version__.split(".")[0]) < 2 or "scipy" in sys.modules:
        return None
    try:
        from importlib.metadata import version
        v = version("scipy")
    except Exception:
        return None                                   # no SciPy installed: nothing to do
    try:
        major, minor = (int(x) for x in v.split(".")[:2])
    except ValueError:
        return None
    if (major, minor) >= (1, 13):                     # first SciPy built for NumPy 2
        return None
    for name in ("scipy", "scipy.sparse", "scipy.sparse.linalg", "scipy.linalg",
                 "scipy.ndimage", "scipy.special", "scipy.signal", "scipy.fft",
                 "scipy.stats", "scipy.interpolate", "scipy.optimize"):
        sys.modules[name] = None                      # "import scipy" -> ImportError
    return v


_SCIPY_HIDDEN = _hide_broken_scipy()

try:
    import cupy as cp
    cp.zeros(1).sum()                         # touch the GPU now: fail here, not later
    _ERR = None
except BaseException as e:                    # no CuPy / no CUDA (Raspberry Pi, PC without GPU)
    cp = None
    _ERR = e

# OpenCV border  ->  numpy/cupy pad mode
#   BORDER_REFLECT_101 (OpenCV default)  = 'reflect'
#   BORDER_REFLECT                        = 'symmetric'
#   BORDER_REPLICATE                      = 'edge'
_PAD_MODES = ("reflect", "symmetric", "edge")


def _gauss_kernel(sigma):
    """cv2.getGaussianKernel as used by GaussianBlur(ksize=(0,0)) on float images."""
    ksize = int(round(sigma * 4 * 2 + 1)) | 1
    x = np.arange(ksize, dtype=np.float64) - (ksize - 1) / 2.0
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return (k / k.sum()).astype(np.float32)


class GpuDepthEstimator:
    """Drop-in for dp.DepthEstimator: .cfg, .baseline, .compute(flow) -> numpy depth."""

    def __init__(self, cfg, box_mode="reflect"):
        if cp is None:
            raise RuntimeError(f"CuPy/CUDA not available ({type(_ERR).__name__}: {_ERR})")
        self.cfg = cfg
        self.baseline = None
        self.box_mode = box_mode
        self._grid, self._kern = {}, {}
        # deltact_plus uses cv2.ximgproc.guidedFilter (window 2r+1) when available,
        # otherwise its NumPy fallback (window r): use the same window as the CPU path
        self.box_from_radius = (lambda r: 2 * r + 1) if dp._HAS_XIMG else (lambda r: r)
        self._add_at = getattr(cp.add, "at", None)

    # ── filters (pure CuPy) ──
    def _gauss(self, a, sigma):
        k = self._kern.get(sigma)
        if k is None:
            k = [float(v) for v in _gauss_kernel(sigma)]
            self._kern[sigma] = k
        r = len(k) // 2
        h, w = a.shape
        p = cp.pad(a, ((r, r), (0, 0)), mode="reflect")          # BORDER_REFLECT_101
        out = k[0] * p[0:h]
        for i in range(1, len(k)):
            out += k[i] * p[i:i + h]
        p = cp.pad(out, ((0, 0), (r, r)), mode="reflect")
        out = k[0] * p[:, 0:w]
        for i in range(1, len(k)):
            out += k[i] * p[:, i:i + w]
        return out

    def _box(self, a, k):
        """Normalised box filter of size k x k (OpenCV anchor: centre, k//2 before)."""
        lo, hi = k // 2, k - 1 - k // 2
        h, w = a.shape
        p = cp.pad(a, ((lo, hi), (0, 0)), mode=self.box_mode)
        c = cp.cumsum(p, axis=0, dtype=cp.float64)
        c = cp.concatenate([cp.zeros((1, w), cp.float64), c], axis=0)
        a = (c[k:k + h] - c[0:h]).astype(cp.float32)
        p = cp.pad(a, ((0, 0), (lo, hi)), mode=self.box_mode)
        c = cp.cumsum(p, axis=1, dtype=cp.float64)
        c = cp.concatenate([cp.zeros((h, 1), cp.float64), c], axis=1)
        return ((c[:, k:k + w] - c[:, 0:w]) / float(k * k)).astype(cp.float32)

    def _scatter_add(self, acc, idx, val):
        if self._add_at is not None:
            self._add_at(acc, idx, val)
        else:                                                     # very old CuPy
            import cupyx
            cupyx.scatter_add(acc, idx, val)

    # ── Gaussian density: bilinear splat of the displaced grid points, then blur ──
    def _density(self, flow_g, sigma, sub):
        h, w = flow_g.shape[:2]
        key = (h, w, sub)
        g = self._grid.get(key)
        if g is None:
            ys, xs = np.mgrid[0:h:sub, 0:w:sub]
            g = (cp.asarray(xs, dtype=cp.float32), cp.asarray(ys, dtype=cp.float32))
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
            self._scatter_add(acc, (yi * w + xi).ravel(), wgt.ravel().astype(cp.float32))
        return self._gauss(acc.reshape(h, w), sigma)

    # ── self-guided filter (I = p = feat) ──
    def _guided(self, x, r, eps):
        k = self.box_from_radius(r)
        m = self._box(x, k)
        var = self._box(x * x, k) - m * m
        a = var / (var + eps)
        b = m - a * m
        return self._box(a, k) * x + self._box(b, k)

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
    s = 0.12 * w
    g = np.exp(-((xx - w * 0.45) ** 2 + (yy - h * 0.55) ** 2) / (2 * s * s))
    g2 = 0.6 * np.exp(-((xx - w * 0.7) ** 2 + (yy - h * 0.25) ** 2) / (2 * (0.5 * s) ** 2))
    fl = np.stack([3.0 * g * (xx - w * 0.45) / s + 2.0 * g2,
                   3.0 * g * (yy - h * 0.55) / s - 1.0 * g2], -1).astype(np.float32)
    return fl


def make(cfg, tol=0.01, verbose=True):
    """-> GpuDepthEstimator checked against the CPU, or raises RuntimeError."""
    if cp is None:
        raise RuntimeError(f"CuPy/CUDA not available ({type(_ERR).__name__}: {_ERR})")
    fl = _test_flow(cfg)
    ref = dp.DepthEstimator(cfg).compute(fl)
    rng = float(ref.max()) or 1.0
    best = None
    for mode in _PAD_MODES:                       # box-filter border used by the CPU path
        est = GpuDepthEstimator(cfg, box_mode=mode)
        err = float(np.abs(est.compute(fl) - ref).max()) / rng
        if not math.isfinite(err):
            continue
        if best is None or err < best[1]:
            best = (mode, err)
    if best is None:
        raise RuntimeError("GPU depth produced no finite result")
    mode, err = best
    if err > tol:
        raise RuntimeError(f"GPU depth differs from CPU by {100 * err:.2f} % (> {100 * tol:.0f} %)")
    if verbose:
        print(f"[GPU] depth on the GPU (CuPy {cp.__version__}): matches CPU within "
              f"{100 * err:.3f} % (box border '{mode}')", flush=True)
    return GpuDepthEstimator(cfg, box_mode=mode)
