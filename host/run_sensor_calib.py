"""
run_sensor_calib.py  —  SINGLE CAMERA, HIGH-FPS (Raspberry Pi 5 / Picamera2, or USB on Pi/Ubuntu)
============================================================================
(turunan run_sensor_fast.py; versi kalibrasi/paten)

Versi satu-sensor yang dioptimasi untuk frame-rate tinggi. Memakai pipeline
deltact_plus yang sama (DIS flow + density-difference depth), tapi:
  - SATU kamera  -> tidak berebut ISP/bandwidth (penyebab utama FPS rendah di dual)
  - Open3D MATI secara default (3D heightmap = pemakan FPS terbesar)
  - menggambar arrow di-grid (ringan) + bisa dimatikan ('a')
  - depth bisa dihitung tiap-N-frame ('depth stride') atau dimatikan total
  - exposure / gain / AWB bisa DIKUNCI ('e')  <-- penting untuk ambil data eksperimen

  *** MODIFIKASI 1: CENTER SUPPRESSION (Gaussian taper) untuk hilangkan indenter peak ***
  *** MODIFIKASI 2: NO-CONTACT PROBE (m / M) untuk ambil data Table IV ***
  *** MODIFIKASI 3: NOISE FLOOR TEST (n) lewat pipeline deltact_plus yang sama ***
  *** MODIFIKASI 4 (29 Sep 2026): kalibrasi paten ***
      e) FIX noise floor ('n'): sigma & flicker hanya dari piksel DI DALAM jendela
         gel (dulu area ter-mask -> sigma 0 DN -> rasio 4,8e7 tidak valid).
      a) Burst 'b' MEMBUANG frame awal sampai kamera stabil (min BURST_SKIP_MIN,
         maks BURST_SKIP_MAX). Stabil = ExposureTime & DigitalGain datar DAN
         kecerahan jendela gel berubah < SETTLE_TOL_LVL antar frame, selama
         SETTLE_N frame berturut-turut. (Burst 15:08:19: 3 frame pertama
         ~1,2% lebih gelap -> sumber "drift 0,9%".)
      b) Metadata PER FRAME (SensorTimestamp, ExposureTime, gain, ColourGains,
         Lux, FrameDuration, suhu) + kecerahan jendela disimpan di meta.json:
         'per_frame' (frame yang disimpan) dan 'settle_log' (frame yang dibuang).
         Frame & metadata diambil SEPASANG (capture_request), bukan
         capture_metadata() terpisah.
      c) Guard lipatan undistort: saat start dihitung radius di mana polinomial
         k1/k2 berhenti monoton (r_u fold) dan radius mentah maksimum yang bisa
         di-undistort dengan benar (r_d max). Kalau jendela gel melewatinya ->
         PERINGATAN + dicatat di meta.json ('fold_guard'). Masking output di
         zona lipatan harus dipasang DI DALAM deltact_plus (tempat remap
         dibuat) -> lihat fold_valid_mask() di bawah.
      d) meta.json mencatat 'roi_thresh_trackbar' (posisi slider saat simpan).
  *** MODIFIKASI 5 (30 Sep 2026): distorsi dari checkerboard (profil full) ***
      Profil 'full' memakai FULL_CALIB (fit checkerboard 30 Sep, burst 14:58:31,
      130 sudut, RMS 0,71 px): focal 356,5, k1 -0,857, k2 1,1134, pusat optik
      (419,7, 307,3), roi_undist (270, 123, 271, 361) di citra ter-undistort
      full-frame. Polinomial ini monoton -> tidak melipat.
      Butuh deltact_plus.py versi 30 Sep (field cx, cy, roi_undist). Kalau
      yang terpasang masih versi lama, skrip berhenti dengan pesan ERROR.
      fold_guard sekarang memakai pusat optik hasil fit (bukan tengah gambar)
      dan meta.json mencatat cx, cy, roi_undist, calib_source.
      Argumen 'oldcal' = pakai kalibrasi lama (k 1080p, pusat asumsi) untuk
      perbandingan saja.

CARA PAKAI
  USB (Pi Zero UVC, juga di Ubuntu):
  python3 run_sensor_calib.py usb             # Tactile UVC Cam pertama, 640x480 @ 15
  python3 run_sensor_calib.py usb --usb=camera2   # pilih Zero berdasarkan hostname
  (tanpa Picamera2 -> otomatis mode usb. Di USB: exposure/gain/AWB diatur di Zero,
   jadi e/[ ]/-/= tidak mengunci; 'u' hanya membersihkan trackbar dan meta.json
   mencatat camera_lock = false)
  CSI (Raspberry Pi 5 + Picamera2):
  python3 run_sensor_calib.py                 # default 1280x720 @ 60
  python3 run_sensor_calib.py turbo           # 640x480  @ 120  (FPS tertinggi)
  python3 run_sensor_calib.py fast            # 1280x720 @ 90
  python3 run_sensor_calib.py quality         # 1920x1080 @ 30
  python3 run_sensor_calib.py full            # 820x616 @ 40, FULL FOV sensor (untuk jendela 18x24 mm)
  python3 run_sensor_calib.py full oldcal     # full dengan kalibrasi LAMA (hanya perbandingan)
  python3 run_sensor_calib.py cam1            # pakai kamera index 1 (default 0)
  python3 run_sensor_calib.py nodepth         # flow saja (FPS lebih tinggi lagi)
  python3 run_sensor_calib.py 3d              # AKTIFKAN Open3D (FPS turun)
  (argumen bisa digabung, mis.: python3 run_sensor_calib.py turbo nodepth)

KONTROL
  r = reset baseline (lakukan saat TANPA kontak)
  d = re-deteksi ROI            a = toggle arrow on/off (FPS)
  e = toggle KUNCI exposure/gain/AWB (untuk eksperimen)
  [ / ] = exposure -/+          - / = : gain -/+   (saat terkunci)
  s = snapshot: tampilan (panel) + FRAME MENTAH kamera + ROI mentah (lihat SNAP MENTAH)
  b = burst: tunggu kamera stabil (buang frame awal), lalu BURST_N frame mentah
      berturut-turut -> folder raw_<timestamp>/ (untuk kalibrasi) (S juga bisa)
  c = save config
  g = mulai/stop REKAM fitur slip -> slip_log_1cam.csv (E, dE/dt, Vx, Vy)
  k = balik label baris CSV: static <-> slip (tekan TEPAT saat slip mulai)
  q / ESC = keluar

KONTROL PENGUKURAN NO-CONTACT (Table IV)
  u = MODE UKUR (M juga bisa): nolkan semua setting yang mengganggu + kunci kamera di titik
      operasi TETAP (MEASURE_EXP_US / MEASURE_GAIN), AWB tetap, NR & sharpening
      ISP mati
  m = mulai/batalkan perekaman 120 frame untuk satu decal
  n = NOISE FLOOR TEST (tanpa kontak, ~beberapa detik, jendela membeku sebentar)

  Baris status kedua menampilkan ||F||, false depth, dan low_f secara live.
  Statusnya: settling -> STABLE -> tekan m. Kalau BLOCKED, alasannya dicetak
  di baris merah di atasnya.

  PENTING: tiga fitur di skrip ini merusak pengukuran no-contact dan semuanya
  aktif secara default:
    center_suppress  mengalikan flow dengan mask -> mengecilkan yang diukur
    rebase hold      reset baseline saat idle -> menghapus drift yang diukur
    deadzone/ema/tavg  runtime stage Sec. VII -> Table IV melaporkan TANPA ini
  Tekan 'M' untuk menonaktifkan ketiganya sekaligus. Probe menolak merekam
  sampai bersih, dan menandai baris hasil sebagai NONCONFORMING kalau dipaksa.

  Kenapa M mengunci di angka TETAP, bukan di nilai AE saat itu: AE memilih
  exposure berbeda untuk tiap decal (decal gelap -> exposure panjang), sehingga
  mengunci "di nilai AE" tetap membuat decal diukur pada titik operasi berbeda.
  Semua decal harus diukur di exposure/gain yang sama.

SNAP MENTAH ('s' dan 'b')
  Dulu 's' hanya menyimpan panel tampilan (teks status, panel flow/depth, di-resize
  ke grid kerja). Itu TIDAK bisa dipakai untuk kalibrasi. Sekarang:
    raw_XX.png   frame kamera APA ADANYA (resolusi profil, fisheye, belum di-undistort,
                 tanpa teks, tanpa trackbar gain/bright)
    roi_XX.png   ROI mentah yang masuk ke DIS (sudah undistort + crop + resize oleh
                 deltact_plus), juga tanpa gain/bright tampilan
    meta.json    exposure, gain, colour gains, NR, mode ukur, profil, undistort, ROI,
                 per_frame (metadata tiap frame), settle_log, fold_guard
  'b' menyimpan BURST_N frame BERBEDA berturut-turut tanpa memblok loop.
  Untuk kalibrasi: tekan 'u' dulu (titik operasi tetap), tunggu STABLE, baru 'b'.
  Catatan: frame yang disimpan = frame yang diproses loop (loop ~13 fps di mode
  full), jadi jarak antar frame burst ~2-3 frame kamera. Lihat SensorTimestamp.

  Mode full (sejak 30 Sep): roi_XX.png berasal dari undistort checkerboard
  (tidak melipat) dan dipotong ke roi_undist. Untuk kalibrasi warna tetap
  pakai raw_XX.png (data mentah). Dengan 'oldcal', roi_XX.png TIDAK valid.

NOISE FLOOR TEST ('n')
  Mengambil N frame tanpa kontak, lalu menjalankan DUA sekuens lewat
  dp.TactileSensor baru dengan cfg yang sama (undistort, ROI, DIS identik):
    measured     frame nyata
    noise-only   gambar bersih (rata-rata N frame) + noise sensor sintetis yang
                 diukur dari data (sigma per kanal, per intensitas). Tidak ada
                 gerakan -> flow di sini murni noise.
  plus 'gain-matched' (frame nyata dinormalisasi kecerahannya ke baseline) untuk
  mendeteksi kedip LED, dan struktur gerakan (global shift / coherent) untuk
  membedakan getaran rig, gerakan gel, dan noise.
  Hasil dicetak di terminal dan disimpan ke noise_floor_<timestamp>.json.

KONTROL CENTER SUPPRESSION (di Kontrol panel)
  - "center_suppress" : 0..100 (kuat suppression, 0=off, 100=maksimal)
  - "supp_radius %" : 10..60 (radius Gaussian, % dari ROI)

CATATAN FPS
  Center suppression (Gaussian mask) sangat murah: O(WxH) -> negligible overhead.
  No-contact probe juga murah (~1 FFT per 10 frame).
  Bottleneck tetap sama: (1) Open3D, (2) depth guided filter, (3) menggambar arrow.

VIEWER 3D TERPISAH (RINGAN)
  Script ini MEM-PUBLISH depth ke shared memory (/dev/shm) tiap frame (sangat
  murah, ~memcpy). Jalankan 3D di PROSES TERPISAH supaya Open3D yang berat TIDAK
  menurunkan FPS loop ini:
      Terminal 1:  python3 run_sensor_calib.py turbo      # JANGAN pakai arg '3d'
      Terminal 2:  python3 view_3d_live.py
  Butuh file sensor_share.py & view_3d_live.py di folder yang sama.
  Catatan: viewer butuh depth -> JANGAN pakai 'nodepth' kalau mau lihat 3D.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import threading
from typing import Optional

import numpy as np
import cv2
import deltact_plus as dp
try:
    from dt3_slip import (SLIP_BINS, SlipLogger,  # << fitur slip (Hu et al. 2024)
                          make_slip_grid, slip_features)
except ImportError:
    sys.exit("[ERROR] dt3_slip.py tidak ada: salin ke folder yang sama (fitur slip).")
try:
    from sensor_share import DepthPublisher   # << publish depth ke shared memory (viewer 3D terpisah)
except ImportError:
    class DepthPublisher:                     # opsional: hanya untuk view_3d_live.py
        def __init__(self, *a, **k): pass
        def publish(self, *a, **k): pass
        def close(self): pass
try:
    from nocontact_probe import NoContactProbe  # << pengukuran no-contact (Table IV)
except ImportError:
    class NoContactProbe:                     # opsional: hanya perekaman 'm'
        recording = False
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): pass
        def overlay(self, panel): return panel
        def toggle_record(self, *a, **k):
            print("[INFO] nocontact_probe.py tidak ada -> perekaman 'm' tidak tersedia")

try:
    from picamera2 import Picamera2
    from libcamera import controls as libcontrols  # noqa: F401 (untuk AE/AWB enums bila tersedia)
except ImportError:
    Picamera2 = None   # Ubuntu / tanpa CSI: hanya mode 'usb'

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "sensor_config.json")   # sama dengan run_sensor_fast

# Manfaatkan semua core untuk operasi OpenCV (blur/resize/remap)
cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 4))
except Exception:
    pass


# Titik operasi TETAP untuk mode ukur (sama dengan capture_nocontact.py / paper)
MEASURE_EXP_US = 4536
MEASURE_GAIN = 1.0
MEASURE_COLOUR_GAINS = (1.2, 2.0)

# Burst frame mentah ('b') untuk kalibrasi / test strip
BURST_N = 10

# ── Settling sebelum burst (MODIFIKASI 4a) ──────────────────────────────────
# Frame yang diproses loop dibuang sampai kamera stabil. Kriteria "stabil":
# pada SETTLE_N selisih berturut-turut (SETTLE_N+1 frame terakhir)
#   - rentang ExposureTime  <= SETTLE_TOL_EXP_US
#   - rentang DigitalGain   <= SETTLE_TOL_DGAIN
#   - |perubahan relatif kecerahan jendela| < SETTLE_TOL_LVL tiap langkah
# Minimal BURST_SKIP_MIN frame selalu dibuang. Kalau belum stabil setelah
# BURST_SKIP_MAX frame, burst tetap jalan tapi ditandai settle_timeout=true.
BURST_SKIP_MIN = 3
BURST_SKIP_MAX = 40
SETTLE_N = 3
SETTLE_TOL_EXP_US = 2
SETTLE_TOL_DGAIN = 1e-3
SETTLE_TOL_LVL = 1.0         # NONAKTIF: fluktuasi ternyata intensitas LED (sd ~0,5 %),
                             # bukan transien kamera -> lvl hanya dicatat, bukan kriteria

# Metadata yang disimpan PER FRAME (MODIFIKASI 4b)
PF_KEYS = ("SensorTimestamp", "ExposureTime", "AnalogueGain", "DigitalGain",
           "ColourGains", "Lux", "FrameDuration", "SensorTemperature")

# Jendela gel (x0, y0, x1, y1) dalam piksel frame mentah. Di luar kotak ini
# di-nol-kan SEBELUM masuk deltact_plus, supaya deteksi ROI (ambang kecerahan)
# tidak menangkap cincin LED yang lebih terang dari kertas. Frame mentah yang
# DISIMPAN tetap utuh. None = tidak dimasking.
WINDOW_BOX = None

# ── Kalibrasi distorsi profil 'full' dari checkerboard (MODIFIKASI 5) ───────
# Fit pinhole + radial 2 suku, f tetap, 130 sudut dalam, satu pose datar.
# roi_undist = (x, y, w, h) di citra ter-undistort full-frame (newK = K),
# rasio 3:4 -> grid kerja 240x320 ~14,8 px/mm.
FULL_CALIB = dict(
    focal=356.5,
    dist=(-0.857, 1.1134, 0.0, 0.0, 0.0),
    cx=419.7, cy=307.3,
    roi_undist=(270, 123, 271, 361),
    source="checkerboard_18x24 burst 2026-09-30 14:58:31; 130 corners; "
           "RMS 0.71 px (max 2.9); monotonic; grid straight 0.91 px",
)


def apply_window(frame, box):
    if box is None:
        return frame
    x0, y0, x1, y1 = box
    out = np.zeros_like(frame)
    out[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Helper kalibrasi (MODIFIKASI 4)
# ─────────────────────────────────────────────────────────────────────────────
def window_level(frame, box, step=4):
    """Kecerahan rata-rata (0..1) di jendela gel, subsampel tiap `step` px (murah)."""
    if frame is None:
        return float("nan")
    if box is None:
        sub = frame[::step, ::step]
    else:
        x0, y0, x1, y1 = box
        sub = frame[y0:y1:step, x0:x1:step]
    return float(sub.mean()) / 255.0


def md_subset(md):
    """Ambil PF_KEYS dari metadata satu frame (JSON-able)."""
    md = md or {}
    return {k: _jsonable(md.get(k)) for k in PF_KEYS if k in md}


def _range(vals):
    v = [float(x) for x in vals if x is not None]
    return (max(v) - min(v)) if len(v) >= 2 else 0.0


def settled(log, n=SETTLE_N, tol_exp=SETTLE_TOL_EXP_US, tol_dg=SETTLE_TOL_DGAIN,
            tol_lvl=SETTLE_TOL_LVL):
    """log: list dict per frame (lvl + metadata). True kalau n langkah terakhir stabil."""
    if len(log) < n + 1:
        return False
    last = log[-(n + 1):]
    if _range([e.get("ExposureTime") for e in last]) > tol_exp:
        return False
    if _range([e.get("DigitalGain") for e in last]) > tol_dg:
        return False
    lv = [e.get("lvl") for e in last]
    if any(x is None or not np.isfinite(x) for x in lv):
        return False
    for a, b in zip(lv[:-1], lv[1:]):
        if abs(b - a) / max(a, 1e-6) >= tol_lvl:
            return False
    return True


def _fold_radius(k1, k2):
    """r_u (ter-normalisasi) tempat r_d = r_u(1+k1 r^2+k2 r^4) berhenti monoton.
    Turunan: 1 + 3 k1 x + 5 k2 x^2 = 0, x = r_u^2. None = tidak melipat."""
    roots = np.roots([5.0 * k2, 3.0 * k1, 1.0]) if k2 != 0 else (
        np.array([-1.0 / (3.0 * k1)]) if k1 != 0 else np.array([]))
    xs = [float(r.real) for r in np.atleast_1d(roots) if abs(r.imag) < 1e-12 and r.real > 0]
    return float(np.sqrt(min(xs))) if xs else None


def undistort_fold_info(focal, dist, cw, ch, box, cx=None, cy=None):
    """Batas validitas undistort k1/k2 di frame mentah (MODIFIKASI 4c).

    Pusat optik = (cx, cy) kalau diberikan (hasil fit checkerboard), selain
    itu diasumsikan tengah gambar. Focal sama di x/y. Hasil:
      r_u_fold       radius ter-normalisasi tempat pemetaan melipat
      r_d_max_px     radius MENTAH maksimum yang punya invers (px)
      frac_window_beyond  fraksi piksel jendela gel di luar r_d_max_px
      window_max_r_px     radius mentah terjauh di jendela gel
    """
    k1 = float(dist[0]); k2 = float(dist[1]) if len(dist) > 1 else 0.0
    center_assumed = cx is None or cy is None
    cx = cw / 2.0 if cx is None else float(cx)
    cy = ch / 2.0 if cy is None else float(cy)
    ru = _fold_radius(k1, k2)
    info = dict(k1=k1, k2=k2, focal_px=float(focal), center_px=[cx, cy],
                center_assumed=center_assumed, r_u_fold=ru)
    x0, y0, x1, y1 = box if box is not None else (0, 0, cw, ch)
    yy, xx = np.mgrid[y0:y1:2, x0:x1:2]
    r = np.hypot(xx - cx, yy - cy)
    if ru is None:
        info.update(r_d_max_px=None, frac_window_beyond=0.0,
                    window_max_r_px=float(r.max()), ok=True)
        return info
    rd_max = ru * (1 + k1 * ru ** 2 + k2 * ru ** 4) * float(focal)
    frac = float((r > rd_max).mean())
    info.update(r_d_max_px=float(rd_max), frac_window_beyond=frac,
                window_max_r_px=float(r.max()), ok=frac == 0.0)
    return info


def fold_valid_mask(K_new, dist, out_w, out_h, margin=0.05):
    """Mask zona VALID pada gambar HASIL undistort (True = boleh dipakai).

    Untuk dipasang di deltact_plus, tepat setelah initUndistortRectifyMap:
        valid = fold_valid_mask(K_new, dist, W, H)
        und[~valid] = 0                      # sebelum deteksi ROI / DIS
    K_new = matriks kamera yang dipakai sebagai newCameraMatrix di remap.
    Piksel dengan 1 + 3 k1 r^2 + 5 k2 r^4 <= margin berada di zona lipatan:
    isinya pantulan dari radius lebih kecil, bukan citra sebenarnya.
    """
    k1 = float(dist[0]); k2 = float(dist[1]) if len(dist) > 1 else 0.0
    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    ru2 = ((xs - K_new[0, 2]) / K_new[0, 0]) ** 2 + ((ys - K_new[1, 2]) / K_new[1, 1]) ** 2
    return (1.0 + 3.0 * k1 * ru2 + 5.0 * k2 * ru2 ** 2) > margin


def deltact_supports_undist_roi():
    """True kalau deltact_plus.py yang terpasang sudah versi 30 Sep (cx, cy, roi_undist)."""
    try:
        c = dp.SensorConfig()
    except Exception:
        return False
    return all(hasattr(c, a) for a in ("cx", "cy", "roi_undist"))


# ─────────────────────────────────────────────────────────────────────────────
#  Center Suppression (Gaussian taper)
# ─────────────────────────────────────────────────────────────────────────────
def make_gaussian_center_mask(h, w, radius_pct=10.0, sigma_scale=1.0):
    """
    Buat Gaussian taper mask untuk menekan artifact di tengah ROI.

    Args:
        h, w: ukuran mask (ROI height/width)
        radius_pct: radius suppression sebagai % dari min(h,w) (default 10%)
        sigma_scale: kontrol kecuraman Gaussian (> 1 = lebih tajam, < 1 = lebih halus)

    Returns:
        mask (h, w, float32) di range [0, 1], 0 di tengah (suppressed), 1 di tepi
    """
    cy, cx = h / 2.0, w / 2.0
    min_dim = min(h, w)
    radius = (radius_pct / 100.0) * min_dim
    sigma = radius / (2.0 * sigma_scale)  # sigma_scale > 1 = lebih tajam

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2

    # Gaussian: exp(-dist^2 / (2*sigma^2))
    # Kita balik: 1 - Gaussian sehingga 0 di center, 1 di tepi
    mask = 1.0 - np.exp(-dist2 / (2.0 * sigma ** 2))
    return mask.astype(np.float32)


def apply_center_suppression(data, mask, strength=50):
    """
    Terapkan center suppression pada data (depth atau flow magnitude).

    Args:
        data: (h, w) atau (h, w, c) array
        mask: (h, w) Gaussian taper mask [0..1]
        strength: 0..100, kuat suppression

    Returns:
        suppressed data dengan format sama
    """
    if data is None or strength == 0:
        return data

    alpha = strength / 100.0

    if len(data.shape) == 2:
        # Scalar field (depth)
        return data * (1.0 - alpha * (1.0 - mask[:, :])).astype(data.dtype)
    else:
        # Vector field (flow): suppress magnitude
        h, w = mask.shape
        mask_3d = (1.0 - alpha * (1.0 - mask[:h, :w]))
        if data.shape[2] == 2:  # flow (h, w, 2)
            return (data * mask_3d[:, :, np.newaxis]).astype(data.dtype)
        else:  # rgb atau lainnya
            return (data * mask_3d[:, :, np.newaxis]).astype(data.dtype)


# ─────────────────────────────────────────────────────────────────────────────
#  Display helpers
# ─────────────────────────────────────────────────────────────────────────────
def label_bottom(img, t, c=(0, 255, 0)):
    """Tulis label di BAWAH gambar (jangan ganggu konten)"""
    h, w = img.shape[:2]
    y_pos = h - 8  # 8 pixel dari bawah
    cv2.putText(img, t, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, t, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return img


def quiver_fast(img, flow, step, scale, thresh2=0.04):
    """Gambar vektor flow hanya pada titik grid (ringan: O(grid), bukan O(piksel))."""
    h, w = flow.shape[:2]
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    fy = flow[gy, gx]                       # (ny, nx, 2)
    mag2 = fy[..., 0] ** 2 + fy[..., 1] ** 2
    sel = np.where(mag2 > thresh2)
    for r, c in zip(*sel):
        x = int(xs[c]); y = int(ys[r])
        dx, dy = fy[r, c]
        cv2.arrowedLine(img, (x, y), (int(x + dx * scale), int(y + dy * scale)),
                        (0, 255, 0), 1, tipLength=0.4)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    return img


# ─────────────────────────────────────────────────────────────────────────────
#  Kamera (Picamera2) + thread pembaca frame-terbaru
# ─────────────────────────────────────────────────────────────────────────────
def open_camera(cam_num, cw, ch, fr, raw=None):
    picam2 = Picamera2(cam_num)
    kw = {}
    if raw is not None:
        # paksa mode sensor tertentu (mis. 1640x1232 = full FOV, binned 2x2)
        kw["raw"] = {"size": tuple(raw)}
    cfg = picam2.create_video_configuration(
        main={"size": (cw, ch), "format": "RGB888"},  # RGB888 -> byte-order BGR di numpy (langsung utk OpenCV)
        controls={"FrameRate": float(fr)},
        buffer_count=4,                                 # pipeline lebih mulus utk FPS tinggi
        **kw,
    )
    picam2.configure(cfg)
    picam2.start()
    return picam2


def _jsonable(v):
    try:
        json.dumps(v); return v
    except TypeError:
        try:
            return list(v)          # mis. tuple/libcamera Rectangle
        except TypeError:
            return str(v)


def save_raw_bundle(folder, idx, frame, res):
    """Simpan frame kamera mentah + ROI mentah (tanpa overlay/gain tampilan)."""
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, f"raw_{idx:02d}.png"), frame)
    roi = None if res is None else res.get("roi")
    if roi is not None:
        cv2.imwrite(os.path.join(folder, f"roi_{idx:02d}.png"), roi)
    return roi is not None


def write_meta(folder, cam, cfg, profile_name, profile, ae_locked, measure_mode,
               n_saved, seqs, note="", extra=None):
    md = cam.get_metadata()
    keys = ("ExposureTime", "AnalogueGain", "DigitalGain", "ColourGains",
            "ColourTemperature", "Lux", "SensorTemperature", "FrameDuration", "ScalerCrop",
            "NoiseReductionMode", "Sharpness", "AeLocked")
    ru = getattr(cfg, "roi_undist", None)
    meta = dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"), note=note,
        n_frames=n_saved, frame_seq=seqs,
        ae_locked=ae_locked, measure_mode=measure_mode,
        camera=("usb " + str(profile["device"])) if "device" in profile else "csi",
        camera_lock=bool(ae_locked) and "device" not in profile,   # UVC: never locked
        measure_point=dict(exposure_us=MEASURE_EXP_US, gain=MEASURE_GAIN,
                           colour_gains=MEASURE_COLOUR_GAINS),
        camera_metadata={k: _jsonable(md.get(k)) for k in keys if k in md},
        profile=profile_name, capture_size=[profile["cw"], profile["ch"]],
        fps_target=profile["fr"],
        undistort=dict(focal=float(cfg.focal), dist=list(cfg.dist),
                       cx=_jsonable(getattr(cfg, "cx", None)),
                       cy=_jsonable(getattr(cfg, "cy", None)),
                       roi_undist=None if ru is None else [int(v) for v in ru],
                       calib_source=profile.get("calib_source", "legacy 1080p fov200"),
                       roi_thresh=int(cfg.roi_thresh), valid_margin=int(cfg.valid_margin)),
        work_size=[int(cfg.work_w), int(cfg.work_h)],
        window_box=None if WINDOW_BOX is None else list(WINDOW_BOX),
        files="raw_XX.png = frame kamera mentah (BGR, fisheye); "
              "roi_XX.png = ROI mentah deltact_plus (undistort+crop+resize)")
    if extra:
        meta.update(extra)
    with open(os.path.join(folder, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    if not measure_mode:
        print("[SNAP] PERINGATAN: bukan mode ukur ('u'); exposure/AWB/NR mungkin "
              "berbeda dari titik operasi pengukuran.")


class CameraReader:
    """Selalu menyimpan frame TERBARU (drop frame lama) -> capture tak memblok proses.

    Setiap frame diberi nomor urut (seq). Loop utama melewati frame yang seq-nya
    sama dengan sebelumnya: kalau loop lebih cepat dari kamera, frame yang sama
    akan diproses dua kali dan menghasilkan flow ~0 palsu, yang menurunkan ||F||
    rata-rata tanpa ada perbaikan nyata.

    MODIFIKASI 4b: frame diambil lewat capture_request(), sehingga metadata
    (ExposureTime, DigitalGain, SensorTimestamp, ...) adalah milik frame yang
    SAMA, bukan frame sesudahnya seperti capture_metadata().
    """
    def __init__(self, picam2, name="cam"):
        self._picam2 = picam2
        self._frame = None
        self._md = {}
        self._seq = 0
        self._lock = threading.Lock()
        self._ok = False
        self._running = False
        self._t = threading.Thread(target=self._run, daemon=True, name=name)

    def start(self):
        self._running = True
        self._t.start()

    def stop(self):
        self._running = False
        self._t.join(timeout=2)
        try:
            self._picam2.stop(); self._picam2.close()
        except Exception:
            pass

    def read(self):
        """-> (ok, frame, seq)"""
        with self._lock:
            return self._ok, (self._frame if self._ok else None), self._seq

    def read_md(self):
        """-> (ok, frame, seq, metadata frame itu)"""
        with self._lock:
            return (self._ok, (self._frame if self._ok else None), self._seq,
                    self._md if self._ok else {})

    def get_metadata(self):
        """Metadata frame TERBARU yang sudah ditangkap (tidak memblok)."""
        with self._lock:
            if self._ok and self._md:
                return dict(self._md)
        try:
            return self._picam2.capture_metadata()
        except Exception:
            return {}

    def set_controls(self, **kw):
        try:
            self._picam2.set_controls(kw)
        except Exception as e:
            print(f"[WARN] set_controls gagal: {e}")

    def _run(self):
        while self._running:
            try:
                req = self._picam2.capture_request()
            except Exception:
                continue
            try:
                f = req.make_array("main")
                md = req.get_metadata()
            except Exception:
                f, md = None, {}
            finally:
                try:
                    req.release()
                except Exception:
                    pass
            if f is not None:
                with self._lock:
                    self._frame = f
                    self._md = md or {}
                    self._seq += 1
                    self._ok = True


# ─────────────────────────────────────────────────────────────────────────────
#  Noise floor test (lewat pipeline deltact_plus yang sama)
# ─────────────────────────────────────────────────────────────────────────────
NF_BINS = 32
NF_COHERENT_SIGMA = 16.0   # px pada grid kerja, ~2x patch DIS


def _flow_stats(flow: np.ndarray, tau: float) -> tuple[float, float, float, float]:
    """-> (||F|| mean, % piksel |F|>tau, global shift, coherent ||F||)"""
    fl = flow.astype(np.float32)
    m = np.linalg.norm(fl, axis=2)
    shift = float(np.hypot(fl[..., 0].mean(), fl[..., 1].mean()))
    sx = cv2.GaussianBlur(fl[..., 0], (0, 0), NF_COHERENT_SIGMA, borderType=cv2.BORDER_REFLECT)
    sy = cv2.GaussianBlur(fl[..., 1], (0, 0), NF_COHERENT_SIGMA, borderType=cv2.BORDER_REFLECT)
    coh = float(np.hypot(sx, sy).mean())
    return float(m.mean()), 100.0 * float((m > tau).mean()), shift, coh


def _run_sequence(cfg, cw, ch, frames, warmup: int, tau: float) -> np.ndarray:
    """Jalankan frame lewat TactileSensor BARU (baseline dari frame awal)."""
    s = dp.TactileSensor(cfg, cw, ch)
    out = []
    for k, f in enumerate(frames):
        res = s.process(f)
        if res is None or k < warmup:
            continue
        out.append(_flow_stats(res["flow"], tau))
    return np.array(out) if out else np.zeros((0, 4))


def run_noise_floor(cam, cfg, cw, ch, n_frames=40, tau=0.5, warmup=2, seed=0, window_box=None):
    print("\n[NF] ===== NOISE FLOOR TEST (jangan sentuh sensor) =====")
    unsafe = []
    if cfg.flow_deadzone > 0: unsafe.append(f"deadzone={cfg.flow_deadzone}")
    if cfg.flow_ema > 0: unsafe.append(f"ema={cfg.flow_ema}")
    if cfg.flow_ema_release > 0: unsafe.append(f"release={cfg.flow_ema_release}")
    if cfg.temporal_avg > 1: unsafe.append(f"tavg={cfg.temporal_avg}")
    if unsafe:
        print(f"[NF] PERINGATAN: runtime filter aktif ({', '.join(unsafe)}). "
              "Tekan 'M' dulu supaya angkanya sebanding dengan Table IV.")

    md = cam.get_metadata()
    exp, gain = md.get("ExposureTime"), md.get("AnalogueGain")
    dgain, cgain = md.get("DigitalGain"), md.get("ColourGains")
    print(f"[NF] kamera: exp={exp}us gain={gain} digital={dgain} colour={cgain}")
    if exp is not None and abs(float(exp) - MEASURE_EXP_US) > 0.1 * MEASURE_EXP_US:
        print(f"[NF] PERINGATAN: exposure {exp}us != titik operasi ukur "
              f"{MEASURE_EXP_US}us (sudah tekan 'M'?)")

    # ---- kumpulkan frame berbeda ------------------------------------------
    frames, last_seq = [], None
    t_end = time.time() + 15.0 + n_frames * 0.2
    while len(frames) < n_frames and time.time() < t_end:
        ok, f, seq = cam.read()
        if ok and seq != last_seq:
            frames.append(apply_window(f.copy(), window_box)); last_seq = seq
        else:
            time.sleep(0.002)
    if len(frames) < 10:
        print("[NF] gagal: frame terlalu sedikit"); return
    K = len(frames)
    print(f"[NF] {K} frame diambil, menghitung...")

    # ---- gambar bersih + sigma per kanal per intensitas -------------------
    acc = np.zeros(frames[0].shape, np.float64)
    for f in frames:
        acc += f
    clean = (acc / K).astype(np.float32)
    g = cv2.cvtColor(np.clip(clean, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    # Hanya piksel DI DALAM jendela gel. Tanpa ini, area luar jendela (di-nol-kan
    # apply_window) = gradien 0 -> terpilih sebagai "flat" -> sigma = 0 DN ->
    # noise-only floor = 0 dan rasio/diagnosis tidak valid (bug 29 Sep).
    valid_px = np.ones(g.shape, bool)
    if window_box is not None:
        x0, y0, x1, y1 = window_box
        valid_px[:] = False
        valid_px[y0 + 5:y1 - 5, x0 + 5:x1 - 5] = True
    flat = valid_px & (grad <= np.percentile(grad[valid_px], 30))

    sigma_maps, sigma_flat = [], []
    for c in range(clean.shape[2]):
        bins = np.clip((clean[..., c] / 256.0 * NF_BINS).astype(int), 0, NF_BINS - 1)
        bf = bins[flat]
        s2, cnt = np.zeros(NF_BINS), np.zeros(NF_BINS)
        for i in range(1, K):
            d = frames[i][..., c].astype(np.float32) - frames[i - 1][..., c]
            s2 += np.bincount(bf, d[flat].astype(np.float64) ** 2, NF_BINS)
            cnt += np.bincount(bf, minlength=NF_BINS)
        valid = cnt > 500
        if not valid.any():
            print("[NF] gagal: noise tidak bisa diestimasi"); return
        idx = np.where(valid)[0]
        sig = np.sqrt(s2[idx] / cnt[idx] / 2.0)
        sigma_maps.append(np.interp(np.arange(NF_BINS), idx, sig)[bins].astype(np.float32))
        sigma_flat.append(float(np.sqrt(s2[valid].sum() / cnt[valid].sum() / 2.0)))
    sigma_map = np.dstack(sigma_maps) * valid_px[..., None]   # tanpa noise di area ter-mask

    means = np.array([float(f[valid_px].mean()) for f in frames])
    flicker_pct = 100.0 * float(np.std(np.diff(means)) / np.sqrt(2) / max(means.mean(), 1e-6))

    # ---- tiga sekuens lewat pipeline yang sama ----------------------------
    c2 = copy.deepcopy(cfg)
    try:
        c2.enable_depth = False          # depth tidak dipakai di sini, hemat waktu
    except Exception:
        pass

    rng = np.random.default_rng(seed)

    def synthetic():
        for _ in range(K):
            n = rng.standard_normal(clean.shape, dtype=np.float32) * sigma_map
            yield np.clip(np.rint(clean + n), 0, 255).astype(np.uint8)

    ref_mean = float(frames[0].mean())

    def gain_matched():
        for f in frames:
            k = ref_mean / max(float(f.mean()), 1e-6)
            yield np.clip(np.rint(f.astype(np.float32) * k), 0, 255).astype(np.uint8)

    meas = _run_sequence(c2, cw, ch, frames, warmup, tau)
    noise = _run_sequence(c2, cw, ch, synthetic(), warmup, tau)
    meas_g = _run_sequence(c2, cw, ch, gain_matched(), warmup, tau)
    if len(meas) == 0 or len(noise) == 0:
        print("[NF] gagal: TactileSensor tidak mengembalikan flow (ROI?)"); return

    ratio = meas[:, 0].mean() / max(noise[:, 0].mean(), 1e-9)
    g_drop = 1.0 - meas_g[:, 0].mean() / max(meas[:, 0].mean(), 1e-9)
    coh_ratio = meas[:, 3].mean() / max(noise[:, 3].mean(), 1e-9)
    shift_frac = meas[:, 2].mean() / max(meas[:, 3].mean(), 1e-9)

    print(f"\n[NF] noise sensor (flat px, DN)  B {sigma_flat[0]:.2f}  "
          f"G {sigma_flat[1]:.2f}  R {sigma_flat[2]:.2f}   flicker {flicker_pct:.3f}%")
    print(f"[NF] {'':<24}{'||F|| px':>14}{f'|F|>{tau} %':>12}{'shift':>9}{'coherent':>10}")
    for lab, a in (("noise-only floor", noise), ("measured", meas),
                   ("measured, gain-matched", meas_g)):
        print(f"[NF] {lab:<24}{a[:, 0].mean():>8.4f}+/-{a[:, 0].std():.3f}"
              f"{a[:, 1].mean():>12.2f}{a[:, 2].mean():>9.4f}{a[:, 3].mean():>10.4f}")
    print(f"[NF] measured / noise floor = {ratio:.2f}   "
          f"gain matching removes {100*g_drop:.0f}%   "
          f"coherent ratio {coh_ratio:.2f}   shift/coherent {shift_frac:.2f}")

    diag = []
    if ratio < 1.25:
        diag.append("AT NOISE FLOOR: ||F|| adalah noise sensor, bukan gerakan.")
    else:
        if g_drop > 0.3:
            diag.append("FLICKER: normalisasi kecerahan menghapus sebagian besar flow. "
                        "Cek driver LED (PWM vs exposure 2 ms).")
        if coh_ratio > 2.0 and shift_frac > 0.5:
            diag.append("RIGID MOTION: seluruh gambar bergeser. Getaran kamera/rig.")
        elif coh_ratio > 2.0:
            diag.append("LOCAL COHERENT MOTION: gel/decal bergerak (creep, suhu) atau aliran udara.")
        if g_drop <= 0.3 and coh_ratio <= 2.0:
            diag.append("EXCESS INKOHEREN: bukan flicker, bukan gerakan. Noise yang tidak "
                        "tertangkap model putih (NR/sharpening ISP, noise berkorelasi).")
    for d in diag:
        print(f"[NF] -> {d}")

    fn = f"noise_floor_{int(time.time())}.json"
    with open(fn, "w") as fh:
        json.dump(dict(
            frames=K, tau=tau, warmup=warmup,
            camera=dict(exposure_us=exp, analogue_gain=gain, digital_gain=dgain,
                        colour_gains=cgain),
            undistort=dict(focal=float(cfg.focal), dist=list(cfg.dist),
                           cx=_jsonable(getattr(cfg, "cx", None)),
                           cy=_jsonable(getattr(cfg, "cy", None)),
                           roi_undist=_jsonable(getattr(cfg, "roi_undist", None)),
                           work_size=[int(cfg.work_w), int(cfg.work_h)]),
            runtime_filters_active=unsafe,
            sigma_flat_bgr=sigma_flat, flicker_pct=flicker_pct,
            noise_floor=noise.mean(axis=0).tolist(),
            measured=meas.mean(axis=0).tolist(),
            measured_gain_matched=meas_g.mean(axis=0).tolist(),
            columns=["flow_mean_px", "pct_above_tau", "global_shift_px", "coherent_px"],
            ratio=ratio, gain_drop=g_drop, coherent_ratio=coh_ratio,
            shift_over_coherent=shift_frac, diagnosis=diag), fh, indent=2, default=str)
    print(f"[NF] disimpan ke {fn}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  (Opsional) Open3D single-window heightmap  — hanya dipakai bila arg '3d'
# ─────────────────────────────────────────────────────────────────────────────
def make_3d_window(z_scale=5.0):
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    import queue as _queue

    STEP, DENT = 4, 0.01

    def build_pcd(depth, color, zs):
        h, w = depth.shape
        rows = np.arange(0, h, STEP); cols = np.arange(0, w, STEP)
        rr, cc = np.meshgrid(rows, cols, indexing="ij"); rr = rr.ravel(); cc = cc.ravel()
        d = depth[rr, cc].astype(np.float32)
        X = cc.astype(np.float32) / max(w - 1, 1)
        Y = (max(h - 1, 1) - rr).astype(np.float32) / max(h - 1, 1)
        d99 = float(np.percentile(d, 99))
        if d99 < DENT:
            Z = np.zeros_like(d); C = np.full((len(Z), 3), 0.35, np.float32)
        else:
            dc = np.clip(d, 0.0, d99); Z = dc * zs
            if color is not None:
                if color.shape[:2] != (h, w):
                    color = cv2.resize(color, (w, h))
                C = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32)[rr, cc] / 255.0
            else:
                rng = float(dc.max() - dc.min()); dn = (dc - dc.min()) / (rng + 1e-6)
                C = cv2.applyColorMap((dn * 255).astype(np.uint8).reshape(-1, 1),
                                      cv2.COLORMAP_JET).reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.column_stack([X, Y, Z]))
        pcd.colors = o3d.utility.Vector3dVector(C)
        return pcd

    class W:
        def __init__(self):
            self.q = _queue.Queue(maxsize=1)
            self.zs = z_scale
            self.running = False
            self.pcd = o3d.geometry.PointCloud()
            self._added = False
            self.mat = rendering.MaterialRecord(); self.mat.shader = "defaultUnlit"; self.mat.point_size = 4.0
            self.t = threading.Thread(target=self._run, daemon=True, name="o3d")

        def start(self): self.running = True; self.t.start()

        def stop(self):
            self.running = False
            try: self.q.put_nowait(None)
            except Exception: pass
            self.t.join(timeout=3)

        def push(self, depth, color):
            try: self.q.get_nowait()
            except Exception: pass
            try: self.q.put_nowait((depth.copy(), None if color is None else color.copy()))
            except Exception: pass

        def _run(self):
            app = gui.Application.instance; app.initialize()
            win = app.create_window("Tactile | Heightmap", 720, 600)
            wd = gui.SceneWidget(); wd.scene = rendering.Open3DScene(win.renderer)
            wd.scene.set_background([0.05, 0.05, 0.05, 1.0]); wd.scene.show_axes(True)
            win.add_child(wd)
            rc = win.content_rect; wd.frame = gui.Rect(rc.x, rc.y, rc.width, rc.height)
            win.set_on_layout(lambda th: setattr(wd, "frame", gui.Rect(*[win.content_rect.x, win.content_rect.y, win.content_rect.width, win.content_rect.height])))
            win.set_on_close(lambda: setattr(self, "running", False) or True)

            def tick():
                if not self.running:
                    app.quit(); return True
                try: item = self.q.get_nowait()
                except Exception: win.post_redraw(); return True
                if item is None: self.running = False; app.quit(); return True
                depth, color = item
                try:
                    np_ = build_pcd(depth, color, self.zs)
                    self.pcd.points = np_.points; self.pcd.colors = np_.colors
                    if self._added: wd.scene.remove_geometry("p")
                    wd.scene.add_geometry("p", self.pcd, self.mat)
                    if not self._added:
                        wd.look_at([0.5, 0.5, 0.0], [0.5, -0.3, 2.5], [0.0, 1.0, 0.0])
                    self._added = True; win.post_redraw()
                except Exception as e:
                    print(f"[O3D] skip: {e}")
                return True

            win.set_on_tick_event(tick)
            app.run()
            self.running = False

    return W()


# ─────────────────────────────────────────────────────────────────────────────
PROFILES = {
    "turbo":   dict(cw=640,  ch=480,  fr=120),
    "fast":    dict(cw=1280, ch=720,  fr=90),
    "default": dict(cw=1280, ch=720,  fr=60),
    "quality": dict(cw=1920, ch=1080, fr=30),
    # FULL FOV: mode sensor 1640x1232 (seluruh sensor IMX219, binned 2x2).
    # Mode lain (1920x1080, 640x480) adalah POTONGAN tengah sensor -> FOV vertikal
    # tidak cukup untuk jendela 18x24 mm. Maks ~40 fps di mode ini.
    "full":    dict(cw=820,  ch=616,  fr=40, raw=(1640, 1232),
                    window=(265, 105, 565, 495)),   # jendela gel di frame 820x616
    # USB: Pi Zero sebagai webcam UVC (YUYV 640x480, ~15 fps). Raspberry Pi atau Ubuntu.
    "usb":     dict(cw=640,  ch=480,  fr=15),
}

# Nilai trackbar yang WAJIB dipakai saat mengambil data no-contact (Table IV).
# Bukan preferensi: masing-masing mengubah angka yang diukur, bukan tampilannya.
MEASURE_SAFE = {
    "center_suppress": 0,   # mask mengalikan flow -> mengecilkan yang diukur
    "rebase hold":     0,   # reset baseline -> menghapus drift yang diukur
    "deadzone x10":    0,   # runtime stage Sec. VII, Table IV tanpa ini
    "ema x100":        0,
    "release x100":    0,
    "tavg":            1,
}


def main():
    args = [a.lower() for a in sys.argv[1:]]
    usb_spec, usb_sensor_w = None, 1280
    for a in sys.argv[1:]:
        if a.lower().startswith("--usb="):
            usb_spec = a.split("=", 1)[1]
        elif a.lower().startswith("--sensor-w="):
            usb_sensor_w = int(a.split("=", 1)[1])
    use_usb = ("usb" in args) or (usb_spec is not None) or (Picamera2 is None)
    if use_usb and Picamera2 is None and "usb" not in args and usb_spec is None:
        print("[INFO] Picamera2 tidak ada (mis. Ubuntu) -> mode USB (Tactile UVC Cam)")
    pk = ("usb" if use_usb else "full" if "full" in args else
          "turbo" if "turbo" in args else ("fast" if "fast" in args else
         ("quality" if "quality" in args else "default")))
    profile = PROFILES[pk]
    use_new_calib = (pk == "full") and ("oldcal" not in args)

    cam_id = 1 if "cam1" in args else 0

    # ── fitur slip: tepi bin histogram & grid sampel ──────────────────
    slip_max_px = 8.0          # --slip-max=N (piksel, tepi atas histogram)
    slip_rows, slip_cols = 7, 9   # --slip-grid=RxC (default 7x9 = 63 titik)
    pattern_type = "unknown"   # --pattern=random|palette|...
    for a in args:
        try:
            if a.startswith("--slip-max="):
                slip_max_px = max(0.5, float(a.split("=", 1)[1]))
            elif a.startswith("--slip-grid="):
                _r, _c = a.split("=", 1)[1].lower().split("x")
                slip_rows, slip_cols = max(2, int(_r)), max(2, int(_c))
            elif a.startswith("--pattern="):
                pattern_type = a.split("=", 1)[1]
            elif a.startswith("--window="):
                v = a.split("=", 1)[1]
                profile = dict(profile, window=None if v in ("off", "none") else
                               tuple(int(t) for t in v.split(",")))
        except ValueError:
            sys.exit(f"[ERROR] argumen '{a}' tidak valid.")
    exp_cap = int(1e6 / profile["fr"])   # exposure tak bisa melebihi 1 frame
    global WINDOW_BOX
    WINDOW_BOX = profile.get("window")
    if WINDOW_BOX is not None:
        print(f"[INFO] masking jendela gel: x {WINDOW_BOX[0]}-{WINDOW_BOX[2]}, "
              f"y {WINDOW_BOX[1]}-{WINDOW_BOX[3]} (ubah dengan --window=x0,y0,x1,y1 "
              f"atau --window=off). Setelah itu atur roi_thresh ~110 lalu tekan 'd'.")

    # deltact_plus versi 30 Sep wajib untuk kalibrasi baru
    if use_new_calib and not deltact_supports_undist_roi():
        sys.exit("[ERROR] deltact_plus.py yang terpasang masih versi LAMA "
                 "(SensorConfig tanpa cx/cy/roi_undist). Pasang deltact_plus.py "
                 "versi 30 Sep dulu, atau jalankan dengan 'oldcal' untuk perbandingan.")

    # config sensor
    if   "fast"    in args and pk == "fast":    cfg = dp.SensorConfig.fast_profile()
    elif "quality" in args:                     cfg = dp.SensorConfig.quality_profile()
    else:
        try:    cfg = dp.SensorConfig.load(CONFIG_FILE)
        except Exception: cfg = dp.SensorConfig()
    if "nodepth" in args:
        cfg.enable_depth = False

    # ── Undistortion dari hasil tuning fov200 ──────────────────────────────
    # Dikalibrasi pada capture 1920x1080. k1,k2 & roi_thresh tidak bergantung
    # resolusi -> dipasang langsung. focal (piksel) bergantung resolusi ROI,
    # jadi diskalakan ke tinggi profil aktif (focal/tinggi dibuat konstan).
    # (Profil 'full' ditimpa FULL_CALIB di bawah, kecuali 'oldcal'.)
    cfg.dist = (-0.60, -0.21, 0.0, 0.0, 0.0)        # k1=-0.60, k2=-0.21
    cfg.roi_thresh = 104
    FOCAL_AT_1080 = 1426.0
    cfg.focal = FOCAL_AT_1080 * (profile["ch"] / 1080.0)
    # sensor_config.json lama bisa membawa cx/cy/roi_undist; hanya dipakai kalau
    # kalibrasi baru aktif.
    for _a in ("cx", "cy", "roi_undist"):
        if hasattr(cfg, _a):
            setattr(cfg, _a, None)

    enable_3d = ("3d" in args) and cfg.enable_depth
    if enable_3d:
        try:
            import open3d  # noqa: F401
        except ImportError:
            print("[WARN] open3d tidak ada -> 3D dimatikan"); enable_3d = False

    if use_usb:
        # ── USB: Pi Zero UVC webcam (Raspberry Pi atau Ubuntu) ──────────
        import run_sensor_fast as rsf               # pembaca UVC + pencarian by-id
        dev = rsf.resolve_usb(usb_spec)

        class _UsbReaderMD(rsf.UsbCameraReader):
            def read_md(self):                      # UVC tidak membawa metadata sensor
                ok, f, seq = self.read()
                return ok, f, seq, {}

        cam = _UsbReaderMD(dev, profile["cw"], profile["ch"], profile["fr"])
        profile = dict(profile, cw=cam.size[0], ch=cam.size[1], device=dev,
                       calib_source="legacy 1080p fov200 (USB, focal by sensor width)")
        cfg.focal = FOCAL_AT_1080 / (usb_sensor_w / float(profile["cw"]))
        print(f"[INFO] USB {dev} -> {profile['cw']}x{profile['ch']} | focal={cfg.focal:.1f}px "
              f"(sensor-w={usb_sensor_w}). Exposure/AWB diatur di Zero (tidak bisa dikunci).")
        picam2 = None
    else:
        # info kamera
        info = Picamera2.global_camera_info()
        print(f"[INFO] {len(info)} kamera terdeteksi; memakai index {cam_id}.")
        if cam_id >= len(info):
            sys.exit(f"[ERROR] kamera index {cam_id} tidak ada.")

        picam2 = open_camera(cam_id, profile["cw"], profile["ch"], profile["fr"],
                             profile.get("raw"))

    # ── focal yang BENAR untuk mode sensor yang dipakai ──────────────────
    # FOCAL_AT_1080 (1426 px) dikalibrasi di mode 1920x1080 = 1 piksel sensor
    # per piksel gambar. Mode lain di-binning dan/atau di-crop, jadi skala yang
    # benar = lebar ScalerCrop (piksel sensor) / lebar gambar, BUKAN tinggi/1080.
    # (Dulu: turbo 640x480 dapat focal 634, padahal seharusnya ~713.)
    try:
        crop = None
        for _ in range(5 if picam2 is not None else 0):
            crop = picam2.capture_metadata().get("ScalerCrop")
            if crop: break
        if crop is None and picam2 is None:
            pass                                    # USB: focal sudah dari lebar sensor
        elif crop:
            native_per_px = float(crop[2]) / float(profile["cw"])
            cfg.focal = FOCAL_AT_1080 / native_per_px
            print(f"[INFO] ScalerCrop={tuple(crop)} -> {native_per_px:.3f} px sensor/px "
                  f"-> focal={cfg.focal:.1f}px")
        else:
            print("[WARN] ScalerCrop tidak tersedia; focal pakai skala tinggi/1080 (lama).")
    except Exception as e:
        print(f"[WARN] gagal membaca ScalerCrop ({e}); focal pakai skala lama.")

    # ── Kalibrasi checkerboard untuk profil full (MODIFIKASI 5) ─────────
    fold_cx = fold_cy = None
    if use_new_calib:
        focal_scalercrop = cfg.focal
        cfg.focal = float(FULL_CALIB["focal"])
        cfg.dist = tuple(FULL_CALIB["dist"])
        cfg.cx = float(FULL_CALIB["cx"])
        cfg.cy = float(FULL_CALIB["cy"])
        cfg.roi_undist = tuple(FULL_CALIB["roi_undist"])
        fold_cx, fold_cy = cfg.cx, cfg.cy
        profile = dict(profile, calib_source=FULL_CALIB["source"])
        if abs(focal_scalercrop - cfg.focal) > 1.0:
            print(f"[WARN] focal dari ScalerCrop ({focal_scalercrop:.1f}px) beda dari "
                  f"focal kalibrasi ({cfg.focal:.1f}px). Mode sensor berubah? "
                  f"Kalibrasi checkerboard hanya berlaku untuk 1640x1232 -> 820x616.")
        print(f"[INFO] kalibrasi checkerboard: focal={cfg.focal}, k1={cfg.dist[0]}, "
              f"k2={cfg.dist[1]}, pusat=({cfg.cx}, {cfg.cy}), roi_undist={cfg.roi_undist}")
    elif pk == "full":
        profile = dict(profile, calib_source="legacy 1080p fov200 (oldcal)")
        print("[WARN] 'oldcal': kalibrasi LAMA (melipat di r~164 px). "
              "Hanya untuk perbandingan; roi_XX.png tidak valid.")

    # ── Guard lipatan undistort (MODIFIKASI 4c) ─────────────────────────
    fold_info = undistort_fold_info(cfg.focal, cfg.dist, profile["cw"], profile["ch"],
                                    WINDOW_BOX, cx=fold_cx, cy=fold_cy)
    if fold_info["r_u_fold"] is None:
        print(f"[INFO] fold guard: polinomial distorsi monoton (tidak melipat); "
              f"pusat {tuple(round(v, 1) for v in fold_info['center_px'])}"
              f"{' (asumsi)' if fold_info['center_assumed'] else ' (fit)'}.")
    else:
        print(f"[INFO] fold guard: undistort valid s/d radius mentah "
              f"{fold_info['r_d_max_px']:.0f}px (r_u fold {fold_info['r_u_fold']:.3f}); "
              f"jendela gel s/d {fold_info['window_max_r_px']:.0f}px.")
        if not fold_info["ok"]:
            print(f"[WARN] fold guard: {100*fold_info['frac_window_beyond']:.1f}% jendela gel "
                  f"DI LUAR radius valid -> undistort MELIPAT (pantulan chart). "
                  f"roi_XX.png tidak valid untuk kalibrasi; pakai raw_XX.png. "
                  f"Perlu fit ulang k1/k2 untuk profil '{pk}'.")

    if picam2 is not None:
        cam = CameraReader(picam2, name=f"cam{cam_id}")
    cam.start()
    sensor = dp.TactileSensor(cfg, profile["cw"], profile["ch"])
    print(f"[INFO] {pk} | {profile['cw']}x{profile['ch']} @ {profile['fr']}fps | depth={'ON' if cfg.enable_depth else 'OFF'} | 3D={'ON' if enable_3d else 'OFF'}")
    print(f"[INFO] undistort: focal={cfg.focal:.1f}px, "
          f"k1={cfg.dist[0]}, k2={cfg.dist[1]}, "
          f"cx={getattr(cfg, 'cx', None)}, cy={getattr(cfg, 'cy', None)}, "
          f"roi_undist={getattr(cfg, 'roi_undist', None)}, roi_thresh={cfg.roi_thresh}")

    live3d = None
    if enable_3d:
        live3d = make_3d_window(z_scale=5.0); live3d.start()

    WIN = "Tactile CALIB [r=base d=ROI a=arrow e=lockAE u=modeUkur m=rekam n=noisefloor s=snap b=burst q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 420)
    TB = "Kontrol"
    cv2.namedWindow(TB, cv2.WINDOW_NORMAL); cv2.resizeWindow(TB, 430, 420)

    def nb(n, v, mx): cv2.createTrackbar(n, TB, int(v), mx, lambda _: None)
    nb("arrow_scale",    5,  25)
    nb("arrow_step",     16, 40)
    nb("deadzone x10",   int(cfg.flow_deadzone * 10),            40)
    nb("ema x100",       int(cfg.flow_ema * 100),                95)
    nb("release x100",   int(cfg.flow_ema_release * 100),        95)
    nb("tavg",           max(cfg.temporal_avg, 1),                8)
    nb("depth_min x100", int(cfg.depth_min * 100),               30)
    nb("depth_open",     cfg.depth_open,                          9)
    nb("roi_thresh",     cfg.roi_thresh,                        200)
    nb("mask margin",    max(cfg.valid_margin, 0),               60)
    nb("rebase hold",    cfg.rebaseline_hold,                    30)
    nb("rebase thr x10", int(cfg.rebaseline_flow_thresh * 10),   40)
    nb("gain x10",       12,  40)
    nb("bright",         30, 150)

    # ─── CENTER SUPPRESSION ───
    nb("center_suppress", 10, 100)  # 0..100, default 10 (10%)
    nb("supp_radius %",   10, 60)   # 10..60%, default 10

    # state
    fps_t, fps, fc = time.time(), 0.0, 0
    draw_arrow = True
    ae_locked = False
    measure_mode = False
    exp_us = 8000      # exposure awal saat dikunci (mikrodetik)
    gain_a = 2.0       # analogue gain awal saat dikunci
    rest_frames = 0
    prev_mgn = cfg.valid_margin
    prev_thr = cfg.roi_thresh
    last_panel = None
    last_seq = None
    frame_md = {}
    depth_pub = DepthPublisher(with_color=True)   # << tulis depth+color ke /dev/shm (murah)

    # ─── BURST FRAME MENTAH ('b') ───
    # burst_phase: None (idle) -> "settle" (buang frame) -> "capture" (simpan)
    burst_phase = None
    burst_dir = None
    burst_seqs = []
    burst_pf = []          # metadata per frame yang DISIMPAN
    settle_log = []        # metadata per frame yang DIBUANG
    settle_timeout = False

    def calib_extra(per_frame):
        return dict(
            per_frame=per_frame,
            roi_thresh_trackbar=int(cv2.getTrackbarPos("roi_thresh", TB)),
            fold_guard=fold_info,
            settle_criteria=dict(skip_min=BURST_SKIP_MIN, skip_max=BURST_SKIP_MAX,
                                 n=SETTLE_N, tol_exp_us=SETTLE_TOL_EXP_US,
                                 tol_dgain=SETTLE_TOL_DGAIN, tol_lvl_rel=SETTLE_TOL_LVL),
            lvl_def="rata-rata BGR/255 di jendela gel (subsampel 4 px) frame mentah")

    # ─── NO-CONTACT PROBE ───
    probe = NoContactProbe(cfg)

    # ─── FITUR SLIP (Hu et al. 2024, arXiv:2303.00935) ───
    slog = SlipLogger("slip_log_1cam.csv", slip_rows, slip_cols,
                      slip_max_px, pattern_type)
    slip_grid = None
    slip_shape = None
    slip_prev_D = None
    slip_prev_E = None
    slip_t_prev = None
    slip_E = float("nan")
    slip_dE = 0.0
    print(f"[INFO] Slip log -> {slog.path} | grid {slip_rows}x{slip_cols} = "
          f"{slog.n_points} titik, {SLIP_BINS} bin 0..{slip_max_px:.1f}px. "
          f"'g' mulai/stop rekam, 'k' ganti label static<->slip.")

    # Cache Gaussian mask (rebuild only jika h/w/radius berubah)
    gaussian_mask = None
    last_supp_params = (None, None, None)

    print("[INFO] Tekan 'r' TANPA kontak untuk baseline bersih.")
    print("[INFO] Untuk eksperimen: tekan 'e' untuk MENGUNCI exposure/gain/AWB (hindari flow palsu).")
    print("[INFO] Gunakan trackbar 'center_suppress' & 'supp_radius %' untuk kontrol Gaussian taper.")
    print("[INFO] Untuk data Table IV: tekan 'u' (mode ukur), tunggu STABLE, lalu 'm' untuk merekam.")
    print("[INFO] Noise floor: tekan 'u', tunggu ~5 detik, lalu 'n' (tanpa kontak).")
    print(f"[INFO] Frame mentah: 's' = 1 frame, 'b' = burst {BURST_N} frame "
          f"(buang >= {BURST_SKIP_MIN} frame awal sampai stabil).")

    try:
        while True:
            if live3d is not None and not live3d.running:
                live3d = None; print("[INFO] Open3D ditutup.")

            a_scale = max(1, cv2.getTrackbarPos("arrow_scale", TB))
            a_step  = max(6, cv2.getTrackbarPos("arrow_step",  TB))
            thr     = cv2.getTrackbarPos("roi_thresh",  TB)
            mgn     = cv2.getTrackbarPos("mask margin", TB)
            gain    = max(1, cv2.getTrackbarPos("gain x10", TB)) / 10.0
            bright  = cv2.getTrackbarPos("bright", TB)

            # ─── CENTER SUPPRESSION ───
            supp_strength = cv2.getTrackbarPos("center_suppress", TB)
            supp_radius_pct = cv2.getTrackbarPos("supp_radius %", TB)

            cfg.flow_deadzone    = cv2.getTrackbarPos("deadzone x10", TB) / 10.0
            cfg.flow_ema         = cv2.getTrackbarPos("ema x100",     TB) / 100.0
            cfg.flow_ema_release = cv2.getTrackbarPos("release x100", TB) / 100.0
            cfg.temporal_avg     = max(1, cv2.getTrackbarPos("tavg",  TB))
            cfg.depth_min        = cv2.getTrackbarPos("depth_min x100", TB) / 100.0
            cfg.depth_open       = cv2.getTrackbarPos("depth_open", TB)
            cfg.auto_rebaseline  = False
            cfg.rebaseline_hold        = cv2.getTrackbarPos("rebase hold", TB)
            cfg.rebaseline_flow_thresh = cv2.getTrackbarPos("rebase thr x10", TB) / 10.0
            reb_hold = cfg.rebaseline_hold
            reb_thr  = cfg.rebaseline_flow_thresh

            if mgn != prev_mgn:
                cfg.valid_margin = mgn; sensor.rebuild_mask(); prev_mgn = mgn
            if thr != prev_thr:
                cfg.roi_thresh = thr; sensor.redetect_roi(); prev_thr = thr

            ok, frame, seq, frame_md = cam.read_md()
            if not ok or seq == last_seq:
                # frame sama dengan sebelumnya: jangan diproses ulang (flow 0 palsu).
                # waitKey sengaja TIDAK dipanggil di sini supaya tombol tidak hilang.
                time.sleep(0.002); continue
            last_seq = seq

            res = sensor.process(apply_window(frame, WINDOW_BOX))

            # ─── BURST: settle (buang) lalu simpan frame mentah (tidak memblok) ───
            if burst_phase is not None:
                pf = dict(seq=int(seq), lvl=window_level(frame, WINDOW_BOX),
                          **md_subset(frame_md))
                if burst_phase == "settle":
                    settle_log.append(pf)
                    is_ok = len(settle_log) >= BURST_SKIP_MIN and settled(settle_log)
                    if is_ok or len(settle_log) >= BURST_SKIP_MAX:
                        settle_timeout = not is_ok
                        burst_phase = "capture"
                        if settle_timeout:
                            print(f"[SNAP] PERINGATAN: belum stabil setelah {len(settle_log)} "
                                  f"frame; burst tetap diambil (settle_timeout=true).")
                        else:
                            print(f"[SNAP] stabil setelah {len(settle_log)} frame dibuang; "
                                  f"mulai simpan {BURST_N} frame.")
                else:
                    i = len(burst_seqs)
                    save_raw_bundle(burst_dir, i, frame, res)
                    burst_seqs.append(int(seq)); burst_pf.append(pf)
                    if len(burst_seqs) >= BURST_N:
                        extra = calib_extra(burst_pf)
                        extra.update(n_skipped=len(settle_log), settle_log=settle_log,
                                     settle_timeout=settle_timeout)
                        write_meta(burst_dir, cam, cfg, pk, profile, ae_locked,
                                   measure_mode, len(burst_seqs), burst_seqs,
                                   note="burst", extra=extra)
                        lv = [p["lvl"] for p in burst_pf]
                        print(f"[SNAP] burst selesai: {len(burst_seqs)} frame -> {burst_dir}/ "
                              f"(dibuang {len(settle_log)}; drift lvl "
                              f"{100*(max(lv)-min(lv))/max(np.mean(lv),1e-6):.2f}%)")
                        burst_phase = None
            npanels = 3 if cfg.enable_depth else 2
            if res is None:
                panel = np.zeros((cfg.work_h, cfg.work_w * npanels, 3), np.uint8)
                label_bottom(panel, "ROI? atur roi_thresh", (0, 0, 255))
            else:
                flow = res["flow"]; roi = res["roi"]

                # ─── NO-CONTACT PROBE ───────────────────────────────────────
                # Dipanggil di sini, SEBELUM center suppression menyentuh flow.
                # Memakai `roi` mentah, bukan `roi_disp`: trackbar gain/bright
                # hanya untuk tampilan, dan akan mengubah low_f serta lambda_min
                # kalau ikut terukur.
                probe.update(flow, roi, supp_strength, reb_hold)

                # ─── FITUR SLIP ────────────────────────────────────────
                # Juga di sini, SEBELUM center suppression: mask
                # mengalikan flow, jadi entropi & Vx/Vy ikut mengecil
                # kalau dihitung sesudahnya.
                if slip_shape != flow.shape[:2]:
                    slip_grid = make_slip_grid(flow.shape[0], flow.shape[1],
                                               slip_rows, slip_cols)
                    slip_shape = flow.shape[:2]
                    slip_prev_D = slip_prev_E = slip_t_prev = None
                t_now = time.monotonic()
                slip_dt = (t_now - slip_t_prev) if slip_t_prev else 0.0
                (slip_E, slip_dE, slip_vx, slip_vy, slip_mg, slip_p90,
                 slip_D) = slip_features(flow, slip_grid, slog.edges,
                                         slip_prev_D, slip_prev_E, slip_dt)
                slip_prev_D, slip_prev_E, slip_t_prev = slip_D, slip_E, t_now
                if slog.recording:
                    _d = res.get("depth") if cfg.enable_depth else None
                    _cf = (float((_d > cfg.depth_min).mean())
                           if _d is not None else float("nan"))
                    slog.write(t_now, f"cam{cam_id}", slip_E, slip_dE,
                               slip_vx, slip_vy, slip_mg, slip_p90,
                               float(np.linalg.norm(flow, axis=2).max()),
                               _cf, slip_dt)

                flow_mag = float(np.linalg.norm(flow, axis=2).max())
                roi_lvl = float(roi.mean()) / 255.0   # kecerahan NYATA, bukan tampilan
                roi_disp = cv2.convertScaleAbs(roi, alpha=gain, beta=bright)
                p1 = roi_disp.copy()
                if draw_arrow:
                    quiver_fast(p1, flow, a_step, a_scale)
                label_bottom(p1, f"Speckle |flow|={flow_mag:.1f}")
                p2 = dp.flow_to_color(flow)
                label_bottom(p2, "Flow")
                panels = [p1, p2]
                depth = res.get("depth") if cfg.enable_depth else None

                # ─── Rebuild Gaussian mask jika parameter berubah ───
                supp_params = (cfg.work_h, cfg.work_w, supp_radius_pct)
                if supp_params != last_supp_params:
                    gaussian_mask = make_gaussian_center_mask(
                        cfg.work_h, cfg.work_w,
                        radius_pct=float(supp_radius_pct),
                        sigma_scale=1.2  # kontrol kecuraman (> 1 = lebih tajam)
                    )
                    last_supp_params = supp_params

                # ─── Terapkan center suppression ───
                if supp_strength > 0 and gaussian_mask is not None:
                    flow = apply_center_suppression(flow, gaussian_mask, strength=supp_strength)
                    if depth is not None:
                        depth = apply_center_suppression(depth, gaussian_mask, strength=supp_strength)

                if cfg.enable_depth:
                    if depth is not None:
                        p3 = dp.depth_to_color(depth)
                        label_bottom(p3, f"Depth {depth.max():.3f}", (255, 255, 255))
                    else:
                        p3 = np.zeros((cfg.work_h, cfg.work_w, 3), np.uint8)
                    panels.append(p3)
                panel = np.hstack(panels)

                # rebaseline otomatis (saat idle)
                # Dilewati saat probe merekam: reset baseline akan menghapus
                # drift yang justru sedang diukur.
                if reb_hold > 0 and not probe.recording and not slog.recording:
                    if flow_mag < reb_thr:
                        rest_frames += 1
                        if rest_frames >= reb_hold:
                            sensor.reset(); rest_frames = 0
                    else:
                        rest_frames = 0

                if depth is not None:
                    depth_pub.publish(depth, roi_disp)      # << viewer 3D terpisah (view_3d_live.py)
                    if live3d is not None:
                        live3d.push(depth, roi_disp)

            # FPS + status (di bawah panel dengan background bar)
            fc += 1
            if fc >= 20:
                now = time.time(); fps = fc / (now - fps_t); fps_t, fc = now, 0
            status = f"{fps:.1f}FPS"
            if res is not None:
                warn = ("  GELAP" if roi_lvl < 0.18 else
                        "  TERANG" if roi_lvl > 0.85 else "")
                status += f" | lvl={roi_lvl:.2f}{warn}"
            if ae_locked:
                status += f" | LOCK exp={exp_us}us gain={gain_a:.1f}"
            if measure_mode:
                status += " | UKUR"
            if burst_phase == "settle":
                status += f" | BURST settle {len(settle_log)}"
            elif burst_phase == "capture":
                status += f" | BURST {len(burst_seqs)}/{BURST_N}"
            if supp_strength > 0:
                status += f" | Supp={supp_strength}% r={supp_radius_pct}%"
            if res is not None and not np.isnan(slip_E):
                status += f" | E={slip_E:.2f} dE/dt={slip_dE:+.1f}"
            if slog.recording:
                status += (f" | REC t{slog.trial} "
                           f"{'SLIP' if slog.label else 'static'}")

            # Buat background bar di bawah untuk status (agar mudah dibaca)
            h, w = panel.shape[:2]
            cv2.rectangle(panel, (0, h - 28), (w, h), (0, 0, 0), -1)
            cv2.putText(panel, status, (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # ─── baris status probe (di atas baris FPS) ───
            panel = probe.overlay(panel)

            cv2.imshow(WIN, panel); last_panel = panel

            # ── keyboard ──
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k == ord('r'):
                sensor.reset(); rest_frames = 0; print("[INFO] baseline reset")
            elif k == ord('d'):
                sensor.redetect_roi(); print("[INFO] re-deteksi ROI")
            elif k == ord('a'):
                draw_arrow = not draw_arrow; print(f"[INFO] arrow: {'ON' if draw_arrow else 'OFF'}")
            elif k == ord('c'):
                cfg.save(CONFIG_FILE); print(f"[INFO] config saved -> {CONFIG_FILE}")
            elif k in (ord('e'), ord('['), ord(']'), ord('-'), ord('=')) and use_usb:
                print("[INFO] USB: exposure/gain/AWB diatur di Pi Zero, tidak bisa dikunci "
                      "dari sini (tombol diabaikan).")
            elif k in (ord('u'), ord('M')) and use_usb:
                for name, val in MEASURE_SAFE.items():
                    cv2.setTrackbarPos(name, TB, val)
                print("[INFO] USB: trackbar dibersihkan (suppression/rebaseline/deadzone/EMA/tavg). "
                      "Kamera TIDAK terkunci (UVC) -> data Table IV dari USB ditandai "
                      "camera_lock=false di meta.json.")
            elif k == ord('e'):
                ae_locked = not ae_locked
                if ae_locked:
                    # kunci di nilai yang SUDAH dipilih AE, bukan di angka
                    # hard-coded, supaya kecerahan tidak melompat saat dikunci
                    md = cam.get_metadata()
                    exp_us = int(md.get("ExposureTime") or exp_us)
                    gain_a = float(md.get("AnalogueGain") or gain_a)
                    cam.set_controls(AeEnable=False, AwbEnable=False,
                                     ExposureTime=int(exp_us), AnalogueGain=float(gain_a))
                    print(f"[INFO] AE/AWB DIKUNCI: exp={exp_us}us gain={gain_a:.2f}")
                else:
                    cam.set_controls(AeEnable=True, AwbEnable=True)
                    if measure_mode:
                        print("[WARN] mode ukur DIBATALKAN: AE/AWB kembali auto.")
                        measure_mode = False
                    print("[INFO] AE/AWB auto (tidak terkunci)")
            elif k in (ord('['), ord(']')) and ae_locked:
                # langkah 10% -> sama terasa di 2000us maupun di 30000us
                step = max(200, int(0.10 * exp_us))
                exp_us = int(np.clip(exp_us + (step if k == ord(']') else -step),
                                     100, exp_cap))
                cam.set_controls(ExposureTime=int(exp_us))
                note = "  (mentok batas fps)" if exp_us >= 0.95 * exp_cap else ""
                if measure_mode:
                    note += "  [PERINGATAN: bukan titik operasi ukur lagi]"
                print(f"[INFO] exposure={exp_us}us{note}")
            elif k in (ord('-'), ord('=')) and ae_locked:
                gain_a = max(1.0, gain_a + (0.5 if k == ord('=') else -0.5))
                cam.set_controls(AnalogueGain=float(gain_a))
                note = "  [PERINGATAN: bukan titik operasi ukur lagi]" if measure_mode else ""
                print(f"[INFO] gain={gain_a:.1f}  (gain naik = sigma_n naik = "
                      f"lantai derau yang diukur ikut naik){note}")
            elif k == ord('g'):
                if not slog.recording:
                    slip_prev_D = slip_prev_E = slip_t_prev = None
                    trial = slog.start(time.monotonic())
                    print(f"[SLIP] REKAM trial {trial}, label=static. "
                          f"'k' = tandai slip, 'g' = stop.")
                    if reb_hold > 0:
                        print(f"[WARN] 'rebase hold'={reb_hold} masih aktif. "
                              f"Dilewati selama merekam, tapi lebih aman "
                              f"set trackbar-nya ke 0.")
                    if supp_strength > 0:
                        print(f"[INFO] center_suppress={supp_strength}% aktif, "
                              f"tapi fitur slip dihitung SEBELUM suppression "
                              f"-> tidak terpengaruh.")
                else:
                    print(f"[SLIP] STOP trial {slog.stop()} -> {slog.path}")
            elif k == ord('k'):
                _lab = slog.toggle_label()
                print(f"[SLIP] label -> {'SLIP' if _lab else 'static'}"
                      + ("" if slog.recording else "  (belum merekam; 'g')"))
            elif k == ord('s') and last_panel is not None:
                ts = time.strftime("%Y%m%d_%H%M%S"); fn = f"snap_fast_{ts}.png"
                cv2.imwrite(fn, last_panel)                         # tampilan (seperti dulu)
                d = f"raw_{ts}"
                has_roi = save_raw_bundle(d, 0, frame, res)         # frame MENTAH + ROI mentah
                pf1 = dict(seq=int(last_seq), lvl=window_level(frame, WINDOW_BOX),
                           **md_subset(frame_md))
                write_meta(d, cam, cfg, pk, profile, ae_locked, measure_mode, 1,
                           [int(last_seq)], note="single snap", extra=calib_extra([pf1]))
                print(f"[INFO] snap tampilan: {fn} | mentah: {d}/raw_00.png"
                      f"{' + roi_00.png' if has_roi else ' (ROI tidak ada)'}"
                      f" ({frame.shape[1]}x{frame.shape[0]})")
            elif k in (ord('b'), ord('S')):          # b = burst (tanpa Shift)
                if burst_phase is not None:
                    print("[SNAP] burst masih berjalan.")
                else:
                    burst_dir = f"raw_{time.strftime('%Y%m%d_%H%M%S')}"
                    burst_phase = "settle"
                    burst_seqs, burst_pf, settle_log = [], [], []
                    settle_timeout = False
                    print(f"[SNAP] burst {BURST_N} frame -> {burst_dir}/ (jangan sentuh sensor); "
                          f"menunggu stabil (buang >= {BURST_SKIP_MIN} frame)...")
                    if not measure_mode:
                        print("[SNAP] PERINGATAN: mode ukur belum aktif ('u').")

            # ─── MODE UKUR: nolkan semua yang mengganggu + kunci kamera ───
            elif k in (ord('u'), ord('M')):          # u = mode ukur (tanpa Shift)
                for name, val in MEASURE_SAFE.items():
                    cv2.setTrackbarPos(name, TB, val)
                exp_us, gain_a = MEASURE_EXP_US, MEASURE_GAIN
                cam.set_controls(AeEnable=False, AwbEnable=False,
                                 ExposureTime=int(exp_us), AnalogueGain=float(gain_a),
                                 ColourGains=MEASURE_COLOUR_GAINS,
                                 NoiseReductionMode=0,     # off: NR = filter spasial
                                 Sharpness=0.0)            # off: sharpening = filter spasial
                ae_locked = True
                measure_mode = True
                print(f"[INFO] MODE UKUR aktif: suppression/rebaseline/runtime OFF; "
                      f"kamera dikunci exp={exp_us}us gain={gain_a} "
                      f"colour={MEASURE_COLOUR_GAINS}, NR & sharpening OFF.")
                print("[INFO] Tunggu status STABLE, lalu 'm' untuk merekam, atau 'n' untuk noise floor.")

            # ─── rekam satu decal ───
            elif k == ord('m'):
                probe.toggle_record(ae_locked, supp_strength, reb_hold)

            # ─── noise floor ───
            elif k == ord('n'):
                if probe.recording:
                    print("[NF] probe sedang merekam; batalkan dulu ('m').")
                else:
                    run_noise_floor(cam, cfg, profile["cw"], profile["ch"],
                                    window_box=WINDOW_BOX)

    finally:
        cam.stop()
        if live3d is not None:
            live3d.stop()
        depth_pub.close()        # << tutup shared memory
        slog.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
