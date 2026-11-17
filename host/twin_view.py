"""DIGITAL TWIN view of the 3-finger tactile hand: one state dict -> one picture.

The same state dict is (a) drawn inside the servo_grasp GUI, (b) sent as JSON over UDP with
--twin-udp=HOST:PORT (every loop, ~40 Hz) and (c) drawn by twin_viewer.py on any other PC. A 3-D twin
(Isaac / Unity / RViz / Blender) only has to read the same JSON - the schema is below and stays stable
(new keys may be added, "v" is raised when a key changes its meaning).

STATE SCHEMA (v 1) - all angles in degrees, loads in N*m, depths in mm, shear in px
  v, t (unix s), mode (IDLE GRASP HOLD OPEN ACTIVE STOPPED), msg, depth_ref, depth_max, grip_load
  joints   [6]  {id, deg (actual, None = no feedback), sent (commanded), load}
  fingers  [3]  {id, rot_id, cam, phase, contact (bool), fresh (bool), fps,
                 flex_deg (actual flexion servo angle), open_deg, closed_deg (taught poses),
                 closed (0 = open pose .. 1 = closed pose, may exceed 1 when advancing),
                 rot_deg (rotation joint, relative to its open pose; + = counter-clockwise seen from above),
                 depth (gel depth), cx, cy (contact centre on the pad, -1 .. 1), area (0 .. 1 of the pad),
                 force, shear_x, shear_y, twist (rad), slip (bool), load (flexion servo),
                 offpad (bool: presses with the finger, not the gel pad), state (short text),
                 grid: the gel depth MAP, or null when the sensor sends none: {"n": [rows, cols],
                       "s": mm per count, "d": base64 of uint8 rows*cols row-major}; decode_grid() reads it.
                       Its columns / rows are the same axes as cx / cy (from the sensor, tactile_ros.py
                       --udp-grid=24; without it the twin draws a bump estimated from cx, cy, area, depth)}
  wrist    {} or the WristTorsion state (valid, active, tau_px, dir_name, rot_slip, coh, ...)
  refine   {} or {phase, finger, step, msg}
  layout   [3]  angle of each finger around the palm, deg, counter-clockwise, 0 = right (--twin-layout)
  load     {} or the load-grip state {loaded, px (latched load), live, g (grams if calibrated), need (grip
           fraction the load asks), rho (grip per px of load), floor (slip memory), ...}
  pad_map  "" or tokens "swap", "flipx", "flipy": how the sensor image lies on the finger in the 3-D hand
           (default: image columns across the finger, image row 0 at the fingertip) (--twin-pad)
"""
import base64
import math

import cv2
import numpy as np

BG = (34, 28, 26)
PANEL = (46, 40, 37)
GRID = (74, 66, 62)
TXT = (225, 225, 225)
DIM = (150, 150, 150)
GREEN = (110, 205, 110)
BLUE = (235, 165, 70)
AMBER = (60, 190, 240)
RED = (70, 70, 235)
VIOLET = (230, 120, 220)
FCOL = [(240, 180, 90), (120, 215, 130), (110, 170, 245)]     # one colour per finger


def _put(img, s, x, y, c=TXT, sc=0.45, th=1):
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, sc, c, th, cv2.LINE_AA)


def heat(v):
    """0 .. 1 -> BGR, dark blue -> cyan -> green -> yellow -> red (touch pressure)."""
    v = float(np.clip(v, 0.0, 1.0))
    stops = [(0.0, (90, 40, 30)), (0.25, (200, 140, 30)), (0.5, (90, 200, 90)), (0.75, (40, 210, 240)),
             (1.0, (50, 60, 235))]
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            f = (v - a) / (b - a)
            return tuple(int(ca[i] + f * (cb[i] - ca[i])) for i in range(3))
    return stops[-1][1]


def _arrow(img, p, q, col, th=2):
    if math.hypot(q[0] - p[0], q[1] - p[1]) < 3:
        return
    cv2.arrowedLine(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), col, th, cv2.LINE_AA, tipLength=0.3)


def _flex_angle(f):
    """Drawn flexion: 12 deg (open, upright) .. 85 deg (closed pose), from the closed fraction."""
    return math.radians(12.0 + 73.0 * float(np.clip(f.get("closed") or 0.0, -0.1, 1.25)))


def tip_xy(f, th_deg, c, R, L):
    """Top view: fingertip position. The base sits on the palm rim at th; the finger points to the palm
    centre, swung by its rotation joint; the more it flexes the further its tip reaches inwards."""
    th = math.radians(th_deg)
    base = (c[0] + R * math.cos(th), c[1] - R * math.sin(th))
    d = th + math.pi + math.radians(f.get("rot_deg") or 0.0)
    reach = L * math.sin(_flex_angle(f))
    return base, (base[0] + reach * math.cos(d), base[1] - reach * math.sin(d)), d


def draw_top(img, S, x0, y0, w, h):
    """Top view of the hand: palm, three fingers (flexion + rotation), pads coloured by touch, the
    object estimated from the touching pads, shear arrows."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    _put(img, "TOP VIEW  (flexion + rotation + touch)", x0 + 10, y0 + 18, DIM, 0.42)
    c = (x0 + w / 2.0, y0 + h / 2.0 + 10)
    R = 0.33 * min(w, h)
    L = 0.80 * R
    cv2.circle(img, (int(c[0]), int(c[1])), int(R), GRID, 1, cv2.LINE_AA)
    cv2.circle(img, (int(c[0]), int(c[1])), 3, GRID, -1)
    fingers = S.get("fingers") or []
    layout = S.get("layout") or [90, 210, 330]
    dmax = max(1e-6, S.get("depth_max") or 0.6)
    tips = []
    for k, f in enumerate(fingers):
        col = FCOL[k % 3]
        th = layout[k % len(layout)]
        base, tip, d = tip_xy(f, th, c, R, L)
        # open-pose ghost
        g0 = dict(f, closed=0.0, rot_deg=0.0)
        _, t0, _ = tip_xy(g0, th, c, R, L)
        cv2.line(img, (int(base[0]), int(base[1])), (int(t0[0]), int(t0[1])), GRID, 1, cv2.LINE_AA)
        ok = f.get("flex_deg") is not None
        lc = col if ok else RED
        cv2.line(img, (int(base[0]), int(base[1])), (int(tip[0]), int(tip[1])), lc, 5, cv2.LINE_AA)
        cv2.circle(img, (int(base[0]), int(base[1])), 7, lc, -1, cv2.LINE_AA)
        # the gel pad: a bar across the tip, colour = depth
        n = (-math.sin(d), -math.cos(d))                      # across the finger (image coords)
        hw = 0.16 * R
        a = (tip[0] - hw * n[0], tip[1] - hw * n[1])
        b = (tip[0] + hw * n[0], tip[1] + hw * n[1])
        pc = heat((f.get("depth") or 0.0) / dmax) if f.get("contact") else (95, 90, 88)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), pc, 9, cv2.LINE_AA)
        if f.get("slip"):
            cv2.circle(img, (int(tip[0]), int(tip[1])), int(hw + 6), RED, 2, cv2.LINE_AA)
        if f.get("offpad"):
            cv2.circle(img, (int(tip[0]), int(tip[1])), int(hw + 10), AMBER, 1, cv2.LINE_AA)
        if f.get("contact"):
            tips.append(tip)
            # shear across the pad (x of the sensor) drawn along the pad bar
            sx = float(f.get("shear_x") or 0.0)
            sx = float(np.clip(sx, -6.0, 6.0))
            _arrow(img, tip, (tip[0] + 4.0 * sx * n[0], tip[1] + 4.0 * sx * n[1]), VIOLET, 2)
        # labels at the base, outside the palm
        th_r = math.radians(th)
        lx = c[0] + (R + 14) * math.cos(th_r)
        ly = c[1] - (R + 14) * math.sin(th_r)
        lx = lx - 108 if math.cos(th_r) < -0.3 else lx + 14 if abs(math.cos(th_r)) <= 0.3 else lx + 4
        ly = ly - 4 if math.sin(th_r) > 0.3 else ly + 10
        lx = float(np.clip(lx, x0 + 4, x0 + w - 118))
        ly = float(np.clip(ly, y0 + 34, y0 + h - 34))
        fd = f.get("flex_deg")
        _put(img, f"ID{f.get('id')} " + (f"{fd:+.1f} deg" if fd is not None else "ERR"), lx, ly, lc, 0.45)
        _put(img, f"ID{f.get('rot_id')} rot {f.get('rot_deg') or 0.0:+.1f}", lx, ly + 15, DIM, 0.4)
    # the object as the touching pads see it
    if len(tips) >= 2:
        oc = np.mean(np.array(tips), axis=0)
        orad = float(np.mean([math.hypot(t[0] - oc[0], t[1] - oc[1]) for t in tips]))
        cv2.circle(img, (int(oc[0]), int(oc[1])), max(4, int(orad - 4)), (120, 120, 120), 1, cv2.LINE_AA)
        cv2.circle(img, (int(oc[0]), int(oc[1])), 2, (160, 160, 160), -1)
        off = math.hypot(oc[0] - c[0], oc[1] - c[1]) / R
        _put(img, f"object  {len(tips)} pad(s)  off-centre {100 * off:.0f} %", x0 + 10, y0 + h - 10, DIM, 0.4)
    else:
        _put(img, "object  not held" if not tips else "object  1 pad", x0 + 10, y0 + h - 10, DIM, 0.4)
    # wrist torsion: an arc around the palm, length ~ torsion
    Wd = S.get("wrist") or {}
    if Wd.get("valid") and Wd.get("active"):
        full = 3.0
        sweep = float(np.clip(Wd.get("tau_px", 0.0) / full, -1, 1)) * 150.0
        col = RED if Wd.get("rot_slip") else AMBER
        cv2.ellipse(img, (int(c[0]), int(c[1])), (int(R + 6), int(R + 6)), 0, -90, -90 - sweep, col, 3, cv2.LINE_AA)
        _put(img, f"wrist {Wd.get('tau_px', 0.0):+.2f} px {Wd.get('dir_name', '')}"
             + ("  ROT-SLIP" if Wd.get("rot_slip") else ""), x0 + w - 190, y0 + 18, col, 0.42)


# ====================================================================== 3-D
_LUT = None


def heat_lut():
    """256 x 3 BGR lookup table of heat() (vectorised colouring of maps / meshes)."""
    global _LUT
    if _LUT is None:
        _LUT = np.array([heat(i / 255.0) for i in range(256)], np.uint8)
    return _LUT


def decode_grid(gd):
    """The 'grid' of a finger (see the schema) -> float32 rows x cols in mm, or None."""
    try:
        if not gd:
            return None
        rows, cols = int(gd["n"][0]), int(gd["n"][1])
        q = np.frombuffer(base64.b64decode(gd["d"]), np.uint8)
        if q.size != rows * cols:
            return None
        return q.reshape(rows, cols).astype(np.float32) * float(gd["s"])
    except Exception:
        return None


def touch_grid(f, n=24):
    """(depth map in mm, measured?) of one finger: the sensor's grid, or - without one - a bump estimated
    from the contact centre, area and depth (marked 'est.' in the views)."""
    g = decode_grid(f.get("grid"))
    if g is not None:
        return g, True
    if not f.get("contact") or (f.get("depth") or 0.0) <= 0.0:
        return np.zeros((n, n), np.float32), False
    yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    a = float(np.clip(f.get("area") or 0.05, 0.01, 1.0))
    sig = max(0.12, math.sqrt(4.0 * a / math.pi) / 1.6)
    cx, cy = float(np.clip(f.get("cx") or 0.0, -1, 1)), float(np.clip(f.get("cy") or 0.0, -1, 1))
    g = float(f.get("depth") or 0.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig))
    return g.astype(np.float32), False


def resample(g, rows, cols):
    if g.shape == (rows, cols):
        return g
    return cv2.resize(g, (cols, rows), interpolation=cv2.INTER_AREA if g.shape[0] > rows else cv2.INTER_LINEAR)


class Cam3:
    """Orthographic camera looking at the origin from azimuth yaw / elevation pitch (deg).
    World: x right, y away, z up. depth(P) grows towards the viewer."""

    def __init__(self, yaw, pitch, scale, cx, cy):
        y, p = math.radians(yaw), math.radians(pitch)
        self.c = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p)])
        self.r = np.array([-math.sin(y), math.cos(y), 0.0])
        self.u = np.cross(self.c, self.r)
        self.s, self.cx, self.cy = scale, cx, cy
        L = 0.75 * self.c + 0.55 * self.u - 0.35 * self.r
        self.L = L / np.linalg.norm(L)

    def proj(self, P):
        P = np.asarray(P, float)
        return np.stack([self.cx + self.s * (P @ self.r), self.cy - self.s * (P @ self.u)], -1), P @ self.c


def mesh_polys(cam, P, col, out, shade=True, mask=None):
    """Quads of a (R x C x 3) surface -> painter list entries (depth, 'poly', pts2, BGR).
    mask (R-1 x C-1 bool): only these quads (the flat rest is drawn as one polygon by the caller)."""
    R, C = P.shape[:2]
    a, b = P[:-1, :-1], P[:-1, 1:]
    c, d = P[1:, 1:], P[1:, :-1]
    xy, dep = cam.proj(np.stack([a, b, c, d], 2))          # (R-1, C-1, 4, 2), (R-1, C-1, 4)
    zc = dep.mean(-1)
    if shade:
        nrm = np.cross(c - a, d - b)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
        facing = np.sign(nrm @ cam.c)
        nrm *= np.where(facing == 0, 1.0, facing)[..., None]
        lam = 0.38 + 0.62 * np.clip(nrm @ cam.L, 0.0, 1.0)
        col = np.clip(col.astype(np.float32) * lam[..., None], 0, 255)
    col = col.astype(np.uint8)
    pts = np.round(xy).astype(np.int32)
    ii, jj = np.nonzero(mask) if mask is not None else np.indices((R - 1, C - 1)).reshape(2, -1)
    zl, cl = zc[ii, jj].tolist(), col[ii, jj].tolist()
    for n_, (i, j) in enumerate(zip(ii.tolist(), jj.tolist())):
        out.append((zl[n_], 0, pts[i, j], tuple(cl[n_])))


def _tex_quad(img, q2, tex):
    """Draw texture tex (h x w x 3) onto the projected parallelogram q2 = [TL, TR, BR, BL] (orthographic
    projection keeps parallelograms, so one affine warp is exact)."""
    q2 = np.asarray(q2, np.float32)
    x0, y0 = np.floor(q2.min(0)).astype(int) - 1
    x1, y1 = np.ceil(q2.max(0)).astype(int) + 2
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(img.shape[1], x1), min(img.shape[0], y1)
    if X1 <= X0 or Y1 <= Y0:
        return
    th, tw = tex.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [0, th]])
    dst = np.float32([q2[0] - (X0, Y0), q2[1] - (X0, Y0), q2[3] - (X0, Y0)])
    M = cv2.getAffineTransform(src, dst)
    W_, H_ = X1 - X0, Y1 - Y0
    warped = cv2.warpAffine(tex, M, (W_, H_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(np.full((th, tw), 255, np.uint8), M, (W_, H_), flags=cv2.INTER_NEAREST)
    roi = img[Y0:Y1, X0:X1]
    roi[mask > 0] = warped[mask > 0]
    cv2.polylines(img, [np.round(q2).astype(np.int32)], True, (150, 150, 150), 1, cv2.LINE_AA)


def paint(img, items):
    """Draw painter-sorted primitives: (depth, kind, data, colour[, extra]). kind 0 = filled polygon,
    1 = line (extra = thickness), 2 = filled circle (extra = (radius, alpha)), 3 = arrow."""
    for it in sorted(items, key=lambda e: e[0]):
        kind = it[1]
        if kind == 0:
            cv2.fillConvexPoly(img, it[2], it[3], cv2.LINE_8)
        elif kind == 1:
            p, q = it[2]
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), it[3], int(it[4]), cv2.LINE_AA)
        elif kind == 2:
            (cx, cy), (r, alpha) = it[2], it[4]
            if alpha >= 1.0:
                cv2.circle(img, (int(cx), int(cy)), int(r), it[3], -1, cv2.LINE_AA)
            else:
                x0, y0 = max(0, int(cx - r - 2)), max(0, int(cy - r - 2))
                x1, y1 = min(img.shape[1], int(cx + r + 3)), min(img.shape[0], int(cy + r + 3))
                if x1 > x0 and y1 > y0:
                    roi = img[y0:y1, x0:x1]
                    ov = roi.copy()
                    cv2.circle(ov, (int(cx) - x0, int(cy) - y0), int(r), it[3], -1, cv2.LINE_AA)
                    cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)
                cv2.circle(img, (int(cx), int(cy)), int(r), (150, 150, 150), 1, cv2.LINE_AA)
        elif kind == 3:
            p, q = it[2]
            _arrow(img, p, q, it[3], int(it[4]))
        elif kind == 5:
            _tex_quad(img, it[2], it[3])
        elif kind == 6:                                  # fingertip block, then its gel map on top
            b2, bc, q2, tex = it[2]
            cv2.fillConvexPoly(img, b2, bc, cv2.LINE_AA)
            _tex_quad(img, q2, tex)


def _flex3d(f):
    """3-D flexion of the finger from vertical: -25 deg (open pose, tilted outwards) .. +35 deg (closed
    pose, tilted inwards), from the closed fraction."""
    return math.radians(-25.0 + 60.0 * float(np.clip(f.get("closed") or 0.0, -0.1, 1.25)))


def finger_frame(f, th_deg, L=1.0):
    """3-D finger: base on the palm rim at th (palm radius 1, palm top at z = 0.12), swung by its rotation
    joint, flexed inwards. Returns base, tip, f (along the finger), n (pad normal, towards the object),
    u (across the finger)."""
    th = math.radians(th_deg)
    base = np.array([0.82 * math.cos(th), 0.82 * math.sin(th), 0.12])
    a = th + math.pi + math.radians(f.get("rot_deg") or 0.0)
    d = np.array([math.cos(a), math.sin(a), 0.0])
    phi = _flex3d(f)
    fa = math.sin(phi) * d + math.cos(phi) * np.array([0.0, 0.0, 1.0])
    n = math.cos(phi) * d - math.sin(phi) * np.array([0.0, 0.0, 1.0])
    u = np.cross(d, np.array([0.0, 0.0, 1.0]))
    return base, base + L * fa, fa, n, u


def pad_texture(g, f, S, measured, px=72):
    """The gel map as a small BGR image (the texture of the pad in the 3-D hand)."""
    dmax = max(1e-6, S.get("depth_max") or 0.6)
    rows, cols = g.shape
    hh = max(8, int(round(px * rows / max(rows, cols))))
    ww = max(8, int(round(px * cols / max(rows, cols))))
    t = heat_lut()[np.clip(cv2.resize(g, (ww, hh), interpolation=cv2.INTER_CUBIC) / dmax * 255, 0, 255).astype(np.uint8)]
    if not f.get("contact") and not measured:
        t[:] = (78, 72, 68)
    if f.get("contact"):
        cx = ww / 2.0 * (1.0 + float(np.clip(f.get("cx") or 0.0, -1, 1)))
        cy = hh / 2.0 * (1.0 + float(np.clip(f.get("cy") or 0.0, -1, 1)))
        cv2.drawMarker(t, (int(cx), int(cy)), (240, 240, 240), cv2.MARKER_CROSS, 7, 1)
    return t


def pad_corners(f, th_deg, rows, cols, pad_map="", size=0.34):
    """3-D corners of the pad on the inner face of the fingertip, in texture order TL, TR, BR, BL
    (texture row 0 = towards the fingertip, columns across the finger), and its centre / axes."""
    base, tip, fa, n, u = finger_frame(f, th_deg)
    hv = size * rows / float(max(rows, cols))
    hu = size * cols / float(max(rows, cols))
    c0 = tip - 0.55 * hv * fa + 0.05 * n
    ax_v, ax_u = fa.copy(), u.copy()
    if "swap" in pad_map:
        ax_v, ax_u, hv, hu = ax_u, ax_v, hu, hv
    if "flipx" in pad_map:
        ax_u = -ax_u
    if "flipy" in pad_map:
        ax_v = -ax_v
    TL = c0 + 0.5 * hv * ax_v - 0.5 * hu * ax_u
    TR = c0 + 0.5 * hv * ax_v + 0.5 * hu * ax_u
    BR = c0 - 0.5 * hv * ax_v + 0.5 * hu * ax_u
    BL = c0 - 0.5 * hv * ax_v - 0.5 * hu * ax_u
    return np.stack([TL, TR, BR, BL]), c0, ax_v, ax_u, n, hu, hv


def draw_hand3d(img, S, x0, y0, w, h, yaw=-62.0, pitch=22.0, zoom=1.0, pan=(0.0, 0.0)):
    """3-D view of the hand: palm, fingers (flexion + rotation), every gel pad with its measured depth
    map on it, shear arrows, slip rings, the object estimated from the touching pads, the wrist torsion
    as an arc around the palm. Pads facing away from the viewer show their back."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    s3 = min(w / 3.1, h / 2.05) * float(np.clip(zoom, 0.3, 6.0))
    cam = Cam3(yaw, pitch, s3, x0 + w / 2.0 + pan[0], y0 + 0.5 * h + pan[1])
    off = cam.proj(np.array([0.0, 0.0, 0.62]))[0] - cam.proj(np.zeros(3))[0]
    cam.cx, cam.cy = cam.cx - off[0], cam.cy - off[1]                 # centre the scene on the fingertips
    sub = img[y0:y0 + h, x0:x0 + w]                                   # zoomed in: nothing drawn outside the box
    img = sub if (zoom > 1.0 or pan[0] or pan[1]) else img
    if img is sub:
        cam.cx, cam.cy, x0, y0 = cam.cx - x0, cam.cy - y0, 0, 0
    fingers = S.get("fingers") or []
    layout = S.get("layout") or [90, 210, 330]
    pmap = S.get("pad_map") or ""
    items, centres, est_any = [], [], False
    # palm: a disc (always below everything)
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    rim = np.stack([1.05 * np.cos(ang), 1.05 * np.sin(ang), np.zeros_like(ang)], 1)
    pr, _ = cam.proj(rim)
    cv2.fillPoly(img, [np.round(pr).astype(np.int32)], (58, 52, 48), cv2.LINE_AA)
    cv2.polylines(img, [np.round(pr).astype(np.int32)], True, GRID, 1, cv2.LINE_AA)
    c0p, _ = cam.proj(np.zeros(3))
    cv2.circle(img, (int(c0p[0]), int(c0p[1])), 2, GRID, -1)
    Wd = S.get("wrist") or {}
    if Wd.get("valid") and Wd.get("active"):                # wrist torsion: an arc on the palm rim
        sweep = float(np.clip(Wd.get("tau_px", 0.0) / 3.0, -1, 1)) * math.radians(150)
        aa = np.linspace(math.radians(-90.0), math.radians(-90.0) + sweep, 24)
        ap, _ = cam.proj(np.stack([1.0 * np.cos(aa), 1.0 * np.sin(aa), np.zeros_like(aa)], 1))
        colw = RED if Wd.get("rot_slip") else AMBER
        cv2.polylines(img, [np.round(ap).astype(np.int32)], False, colw, 3, cv2.LINE_AA)
        _arrow(img, ap[-2], ap[-1], colw, 3)
        _put(img, f"wrist {Wd.get('tau_px', 0.0):+.2f} px {Wd.get('dir_name', '')}"
             + ("  ROT-SLIP" if Wd.get("rot_slip") else ""), x0 + w - 200, y0 + 18, colw, 0.42)
    for k, f in enumerate(fingers):
        th = layout[k % len(layout)]
        col = FCOL[k % 3] if f.get("flex_deg") is not None else RED
        base, tip, fa, n, u = finger_frame(f, th)
        q, dq = cam.proj(np.stack([base - np.array([0, 0, 0.12]), base, tip]))
        th_px = max(5, int(0.13 * cam.s))
        items.append((float(dq[:2].mean()), 1, (q[0], q[1]), tuple(int(0.65 * c) for c in col), th_px))
        items.append((float(dq[1:].mean()) - 0.08, 1, (q[1], q[2]), col, th_px))
        items.append((float(dq[1]) + 0.01, 2, q[1], col, (0.075 * cam.s, 1.0)))
        # the pad with the measured touch on it
        g, measured = touch_grid(f)
        est_any |= (not measured) and bool(f.get("contact"))
        Q, pc, av, au, pn, hu, hv = pad_corners(f, th, g.shape[0], g.shape[1], pmap)
        q2, qd = cam.proj(Q)
        facing = float(pn @ cam.c)
        # fingertip block behind the gel (a little larger), then the gel with its touch map. Seen from
        # behind the finger is drawn see-through: the map stays visible, dimmer (X-RAY)
        blk = pc - 0.05 * pn + 1.18 * (Q - pc)
        b2, bd = cam.proj(blk)
        tex = pad_texture(g, f, S, measured)
        lam = (0.55 + 0.45 * min(1.0, facing / 0.6)) if facing > 0.05 else 0.5
        items.append((float(max(bd.mean(), qd.mean())) + 0.02, 6,
                      (np.round(b2).astype(np.int32), tuple(int(0.55 * c) for c in col), q2,
                       np.clip(tex * lam, 0, 255).astype(np.uint8)), None))
        if f.get("slip"):
            sp, sd = cam.proj(pc + 0.05 * pn)
            items.append((float(sd) + 0.3, 2, sp, RED, (0.28 * cam.s, 0.35)))
        if f.get("contact"):
            centres.append(pc + 0.2 * pn)
            cx, cy = float(np.clip(f.get("cx") or 0.0, -1, 1)), float(np.clip(f.get("cy") or 0.0, -1, 1))
            cpt = pc - cy * 0.5 * hv * av + cx * 0.5 * hu * au + 0.03 * pn
            sx, sy = float(np.clip(f.get("shear_x") or 0.0, -6, 6)), float(np.clip(f.get("shear_y") or 0.0, -6, 6))
            if abs(sx) + abs(sy) > 0.3 and facing > -0.3:
                end = cpt + 0.05 * (sx * au - sy * av)
                aa, ad = cam.proj(np.stack([cpt, end]))
                items.append((float(ad.max()) + 0.35, 3, (aa[0], aa[1]), VIOLET, 2))
        lp, _ = cam.proj(base + np.array([0, 0, -0.12]) + 0.42 * np.array([math.cos(math.radians(th)), math.sin(math.radians(th)), 0.0]))
        fd = f.get("flex_deg")
        items.append((1e9, 4, (float(np.clip(lp[0] - 40, x0 + 4, x0 + w - 150)), float(np.clip(lp[1] + 18, y0 + 34, y0 + h - 22))),
                      col, f"ID{f.get('id')} " + (f"{fd:+.1f}" if fd is not None else "ERR") + f"  rot {f.get('rot_deg') or 0.0:+.1f}"))
    if len(centres) >= 2:
        oc = np.mean(centres, 0)
        rad = float(np.mean([np.linalg.norm(c - oc) for c in centres]))
        op, od = cam.proj(oc)
        items.append((float(od), 2, op, (150, 145, 140), (max(4.0, rad * cam.s), 0.30)))
        obj_txt = f"object  {len(centres)} pads  off-centre {100 * float(np.linalg.norm(oc[:2])):.0f} %"
    else:
        obj_txt = "object  not held" if not centres else "object  1 pad"
    paint(img, [it for it in items if it[1] != 4])
    for it in items:
        if it[1] == 4:
            _put(img, it[4], it[2][0], it[2][1], it[3], 0.42)
    _put(img, "3-D HAND  (flexion + rotation + touch on the pads)", x0 + 10, y0 + 18, DIM, 0.42)
    _put(img, obj_txt + ("   touch: est." if est_any else "") + "   (pad from behind: dimmed)", x0 + 10,
         y0 + h - 10, DIM, 0.38)
    _put(img, "[ ] - = rotate", x0 + w - 108, y0 + h - 10, (110, 110, 110), 0.38)


def draw_touch3d(img, g, f, S, x0, y0, w, h, measured=True, n_max=18):
    """The gel of one finger as a 3-D surface (height = gel depth, same scale for all fingers), with the
    pad outline, the contact centre and the shear arrow."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 35, 32), -1)
    dmax = max(1e-6, S.get("depth_max") or 0.6)
    rows, cols = g.shape
    k = max(rows, cols) / float(n_max)
    if k > 1.0:
        g = resample(g, max(3, int(round(rows / k))), max(3, int(round(cols / k))))
        rows, cols = g.shape
    hx = 1.0 * cols / max(rows, cols)
    hy = 1.0 * rows / max(rows, cols)
    xs = np.linspace(-hx, hx, cols)[None, :]
    ys = np.linspace(hy, -hy, rows)[:, None]                 # image row 0 = far edge
    Z = 0.85 * np.clip(g / dmax, 0.0, 1.3)
    P = np.stack(np.broadcast_arrays(xs, ys, Z), -1)
    cam = Cam3(-60.0, 38.0, 0.34 * min(w, 1.45 * h), x0 + w / 2.0, y0 + h / 2.0 + 0.16 * h)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 35, 32), -1)
    lut = heat_lut()
    cq = lut[np.clip((0.25 * (g[:-1, :-1] + g[1:, 1:] + g[:-1, 1:] + g[1:, :-1]) / dmax) * 255, 0, 255).astype(np.uint8)]
    items = []
    # the flat (untouched) gel as one plate, only the raised part as a shaded mesh
    corners = np.array([[-hx, -hy, 0], [hx, -hy, 0], [hx, hy, 0], [-hx, hy, 0]], float)
    cp, _ = cam.proj(corners)
    cv2.fillConvexPoly(img, np.round(cp).astype(np.int32), tuple(int(0.8 * c) for c in heat(0.0)), cv2.LINE_AA)
    cv2.polylines(img, [np.round(cp).astype(np.int32)], True, GRID, 1, cv2.LINE_AA)
    up = g > 0.02 * dmax
    m = up[:-1, :-1] | up[1:, 1:] | up[:-1, 1:] | up[1:, :-1]
    if m.any():
        mesh_polys(cam, P, cq, items, mask=m)
        paint(img, items)
    post, _ = cam.proj(np.array([[hx, hy, 0.0], [hx, hy, 0.85]]))
    cv2.line(img, tuple(np.round(post[0]).astype(int)), tuple(np.round(post[1]).astype(int)), GRID, 1, cv2.LINE_AA)
    _put(img, f"{dmax:.2f}", post[1][0] + 3, post[1][1] + 4, DIM, 0.33)
    if f.get("contact"):
        sx, sy = float(np.clip(f.get("shear_x") or 0.0, -6, 6)), float(np.clip(f.get("shear_y") or 0.0, -6, 6))
        cx, cy = float(np.clip(f.get("cx") or 0.0, -1, 1)), float(np.clip(f.get("cy") or 0.0, -1, 1))
        i_, j_ = int(np.clip((cy + 1) / 2 * (rows - 1), 0, rows - 1)), int(np.clip((cx + 1) / 2 * (cols - 1), 0, cols - 1))
        top = np.array([cx * hx, -cy * hy, float(Z[i_, j_]) + 0.08])
        if abs(sx) + abs(sy) > 0.3:
            ab, _ = cam.proj(np.stack([top, top + 0.11 * np.array([sx, -sy, 0.0])]))
            _arrow(img, ab[0], ab[1], VIOLET, 2)
    _put(img, "3-D touch" + ("" if measured else " (est.)"), x0 + 5, y0 + 13, DIM if measured else AMBER, 0.36)


def draw_map2d(img, g, f, S, x0, y0, ps, measured=True):
    """The gel of one finger seen straight on: the depth map as a heat map, contact centre, shear, twist."""
    dmax = max(1e-6, S.get("depth_max") or 0.6)
    rows, cols = g.shape
    hh = max(8, int(round(ps * rows / max(rows, cols))))
    ww = max(8, int(round(ps * cols / max(rows, cols))))
    yo, xo = y0 + (ps - hh) // 2, x0 + (ps - ww) // 2
    big = cv2.resize(g, (ww, hh), interpolation=cv2.INTER_CUBIC)
    rgb = heat_lut()[np.clip(big / dmax * 255, 0, 255).astype(np.uint8)]
    if not f.get("contact") and not measured:
        rgb[:] = (60, 54, 50)
    img[yo:yo + hh, xo:xo + ww] = rgb
    cv2.rectangle(img, (xo, yo), (xo + ww, yo + hh), GRID if f.get("fresh") else RED, 1)
    if f.get("contact"):
        cx = xo + ww / 2.0 * (1.0 + float(np.clip(f.get("cx") or 0.0, -1, 1)))
        cy = yo + hh / 2.0 * (1.0 + float(np.clip(f.get("cy") or 0.0, -1, 1)))
        cv2.drawMarker(img, (int(cx), int(cy)), (240, 240, 240), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        sv = np.array([float(f.get("shear_x") or 0.0), float(f.get("shear_y") or 0.0)]) * 6.0
        nv = float(np.linalg.norm(sv))
        if nv > 0.45 * ps:
            sv *= 0.45 * ps / nv
        _arrow(img, (cx, cy), (cx + sv[0], cy + sv[1]), VIOLET, 2)
        tw = math.degrees(float(f.get("twist") or 0.0))
        if abs(tw) > 0.3:
            cv2.ellipse(img, (int(cx), int(cy)), (12, 12), 0, -90, -90 + float(np.clip(tw * 8.0, -170, 170)),
                        AMBER, 2, cv2.LINE_AA)
        if f.get("slip"):
            cv2.rectangle(img, (xo - 2, yo - 2), (xo + ww + 2, yo + hh + 2), RED, 2)
    elif not f.get("fresh"):
        _put(img, "no data", xo + 6, yo + hh // 2 + 4, RED, 0.4)


def draw_pad(img, f, k, S, x0, y0, w, h):
    """One finger card: side view (flexion angle) | gel depth map | gel in 3-D | numbers."""
    col = FCOL[k % 3]
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), PANEL, -1)
    cv2.rectangle(img, (x0, y0), (x0 + 4, y0 + h), col, -1)
    fd = f.get("flex_deg")
    _put(img, f"ID{f.get('id')}", x0 + 12, y0 + 20, col, 0.6, 2)
    st_txt = (f.get("state") or f.get("phase") or "").upper()
    sc = (RED if f.get("slip") else AMBER if f.get("offpad") else GREEN if f.get("contact") else DIM)
    _put(img, ("SLIP  " if f.get("slip") else "") + ("OFF-PAD  " if f.get("offpad") else "") + st_txt,
         x0 + 56, y0 + 19, sc, 0.42)
    body = h - 34
    # ---- side view: flexion angle (grey = open pose and closed pose)
    sx0, sy0 = x0 + 12, y0 + 28
    bx, by = sx0 + 8, sy0 + body - 6
    cv2.line(img, (sx0, by), (sx0 + body, by), GRID, 2)
    Lk = body - 14
    for a_, c_, t_ in ((math.radians(12), GRID, 1), (math.radians(85), GRID, 1),
                       (_flex_angle(f), col if fd is not None else RED, 5)):
        ex, ey = bx + Lk * math.sin(a_), by - Lk * math.cos(a_)
        cv2.line(img, (int(bx), int(by)), (int(ex), int(ey)), c_, t_, cv2.LINE_AA)
    cv2.circle(img, (int(bx), int(by)), 5, col, -1, cv2.LINE_AA)
    cv2.ellipse(img, (int(bx), int(by)), (20, 20), 0, -90, -90 + math.degrees(_flex_angle(f)), DIM, 1, cv2.LINE_AA)
    # ---- gel: depth map + 3-D surface
    g, measured = touch_grid(f)
    ps = body - 4
    px0, py0 = sx0 + body + 8, sy0
    draw_map2d(img, g, f, S, px0, py0, ps, measured)
    t3w = int(np.clip(w - (px0 - x0) - ps - 8 - 214, 110, 230))
    draw_touch3d(img, g, f, S, px0 + ps + 6, y0 + 6, t3w, h - 12, measured)
    # ---- numbers
    tx = px0 + ps + 6 + t3w + 10
    ld = abs(f.get("load") or 0.0)
    gl = max(1e-6, S.get("grip_load") or 0.6)
    lc = GREEN if ld < 0.75 * gl else AMBER if ld < gl else RED
    d = f.get("depth") or 0.0
    dmax = max(1e-6, S.get("depth_max") or 0.6)
    ref = S.get("depth_ref") or 0.0
    step = max(13, min(16, (body - 8) // 6))
    y = sy0 + 10
    _put(img, ("flex " + (f"{fd:+.1f}" if fd is not None else "ERR") + f"  {100 * (f.get('closed') or 0.0):.0f}%"),
         tx, y, TXT, 0.42)
    y += step
    _put(img, f"load {ld:.2f} Nm", tx, y, lc, 0.4)
    _put(img, f"rot {f.get('rot_deg') or 0.0:+.1f}", tx + 104, y, DIM, 0.4)
    y += step
    _put(img, f"depth {d:.2f} mm", tx, y, GREEN if f.get("contact") else DIM, 0.42)
    y += 5
    bw = max(40, x0 + w - tx - 10)
    cv2.rectangle(img, (tx, y), (tx + bw, y + 6), GRID, 1)
    cv2.rectangle(img, (tx, y), (tx + int(bw * min(1.0, d / dmax)), y + 6),
                  heat(d / dmax) if f.get("contact") else (110, 110, 110), -1)
    xr = tx + int(bw * min(1.0, ref / dmax))
    cv2.line(img, (xr, y - 3), (xr, y + 9), AMBER, 2)
    y += step + 2
    _put(img, f"shear {f.get('shear_x') or 0.0:+.1f} {f.get('shear_y') or 0.0:+.1f} px", tx, y, VIOLET, 0.4)
    y += step
    _put(img, f"twist {math.degrees(float(f.get('twist') or 0.0)):+.1f}  area {100 * (f.get('area') or 0.0):.0f}%",
         tx, y, DIM, 0.4)
    y += step
    _put(img, f"{f.get('cam') or '-'} {f.get('fps') or 0.0:.0f}fps", tx, y, DIM if f.get("fresh") else RED, 0.4)


def render(S, w=640, h=700, view=None):
    """The whole twin panel (BGR image w x h) from one state dict.
    view: dict(top="3d" | "2d", yaw=deg, pitch=deg) - how the hand is drawn on top."""
    view = view or {}
    img = np.full((h, w, 3), BG, np.uint8)
    mcol = {"IDLE": DIM, "GRASP": GREEN, "ACTIVE": VIOLET, "OPEN": BLUE, "HOLD": TXT, "STOPPED": RED}.get(
        S.get("mode", ""), TXT)
    _put(img, "DIGITAL TWIN", 12, 22, TXT, 0.55, 2)
    _put(img, S.get("mode", "-"), 150, 22, mcol, 0.55, 2)
    rf = S.get("refine") or {}
    if rf.get("phase"):
        _put(img, f"REFINE {rf['phase']}" + (f" ID{rf['finger']}" if rf.get("finger") else ""), 230, 22, AMBER, 0.45)
    _put(img, f"ref {S.get('depth_ref') or 0.0:.2f} mm", w - 110, 22, AMBER, 0.42)
    Ld = S.get("load") or {}
    if Ld.get("loaded"):
        _put(img, f"LOAD {Ld.get('px') or 0.0:.2f} px" + (f" = {Ld['g']:.0f} g" if Ld.get("g") else "")
             + (f"  grip +{100 * Ld['need']:.0f} %" if Ld.get("need") is not None else ""), w - 330, 22, AMBER, 0.42)
    fingers = S.get("fingers") or []
    n = max(1, len(fingers))
    card_h = int(np.clip((h - 40) * 0.19, 118, 150))
    top_h = h - 34 - n * (card_h + 6) - 6
    if view.get("top", "3d") == "2d":
        draw_top(img, S, 8, 32, w - 16, top_h)
    else:
        draw_hand3d(img, S, 8, 32, w - 16, top_h, view.get("yaw", -62.0), view.get("pitch", 22.0),
                    view.get("zoom", 1.0), (view.get("pan_x", 0.0), view.get("pan_y", 0.0)))
        if view.get("zoom", 1.0) != 1.0 or view.get("pan_x") or view.get("pan_y"):
            _put(img, f"zoom {view.get('zoom', 1.0):.1f}x  (double-click = reset)", 16, 32 + 34, DIM, 0.38)
    view["_box"] = (8, 32, w - 16, top_h)                             # where the 3-D hand is (for the mouse)
    y = 32 + top_h + 6
    for k, f in enumerate(fingers):
        draw_pad(img, f, k, S, 8, y, w - 16, card_h)
        y += card_h + 6
    return img


class Mouse3D:
    """3-D mouse for the twin hand, for any OpenCV window that shows render()'s image at x offset x_off:
    left drag = rotate (yaw / pitch), right drag or Ctrl + left drag = pan, wheel = zoom (towards the
    cursor), Shift + wheel = tilt, double-click = reset the view. Changes view in place, sets view['dirty']."""

    HOME = dict(yaw=-62.0, pitch=22.0, zoom=1.0, pan_x=0.0, pan_y=0.0)

    def __init__(self, view, x_off=0):
        self.v, self.x_off, self.d = view, x_off, None
        self.home = {k: view.get(k, d) for k, d in self.HOME.items()}

    def inside(self, x, y):
        bx, by, bw, bh = self.v.get("_box", (0, 0, 10 ** 5, 10 ** 5))
        return bx <= x < bx + bw and by <= y < by + bh

    def __call__(self, ev, x, y, flags, _p=None):
        v = self.v
        x -= self.x_off
        if ev in (cv2.EVENT_LBUTTONDBLCLK, cv2.EVENT_MBUTTONDOWN) and self.inside(x, y):
            v.update(self.home)
        elif ev in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) and self.inside(x, y):
            pan = ev == cv2.EVENT_RBUTTONDOWN or bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            self.d = dict(x=x, y=y, pan=pan, **{k: v.get(k, d) for k, d in self.HOME.items()})
            return
        elif ev == cv2.EVENT_MOUSEMOVE and self.d is not None:
            if not flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON):
                self.d = None
                return
            dx, dy = x - self.d["x"], y - self.d["y"]
            if self.d["pan"]:
                v["pan_x"], v["pan_y"] = self.d["pan_x"] + dx, self.d["pan_y"] + dy
            else:
                v["yaw"] = (self.d["yaw"] - 0.5 * dx + 180.0) % 360.0 - 180.0
                v["pitch"] = float(np.clip(self.d["pitch"] + 0.4 * dy, -10.0, 85.0))
        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.d = None
            return
        elif ev in (cv2.EVENT_MOUSEWHEEL, getattr(cv2, "EVENT_MOUSEHWHEEL", -99)) and self.inside(x, y):
            try:
                up = cv2.getMouseWheelDelta(flags) > 0
            except Exception:
                up = flags > 0
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                v["pitch"] = float(np.clip(v.get("pitch", 22.0) + (5.0 if up else -5.0), -10.0, 85.0))
            else:
                z0 = v.get("zoom", 1.0)
                z1 = float(np.clip(z0 * (1.15 if up else 1 / 1.15), 0.4, 6.0))
                bx, by, bw, bh = v.get("_box", (0, 0, 1, 1))     # keep the point under the cursor in place
                cx, cy = bx + bw / 2.0 + v.get("pan_x", 0.0), by + bh / 2.0 + v.get("pan_y", 0.0)
                v["pan_x"] = v.get("pan_x", 0.0) + (x - cx) * (1 - z1 / z0)
                v["pan_y"] = v.get("pan_y", 0.0) + (y - cy) * (1 - z1 / z0)
                v["zoom"] = z1
                if abs(z1 - 1.0) < 0.02 and abs(v["pan_x"]) < 3 and abs(v["pan_y"]) < 3:
                    v.update(zoom=1.0, pan_x=0.0, pan_y=0.0)
        else:
            return
        v["dirty"] = True


def encode_grid(g):
    """float mm map -> the compact 'grid' dict (same format as tactile_ros.depth_grid)."""
    g = np.clip(np.nan_to_num(np.asarray(g, np.float32)), 0.0, None)
    sc = max(float(g.max()), 1e-4) / 255.0
    q = np.clip(np.round(g / sc), 0, 255).astype(np.uint8)
    return {"n": [int(g.shape[0]), int(g.shape[1])], "s": round(sc, 7), "d": base64.b64encode(q.tobytes()).decode("ascii")}


def demo_state(t=0.0):
    """A moving example state with measured-looking gel maps (for twin_viewer.py --demo and for tests):
    ID1 a round contact that wanders, ID2 an elongated (edge-like) contact, ID3 two small contacts."""
    fs = []
    yy, xx = np.mgrid[-1:1:24j, -1:1:24j]
    for k in range(3):
        cl = 0.5 + 0.45 * math.sin(0.6 * t + 0.3 * k)
        touch = cl > 0.75
        dep = 0.5 * max(0.0, cl - 0.75) / 0.2
        cx, cy = 0.4 * math.sin(t + k), 0.3 * math.cos(0.7 * t)
        if k == 0:
            g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.12)
        elif k == 1:
            g = np.exp(-((xx - cx) ** 2) / 0.5 - ((yy - 0.55) ** 2) / 0.03)
        else:
            g = np.exp(-((xx + 0.4) ** 2 + (yy - cy) ** 2) / 0.05) + 0.7 * np.exp(-((xx - 0.45) ** 2 + (yy + 0.2) ** 2) / 0.05)
        g = dep * g / max(1e-6, g.max())
        g[g < 0.02] = 0.0
        fs.append(dict(id=k + 1, rot_id=k + 4, cam=f"camera{k + 1}", phase="squeeze" if touch else "close",
                       contact=touch, fresh=True, fps=15.0, flex_deg=-15.0 - 65.0 * cl, open_deg=-15.0,
                       closed_deg=-80.0, closed=cl, rot_deg=8.0 * math.sin(0.4 * t + k),
                       depth=dep, cx=cx, cy=cy, area=0.2 * max(0.0, cl - 0.7), force=900.0 * max(0.0, cl - 0.75),
                       shear_x=2.5 * math.sin(1.3 * t + k), shear_y=1.5 * math.cos(1.1 * t), twist=0.03 * math.sin(t),
                       slip=touch and math.sin(2.0 * t + k) > 0.95, load=0.5 * cl, offpad=False, state="",
                       grid=encode_grid(g) if touch else None))
    return dict(v=1, t=t, mode="GRASP", msg="demo", depth_ref=0.3, depth_max=0.6, grip_load=0.6,
                joints=[dict(id=i + 1, deg=-40.0, sent=-40.0, load=0.2) for i in range(6)], fingers=fs,
                wrist=dict(valid=True, active=True, tau_px=1.5 * math.sin(0.5 * t), dir_name="CW", rot_slip=False),
                refine={}, layout=[90, 210, 330], pad_map="")
