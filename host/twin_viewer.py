"""Stand-alone DIGITAL TWIN viewer / recorder / replayer for the 3-finger tactile hand.

  live      python3 twin_viewer.py --port=9900            (servo_grasp.py ... --twin-udp=THIS_PC_IP:9900)
  record    python3 twin_viewer.py --port=9900 --record=run1.jsonl
  replay    python3 twin_viewer.py --replay=run1.jsonl [--speed=1.0]
  from CSV  python3 twin_viewer.py --csv=grasp_logs/grasp_20261117_143548.csv [--speed=2] [--start=60]
  demo      python3 twin_viewer.py --demo
  to video  add --video=out.mp4 (with --replay / --csv / --demo; no window needed: --headless)

Keys: q quit, space pause, . one step (while paused), [ ] - = rotate / tilt the 3-D hand, v 2-D/3-D.
Mouse (3-D hand): left drag rotate, right drag / Ctrl+drag pan, wheel zoom, Shift+wheel tilt, double-click reset.  --yaw=-62 --pitch=22 --top=3d|2d --pad=swap,flipx,flipy
The state it draws is exactly the JSON servo_grasp sends (schema: twin_view.py) - a 3-D twin reads the
same stream:   import socket, json;  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
               s.bind(("0.0.0.0", 9900));  state = json.loads(s.recv(65536))
               state["fingers"][0]["flex_deg"], ["rot_deg"], ["depth"], ["cx"], ["cy"], ["shear_x"] ...
"""
import json
import socket
import sys
import time

import cv2

import twin_view


def opt(name, default, cast=str):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return cast(a.split("=", 1)[1])
    return default


def csv_states(path, start=0.0):
    """A servo_grasp CSV log -> twin states (one per row)."""
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    def fl(r, k, d=0.0):
        try:
            v = r.get(k, "")
            return float(v) if v not in ("", None) else d
        except ValueError:
            return d
    if not rows:
        return
    n_f = sum(1 for i in range(1, 7) if f"f{i}_cam" in rows[0])
    rot0 = [fl(rows[0], f"id{i + n_f + 1}_act_deg") for i in range(n_f)]
    ang = {i: [fl(r, f"id{i}_act_deg", None) for r in rows] for i in range(1, n_f + 1)}
    cf = {i: [fl(r, f"f{i}_closed_frac") for r in rows] for i in range(1, n_f + 1)}
    for r in rows:
        t = fl(r, "t_s")
        if t < start:
            continue
        fs = []
        for i in range(1, n_f + 1):
            a = fl(r, f"id{i}_act_deg", None)
            fs.append(dict(
                id=i, rot_id=i + n_f, cam=r.get(f"f{i}_cam", ""), phase=r.get(f"f{i}_phase", ""),
                contact=fl(r, f"f{i}_contact") > 0.5, fresh=fl(r, f"f{i}_fresh", 1.0) > 0.5, fps=fl(r, f"f{i}_s_fps"),
                flex_deg=a, open_deg=None, closed_deg=None, closed=fl(r, f"f{i}_closed_frac"),
                rot_deg=fl(r, f"id{i + n_f}_act_deg") - rot0[i - 1], depth=fl(r, f"f{i}_depth_mm"),
                cx=fl(r, f"f{i}_s_cx"), cy=fl(r, f"f{i}_s_cy"), area=fl(r, f"f{i}_s_area_frac"),
                force=fl(r, f"f{i}_s_force"), shear_x=fl(r, f"f{i}_s_shear_x"), shear_y=fl(r, f"f{i}_s_shear_y"),
                twist=fl(r, f"f{i}_s_twist"), slip=fl(r, f"f{i}_s_slip") > 0.5 and fl(r, f"f{i}_contact") > 0.5,
                load=fl(r, f"id{i}_load_Nm"), offpad=r.get(f"f{i}_phase", "") == "offpad", state=""))
        yield dict(v=1, t=t, mode=r.get("mode", ""), msg=r.get("message", ""), depth_ref=fl(r, "depth_ref_mm"),
                   depth_max=0.6, grip_load=0.6,
                   joints=[dict(id=i, deg=fl(r, f"id{i}_act_deg", None), sent=fl(r, f"id{i}_sent_deg", None),
                                load=fl(r, f"id{i}_load_Nm")) for i in range(1, 2 * n_f + 1)],
                   fingers=fs,
                   wrist=dict(valid=fl(r, "wrist_valid") > 0.5, active=fl(r, "wrist_active") > 0.5,
                              tau_px=fl(r, "wrist_tau_px"), rot_slip=fl(r, "wrist_rot_slip") > 0.5, dir_name=""),
                   refine={}, layout=[90, 210, 330],
                   load=(dict(loaded=True, px=fl(r, "load_px"), g=fl(r, "load_g") or None,
                              need=None) if fl(r, "load_px") > 0 else {}))


def main():
    w, h = opt("width", 640, int), opt("height", 720, int)
    view = dict(top=opt("top", "3d"), yaw=opt("yaw", -62.0, float), pitch=opt("pitch", 22.0, float))
    pad = opt("pad", None)
    mouse = twin_view.Mouse3D(view)
    speed, headless = opt("speed", 1.0, float), "--headless" in sys.argv
    if not headless:
        cv2.namedWindow("digital twin")
    video, vw = opt("video", ""), None
    rec = open(opt("record", ""), "a") if opt("record", "") else None
    src = None
    if "--demo" in sys.argv:
        src = (dict(twin_view.demo_state(0.05 * i), t=0.05 * i) for i in range(10 ** 9))
    elif opt("replay", ""):
        src = (json.loads(ln) for ln in open(opt("replay", "")) if ln.strip())
    elif opt("csv", ""):
        src = csv_states(opt("csv", ""), opt("start", 0.0, float))
    sock = None
    if src is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", opt("port", 9900, int)))
        sock.settimeout(0.05)
        print(f"listening on udp {opt('port', 9900, int)} (servo_grasp.py --twin-udp=<this pc>:{opt('port', 9900, int)})")
    S, t_prev, paused, n, t_rx = None, None, False, 0, 0.0
    limit = opt("frames", 0, int)
    step, mouse_on = False, False
    while True:
        if sock is not None:
            try:
                data = None
                while True:                              # newest packet only
                    data = sock.recv(65536)
                    sock.settimeout(0.0)
            except (socket.timeout, BlockingIOError, OSError):
                sock.settimeout(0.05)
            if data:
                try:
                    S, t_rx = json.loads(data), time.time()
                    if rec:
                        rec.write(json.dumps(S) + "\n")
                except ValueError:
                    pass
        elif not paused or step:
            step = False
            try:
                S = next(src)
            except StopIteration:
                break
            if t_prev is not None and not headless and not video:
                time.sleep(max(0.0, min(0.5, (S.get("t", 0.0) - t_prev) / max(speed, 1e-3))))
            t_prev = S.get("t", 0.0)
        if S is None:
            img = twin_view.render(dict(mode="-", fingers=[]), w, h, view)
            twin_view._put(img, "waiting for data ...", 20, h // 2, twin_view.AMBER, 0.6)
        else:
            if pad is not None:
                S = dict(S, pad_map=pad)
            img = twin_view.render(S, w, h, view)
            if sock is not None and time.time() - t_rx > 1.0:
                twin_view._put(img, f"NO DATA for {time.time() - t_rx:.0f} s", w - 170, 44, twin_view.RED, 0.5, 2)
            elif sock is None:
                twin_view._put(img, f"t {S.get('t', 0.0):.1f} s", w - 110, 44, twin_view.DIM, 0.42)
        if video:
            if vw is None:
                vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 20.0 * speed, (w, h))
            vw.write(img)
        n += 1
        if limit and n >= limit:
            break
        if not headless:
            cv2.imshow("digital twin", img)
            if not mouse_on:
                try:                                          # after the first imshow (Qt needs the window)
                    cv2.setMouseCallback("digital twin", mouse)
                    mouse_on = True
                except cv2.error:
                    pass
            k = cv2.waitKey(1 if sock is None else 15) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                paused = not paused
            step = k == ord(".")
            if k in (ord("["), ord("]")):
                view["yaw"] = (view["yaw"] + (-15.0 if k == ord("[") else 15.0) + 180.0) % 360.0 - 180.0
            elif k in (ord("-"), ord("=")):
                view["pitch"] = max(-10.0, min(85.0, view["pitch"] + (-10.0 if k == ord("-") else 10.0)))
            elif k == ord("v"):
                view["top"] = "2d" if view["top"] == "3d" else "3d"
    if vw is not None:
        vw.release()
        print("video written:", video)
    if rec:
        rec.close()


if __name__ == "__main__":
    main()
