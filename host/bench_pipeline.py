#!/usr/bin/env python3
"""
bench_pipeline.py — how fast is ONE camera pipeline on this machine (no camera, no GUI,
no other threads)? Separates "the machine/libraries are slow" from "threads fight".

  ./run.sh --script=bench_pipeline.py
Reference (2-core cloud VM): flow ~7 ms, depth ~6 ms, display colours ~2 ms per frame.
A Raspberry Pi 5 should be in the same range.
"""
import os
import platform
import sys
import time

import cv2
import numpy as np

import deltact_plus as dp


def cpu_info():
    def rd(p):
        try:
            with open(p) as fh:
                return fh.read().strip()
        except OSError:
            return "?"
    base = "/sys/devices/system/cpu/cpu0/cpufreq/"
    print(f"python {platform.python_version()}  numpy {np.__version__}  opencv {cv2.__version__} "
          f"({os.path.dirname(cv2.__file__)})")
    print(f"cores {os.cpu_count()}  governor {rd(base + 'scaling_governor')}  "
          f"freq {rd(base + 'scaling_cur_freq')} kHz (max {rd(base + 'scaling_max_freq')})  "
          f"cv2 threads {cv2.getNumThreads()}  optimized {cv2.useOptimized()}")
    bi = cv2.getBuildInformation()
    for key in ("Parallel framework", "CPU/HW features", "Baseline", "Dispatched code"):
        for line in bi.splitlines():
            if key in line:
                print("  " + line.strip())
                break
    try:
        print("  " + open("/sys/class/thermal/thermal_zone0/temp").read().strip()[:-3] + " °C")
    except OSError:
        pass


def frames(n=20, w=640, h=480):
    rng = np.random.default_rng(0)
    img = np.zeros((h, w, 3), np.uint8)
    img[40:440, 80:560] = cv2.resize(rng.integers(0, 255, (100, 120, 3)).astype(np.uint8),
                                     (480, 400), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    g = np.exp(-((xx - 330) ** 2 + (yy - 250) ** 2) / (2 * 45 ** 2))
    out = []
    for a in np.linspace(0, 1, n):
        d = 8 * a * g / 45
        out.append(cv2.remap(img, (xx - (xx - 330) * d).astype(np.float32),
                             (yy - (yy - 250) * d).astype(np.float32), cv2.INTER_LINEAR))
    return out


CFG_FILE = None          # --config : use host/sensor_config.json (what the sensor programs load)
REAL = None              # --camera[=name] : real frames from a Pi Zero instead of synthetic ones


def grab_real(name=None, n=40):
    """Grab n frames from a Tactile UVC Cam (same reader as the sensor programs)."""
    import run_sensor_fast as rsf
    dev = rsf.resolve_usb(name)
    rd = rsf.UsbCameraReader(dev, 640, 480, 15); rd.start()
    out, last = [], None
    t_end = time.time() + 15
    while len(out) < n and time.time() < t_end:
        ok, f, seq = rsf.wait_frame(rd, last, 0.5)
        if ok and seq != last:
            out.append(f.copy()); last = seq
    rd.stop()
    print(f"real frames: {len(out)} from {dev}, shape {out[0].shape if out else None}")
    return out


def substeps(s, f, n=30):
    """Time the parts of TactileSensor.process on frame f (CPU time, this thread)."""
    t = dict(undistort=0.0, gray_blur=0.0, dis_flow=0.0, mask=0.0, depth=0.0)
    cfg = s.cfg
    for _ in range(n):
        c0 = time.thread_time(); roi = s.undist.process(f); c1 = time.thread_time()
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if cfg.denoise_ksize > 0:
            k = cfg.denoise_ksize | 1
            g = cv2.GaussianBlur(g, (k, k), 0)
        c2 = time.thread_time(); fl = s.tracker.update(g); c3 = time.thread_time()
        if s.mask is not None:
            fl = fl * s.mask[..., None]
        c4 = time.thread_time(); s.depth.compute(fl); c5 = time.thread_time()
        for k_, v_ in zip(t, (c1 - c0, c2 - c1, c3 - c2, c4 - c3, c5 - c4)):
            t[k_] += v_
    print("   substeps (cpu): " + "  ".join(f"{k} {v / n * 1000:.1f}" for k, v in t.items())
          + f"   roi {s.undist.roi}")


def bench(nthreads):
    cv2.setNumThreads(nthreads)
    cfg = dp.SensorConfig.load(CFG_FILE) if CFG_FILE else dp.SensorConfig()
    for k, v in dict(flow_deadzone=0.0, flow_ema=0.0, flow_ema_release=0.0, temporal_avg=1,
                     auto_rebaseline=False, rebaseline_hold=0, valid_margin=15).items():
        setattr(cfg, k, v)
    cfg.dist, cfg.roi_thresh, cfg.focal = (-0.6, -0.21, 0, 0, 0), 104, 713.0
    fr = REAL if REAL else frames()
    s = dp.TactileSensor(cfg, 640, 480)
    s.process(fr[0])
    t = {"flow": 0.0, "depth": 0.0, "colours": 0.0}
    n = 60
    for i in range(n):
        f = fr[i % len(fr)]
        cfg.enable_depth = False
        t0 = time.perf_counter(); r = s.process(f); t1 = time.perf_counter()
        d = s.depth.compute(r["flow"]); t2 = time.perf_counter()
        dp.flow_to_color(r["flow"]); dp.depth_to_color(d); t3 = time.perf_counter()
        t["flow"] += t1 - t0; t["depth"] += t2 - t1; t["colours"] += t3 - t2
    tot = sum(t.values()) / n * 1000
    print(f"cv2 threads {nthreads}: " + "  ".join(f"{k} {v / n * 1000:.1f} ms" for k, v in t.items())
          + f"  | total {tot:.1f} ms -> {1000 / tot:.0f} fps for one camera")
    substeps(s, fr[len(fr) // 2])


def poller(stop):
    """What the old code did: a thread checking for a new frame every 0.5 ms."""
    import threading
    lock = threading.Lock()
    while not stop.is_set():
        with lock:
            pass
        time.sleep(0.0005)


if __name__ == "__main__":
    import threading
    for a in sys.argv[1:]:
        if a == "--camera" or a.startswith("--camera="):
            REAL = grab_real(a.split("=", 1)[1] if "=" in a else None)
    if "--config" in sys.argv:
        CFG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_config.json")
        c = dp.SensorConfig.load(CFG_FILE)
        print(f"config {CFG_FILE}: work {c.work_w}x{c.work_h}, DIS {c.dis_preset} "
              f"finest {c.dis_finest_scale} grad {c.dis_grad_iters} var {c.dis_var_iters}, "
              f"tracking {c.tracking_mode}, denoise {c.denoise_ksize}, sigma {c.sigma}, "
              f"guided r {c.guided_radius}, subsample {c.density_subsample}")
    cpu_info()
    for nt in (1, os.cpu_count() or 4):
        bench(nt)
    print("same, but computed in a separate THREAD (like the sensor programs):")
    th = threading.Thread(target=bench, args=(1,)); th.start(); th.join()
