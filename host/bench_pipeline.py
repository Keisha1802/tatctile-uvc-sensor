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


def bench(nthreads):
    cv2.setNumThreads(nthreads)
    cfg = dp.SensorConfig()
    for k, v in dict(flow_deadzone=0.0, flow_ema=0.0, flow_ema_release=0.0, temporal_avg=1,
                     auto_rebaseline=False, rebaseline_hold=0, valid_margin=15).items():
        setattr(cfg, k, v)
    cfg.dist, cfg.roi_thresh, cfg.focal = (-0.6, -0.21, 0, 0, 0), 104, 713.0
    fr = frames()
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
    cpu_info()
    for nt in (1, os.cpu_count() or 4):
        bench(nt)
    stop = threading.Event()
    th = threading.Thread(target=poller, args=(stop,), daemon=True); th.start()
    print("with a 0.5 ms polling thread next to it (the old recorder/pipeline loops):")
    bench(1)
    stop.set(); th.join()
