#!/usr/bin/env python3
"""
record_dual.py — RECORD raw frames from 2 cameras at once: CSI (Picamera2) + USB (Pi Zero UVC)
==============================================================================================
No processing (flow/depth) while recording -> the Pi 5 CPU is used only to capture
and write, so no frames are lost. Heavy processing is done afterwards.

Output (one folder per recording):
  rec_YYYYmmdd_HHMMSS/
    csi/000000.png ...     raw CSI camera frames (BGR, no overlay)
    usb/000000.png ...     raw USB camera frames (green/corrupt frames are NOT saved)
    frames.csv             1 row per frame: cam, idx, seq, t_host, t_sensor_ns, file
    meta.json              profile, size, measured fps, lost frames per camera
  t_host = time.monotonic() when the frame was grabbed (the SAME clock for both cameras)
  -> align the two cameras via t_host. t_sensor_ns = SensorTimestamp (CSI only).

Usage (on the Pi 5, in the same folder as run_sensor_fast.py):
  python3 record_dual.py                     # CSI turbo + USB, PNG
  python3 record_dual.py full                # CSI full FOV + USB
  python3 record_dual.py --out=/mnt/ssd      # save to SSD (recommended)
  python3 record_dual.py --fmt=npy           # .npy: fastest to write, larger files
  python3 record_dual.py ram                 # record to RAM first, write to disk after stop
  python3 record_dual.py ram --pre=5         # + the 5 s BEFORE the record key are kept too
  python3 record_dual.py nousb               # CSI only      | nocsi = USB only
  python3 record_dual.py --usb=/dev/video8 cam1

Keys (in the preview window):
  SPACE / g = start / stop recording (each start = new folder)
  e         = lock / unlock CSI camera AE/AWB (USB cannot be locked)
  q / ESC   = quit (use this, NOT Ctrl+C, so the USB camera does not get stuck)
"""
import os
import sys
import time

import cv2
import numpy as np

import run_sensor_fast as rsf


# Recorder / finish_recording / mem_available_bytes now live in run_sensor_fast.py
# (so run_sensor_fast.py can record on its own without depending on this file).
Recorder = rsf.Recorder
finish_recording = rsf.finish_recording
mem_available_bytes = rsf.mem_available_bytes


def main():
    args = [a.lower() for a in sys.argv[1:]]
    use_csi, use_usb = "nocsi" not in args, "nousb" not in args
    if not (use_csi or use_usb):
        sys.exit("[ERROR] nocsi and nousb cannot be used together.")
    pk = ("full" if "full" in args else "fast" if "fast" in args else
          "quality" if "quality" in args else "turbo")
    cam_id = 1 if "cam1" in args else 0
    fmt, out_root, usb_dev = "png", ".", None
    use_ram, pre_s = "ram" in args, 0.0
    for a in sys.argv[1:]:
        al = a.lower()
        if al.startswith("--pre="):
            pre_s = float(al.split("=", 1)[1])
        elif al.startswith("--fmt="):
            fmt = al.split("=", 1)[1]
        elif al.startswith("--out="):
            out_root = a.split("=", 1)[1]
        elif al.startswith("--usb="):
            usb_dev = a.split("=", 1)[1]
    if fmt not in ("png", "npy"):
        sys.exit("[ERROR] --fmt must be png or npy")
    os.makedirs(out_root, exist_ok=True)

    recs, meta_cams, csi_reader = [], {}, None
    if use_csi:
        if rsf.Picamera2 is None:
            sys.exit("[ERROR] picamera2 is missing. Use 'nocsi' for USB only.")
        prof = rsf.PROFILES[pk]
        csi = rsf.csi_cameras()
        if cam_id >= len(csi):
            sys.exit(f"[ERROR] CSI camera cam{cam_id} does not exist. Use 'nocsi' for USB only.")
        picam2 = rsf.open_camera(csi[cam_id], prof["cw"], prof["ch"], prof["fr"], prof.get("raw"))
        csi_reader = rsf.CameraReader(picam2, name="csi"); csi_reader.start()
        recs.append(Recorder("csi", csi_reader, True, fmt, False, ram=use_ram, pre_s=pre_s))
        meta_cams["csi"] = dict(profile=pk, size=[prof["cw"], prof["ch"]],
                                fps_target=prof["fr"], cam_index=cam_id)
        print(f"[INFO] CSI cam{cam_id} {pk} {prof['cw']}x{prof['ch']} @ {prof['fr']} fps")
    if use_usb:
        dev = rsf.resolve_usb(usb_dev)
        prof = rsf.PROFILES["usb"]
        usb_reader = rsf.UsbCameraReader(dev, prof["cw"], prof["ch"], prof["fr"])
        usb_reader.start()
        recs.append(Recorder("usb", usb_reader, False, fmt, True, ram=use_ram, pre_s=pre_s))
        meta_cams["usb"] = dict(device=dev, size=list(usb_reader.size), fps_target=prof["fr"])
        print(f"[INFO] USB {dev} {usb_reader.size[0]}x{usb_reader.size[1]} @ {prof['fr']} fps")
    for r in recs:
        r.start()

    WIN = "record_dual  [SPACE/g = record | e = lock CSI AE | q = quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 400)
    folder, t_rec, ae_locked = None, None, False
    print(f"[INFO] format {fmt.upper()} -> {os.path.abspath(out_root)}  |  "
          f"mode {'RAM' if use_ram else 'disk'}  |  pre-trigger {pre_s:.1f}s  |  SPACE = start recording")

    def finish():
        finish_recording(folder, recs, meta_cams, fmt, ae_locked)

    try:
        while True:
            tiles = []
            for r in recs:
                f = r.latest
                tile = (cv2.resize(f, (480, 360)) if f is not None
                        else np.zeros((360, 480, 3), np.uint8))
                txt = f"{r.name_.upper()} {r.live_fps:.1f} fps"
                if r.recording:
                    txt += f"  REC {r.idx}  q={r.q.qsize()}"
                cv2.putText(tile, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(tile, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if r.recording else (0, 255, 0), 1)
                tiles.append(tile)
            panel = np.hstack(tiles)
            if t_rec is not None:
                cv2.putText(panel, f"REC {time.monotonic() - t_rec:5.1f}s", (panel.shape[1] - 150, 350),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(WIN, panel)

            k = cv2.waitKey(30) & 0xFF
            if k in (27, ord("q")):
                break
            elif k in (ord(" "), ord("g")):
                if t_rec is None:
                    folder = os.path.join(out_root, time.strftime("rec_%Y%m%d_%H%M%S"))
                    os.makedirs(folder, exist_ok=True)
                    for r in recs:
                        r.start_rec(folder)
                    t_rec = time.monotonic()
                    print(f"[REC] START -> {folder}/  (SPACE again to stop)")
                else:
                    finish(); t_rec = None
            elif k == ord("e") and csi_reader is not None:
                ae_locked = not ae_locked
                if ae_locked:
                    md = csi_reader.get_metadata()
                    csi_reader.set_controls(AeEnable=False, AwbEnable=False,
                                            ExposureTime=int(md.get("ExposureTime") or 8000),
                                            AnalogueGain=float(md.get("AnalogueGain") or 2.0))
                    print("[INFO] CSI AE/AWB locked")
                else:
                    csi_reader.set_controls(AeEnable=True, AwbEnable=True)
                    print("[INFO] AE/AWB CSI auto")
    finally:
        if t_rec is not None:
            finish()
        for r in recs:
            r.running = False
        for r in recs:
            r.join(timeout=2)
            r.reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
