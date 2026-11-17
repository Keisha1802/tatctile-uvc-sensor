#!/usr/bin/env python3
"""
preview_uvc.py — test the Pi Zero camera (USB / UVC webcam) on the Pi 5.

Usage (on the Pi 5):
  python3 preview_uvc.py                 # auto-detect "Tactile UVC Cam", 640x480
  python3 preview_uvc.py 384 288         # other resolution (640x480, 384x288, 320x256)
  python3 preview_uvc.py 640 480 /dev/video8   # force a specific device

Keys:
  f = flow ON/OFF (frame-to-frame DIS optical flow, to check the speckle is tracked)
  r = reset flow baseline (flow against a reference frame; press with no contact)
  s = save frame (PNG)
  q / ESC = quit
"""
import glob
import sys
import time

import cv2
import numpy as np

W = int(sys.argv[1]) if len(sys.argv) > 1 else 640
H = int(sys.argv[2]) if len(sys.argv) > 2 else 480
DEV = sys.argv[3] if len(sys.argv) > 3 else None


def find_device():
    hits = sorted(glob.glob("/dev/v4l/by-id/*Tactile*video-index0"))
    if hits:
        return hits[0]
    hits = sorted(glob.glob("/dev/v4l/by-id/*video-index0"))
    if hits:
        print("[WARN] 'Tactile UVC Cam' not found by-id, using:", hits[0])
        return hits[0]
    sys.exit("[ERROR] USB camera not found. Check: v4l2-ctl --list-devices")


dev = DEV or find_device()
cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
if not cap.isOpened():
    sys.exit(f"[ERROR] cannot open {dev}")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_FPS, 60)
print(f"[INFO] {dev} -> {int(cap.get(3))}x{int(cap.get(4))}")
print("[INFO] f=flow  r=reset baseline  s=save  q=quit")

dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
show_flow, base_gray, prev_gray = False, None, None
n, t0, fps = 0, time.time(), 0.0
fails = 0

while True:
    ok, frame = cap.read()
    if not ok:
        fails += 1
        print(f"[WARN] failed to read frame ({fails})")
        if fails >= 5:
            sys.exit("[ERROR] no frames from the Zero. Check first with:\n"
                     f"  v4l2-ctl -d {dev} --set-fmt-video=width={W},height={H},pixelformat=YUYV "
                     "--stream-mmap --stream-count=50\n"
                     "If that also hangs: on the Zero run  sudo systemctl restart uvc-gadget")
        continue
    fails = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    n += 1
    if time.time() - t0 >= 1.0:
        fps, n, t0 = n / (time.time() - t0), 0, time.time()

    view = frame.copy()
    info = f"{frame.shape[1]}x{frame.shape[0]}  {fps:.1f} fps"

    if show_flow:
        ref = base_gray if base_gray is not None else prev_gray
        if ref is not None:
            flow = dis.calc(ref, gray, None)
            mag = cv2.magnitude(flow[..., 0], flow[..., 1])
            info += f"  |flow| mean={mag.mean():.2f} max={mag.max():.1f}px"
            hsv = np.zeros_like(frame)
            ang = np.arctan2(flow[..., 1], flow[..., 0])
            hsv[..., 0] = ((ang + np.pi) * 90 / np.pi).astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = np.clip(mag * 40, 0, 255).astype(np.uint8)
            view = np.hstack([view, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)])
        info += "  [baseline]" if base_gray is not None else "  [frame-to-frame]"
    prev_gray = gray

    cv2.rectangle(view, (0, 0), (view.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(view, info, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("Zero UVC", view)

    k = cv2.waitKey(1) & 0xFF
    if k in (27, ord("q")):
        break
    elif k == ord("f"):
        show_flow = not show_flow
    elif k == ord("r"):
        base_gray = gray.copy(); show_flow = True
        print("[INFO] baseline reset (make sure there is no contact)")
    elif k == ord("s"):
        name = time.strftime("uvc_%Y%m%d_%H%M%S.png")
        cv2.imwrite(name, frame); print("[INFO] saved:", name)

cap.release()
cv2.destroyAllWindows()
