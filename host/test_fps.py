"""
test_fps.py — measure the Pi Zero (UVC) webcam FPS (Linux: Pi 5 / Ubuntu, or Windows).

Usage:
  python3 test_fps.py                 # Linux: first "Tactile UVC Cam" (by-id); Windows: index 1
  python3 test_fps.py camera2         # Linux: Zero by hostname
  python3 test_fps.py /dev/video2     # Linux: explicit device
  python test_fps.py 2                # camera index (Windows: if 1 is the laptop camera)
  python3 test_fps.py camera1 MJPG    # test the MJPEG format (default YUY2)

Each resolution is measured for 5 s. A small preview window appears so you can
confirm the image really comes from the Zero (press q to skip).
"""
import sys
import time
import cv2

import glob
import os

FMT = (sys.argv[2] if len(sys.argv) > 2 else "YUY2").upper()
arg = sys.argv[1] if len(sys.argv) > 1 else None
if os.name == "nt":                                   # Windows: DirectShow, by index
    idx, API = int(arg) if arg else 1, cv2.CAP_DSHOW
else:                                                 # Linux: V4L2, by-id path / hostname
    API = cv2.CAP_V4L2
    if arg and arg.isdigit():
        idx = int(arg)
    elif arg and arg.startswith("/dev/"):
        idx = arg
    else:
        pat = f"/dev/v4l/by-id/usb-{arg}_Tactile*video-index0" if arg else               "/dev/v4l/by-id/*Tactile*video-index0"
        found = sorted(glob.glob(pat))
        if not found:
            sys.exit(f"[ERROR] no Tactile UVC Cam found ({pat}). Check: ls /dev/v4l/by-id/")
        idx = found[0]
    print(f"[INFO] device {idx}")
TESTS = [(320, 240), (640, 480)]


def fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


for w, h in TESTS:
    cap = cv2.VideoCapture(idx, API)
    if not cap.isOpened():
        sys.exit(f"[ERROR] camera index {idx} cannot be opened. Try: python test_fps.py 2")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FMT))   # YUY2 = uncompressed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FMT))   # DirectShow: set again after the size

    for _ in range(10):          # warm-up
        cap.read()

    n, t0, frame = 0, time.time(), None
    while time.time() - t0 < 5:
        ok, f = cap.read()
        if ok:
            n += 1
            frame = f
            cv2.imshow(f"Zero {w}x{h} (q = skip)", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    dt = time.time() - t0
    got = f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "-"
    print(f"requested {w}x{h} {FMT} -> got {got}, format {fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))}: "
          f"{n / dt:.1f} fps")
    cap.release()
    cv2.destroyAllWindows()
