#!/usr/bin/env python3
"""
KALIBRASI ULANG ENCODER GRIPPER (semua 6 servo)
===============================================
Homing Offset (EEPROM addr 20) digeser supaya pembacaan encoder punya arti tetap:

    ROTASI ID 1, 2, 3 : posisi LURUS / home        -> 2048
    FLEKSI ID 4, 5, 6 : jari TERBUKA penuh          -> 1700
    lalu jari DITUTUP penuh dengan tangan          -> batas CLOSE diukur (dikurangi margin)

Hasil (offset di servo, permanen) + batas open / close per jari -> calib.json, dipakai otomatis oleh
chromasense_gui.py (FLEX_LIMITS / ROT_HOME). Tidak perlu bongkar mekanik.

  ./run_gripper.sh calibrate_encoders.py            semua (home + open, lalu close)
  ./run_gripper.sh calibrate_encoders.py --only 4,5 hanya ID tertentu
  ./run_gripper.sh calibrate_encoders.py --read     hanya tampilkan posisi + offset (tidak menulis)

Tutup GUI dulu (port hanya untuk satu program). Torque dimatikan selama kalibrasi: jari bebas digerakkan
dengan tangan. Pasangan jari: ROT 1 + FLEX 4, ROT 2 + FLEX 5, ROT 3 + FLEX 6.
"""
import json
import os
import sys
import threading
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
BAUDRATE = 57600
ROT_IDS, FLEX_IDS = [1, 2, 3], [4, 5, 6]
ROT_TARGET, FLEX_OPEN = 2048, 1700
CLOSE_MARGIN = 40                 # ticks short of the measured hard stop (3.5 deg): never push into it
ADDR_HOMING_OFFSET, ADDR_TORQUE, ADDR_PRESENT_POSITION = 20, 64, 132
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")
NAME = {1: "ROT jari 1", 2: "ROT jari 2", 3: "ROT jari 3", 4: "FLEX jari 1", 5: "FLEX jari 2", 6: "FLEX jari 3"}


def s32(v):
    return v - 4294967296 if v > 2147483647 else v


class Bus:
    def __init__(self):
        self.port, self.pk = PortHandler(DEVICENAME), PacketHandler(2.0)
        try:
            ok = self.port.openPort() and self.port.setBaudRate(BAUDRATE)
        except Exception as e:
            ok = False
            print(f"!! {DEVICENAME}: {e}")
        if not ok:
            sys.exit(f"!! tidak bisa membuka {DEVICENAME} (GUI masih jalan? U2D2 tercolok?)")
        self.lock = threading.Lock()

    def pos(self, i):
        with self.lock:
            v, res, _ = self.pk.read4ByteTxRx(self.port, i, ADDR_PRESENT_POSITION)
        return s32(v) if res == COMM_SUCCESS else None

    def offset(self, i):
        with self.lock:
            v, res, _ = self.pk.read4ByteTxRx(self.port, i, ADDR_HOMING_OFFSET)
        return s32(v) if res == COMM_SUCCESS else None

    def torque(self, i, on):
        with self.lock:
            res, err = self.pk.write1ByteTxRx(self.port, i, ADDR_TORQUE, 1 if on else 0)
        return res == COMM_SUCCESS and not err

    def set_offset(self, i, v):
        with self.lock:
            res, err = self.pk.write4ByteTxRx(self.port, i, ADDR_HOMING_OFFSET, int(v) & 0xFFFFFFFF)
        if res != COMM_SUCCESS or err:
            print(f"   !! ID {i}: offset DITOLAK ({self.pk.getTxRxResult(res) if res != COMM_SUCCESS else self.pk.getRxPacketError(err)})")
            return False
        return True


def live(bus, ids, stop):
    """one status line, refreshed, while the user moves the fingers"""
    import shutil
    while not stop.is_set():
        vals = [bus.pos(i) for i in ids]
        line = " ".join(f"{i}:{'ERR' if v is None else v}" for i, v in zip(ids, vals))
        w = max(20, shutil.get_terminal_size((80, 20)).columns - 1)
        line = ("posisi " + line)[:w]                     # never longer than the terminal: \r stays on one line
        print("\r" + line.ljust(w), end="", flush=True)
        time.sleep(0.3)
    print()


def wait(bus, ids, prompt):
    stop = threading.Event()
    th = threading.Thread(target=live, args=(bus, ids, stop), daemon=True)
    th.start()
    try:
        input()
    finally:
        stop.set()
        th.join()


def main():
    args = sys.argv[1:]
    ids = ROT_IDS + FLEX_IDS
    if "--only" in args:
        ids = [int(x) for x in args[args.index("--only") + 1].split(",")]
    bus = Bus()
    missing = [i for i in ids if bus.pos(i) is None]
    if missing:
        sys.exit(f"!! ID {missing} tidak menjawab -> ./run_gripper.sh setup_ids.py check")
    if "--read" in args:
        for i in ids:
            print(f"ID {i} ({NAME.get(i, '')}): posisi {bus.pos(i)}  homing offset {bus.offset(i)}")
        return

    print(__doc__.split("Hasil")[0])
    for i in ids:
        bus.torque(i, False)
    print("Torque OFF.\n")
    print(">>> LANGKAH 1: putar ROTASI (ID 1-3) ke posisi LURUS / home, buka semua jari PENUH <<<")
    print("    (angka ikut berubah saat digerakkan; tekan ENTER kalau sudah pas - JANGAN Ctrl+C,")
    print("     Ctrl+C membatalkan tanpa menulis apa-apa)")
    wait(bus, ids, "")

    rep = {}
    for i in ids:
        target = ROT_TARGET if i in ROT_IDS else FLEX_OPEN
        pos, old = bus.pos(i), bus.offset(i)
        if pos is None or old is None:
            print(f"ID {i}: !! gagal baca, dilewati")
            continue
        raw = pos - old                                   # what the encoder itself reads
        new = old + (target - pos)
        ok = bus.set_offset(i, new)
        time.sleep(0.1)
        chk = bus.pos(i)
        good = ok and chk is not None and abs(chk - target) <= 5
        rep[i] = dict(raw=raw, offset=new if ok else old, home=target)
        print(f"ID {i} ({NAME.get(i, '')}): baca {pos} -> {chk}  (offset {old} -> {new})  "
              f"{'OK' if good else '!! verifikasi gagal'}")

    flex = [i for i in ids if i in FLEX_IDS]
    limits = {}
    if flex:
        print("\n>>> LANGKAH 2: TUTUP tiap jari (ID " + ",".join(map(str, flex)) + ") PENUH dengan tangan, "
              "pelan, sampai mentok <<<")
        print("    (boleh satu per satu; posisi paling jauh tiap jari yang dipakai. ENTER kalau selesai)")
        best = {i: FLEX_OPEN for i in flex}
        stop = threading.Event()

        def track():
            while not stop.is_set():
                for i in flex:
                    p = bus.pos(i)
                    if p is not None and abs(p - FLEX_OPEN) > abs(best[i] - FLEX_OPEN):
                        best[i] = p
                time.sleep(0.05)
        th = threading.Thread(target=track, daemon=True)
        th.start()
        wait(bus, flex, "")
        stop.set()
        th.join()
        for i in flex:
            d = best[i] - FLEX_OPEN
            if abs(d) < 150:
                print(f"ID {i}: !! jari hampir tidak ditutup ({best[i]}) -> batas lama dipakai")
                continue
            close = int(best[i] - CLOSE_MARGIN * (1 if d > 0 else -1))
            limits[i] = [FLEX_OPEN, close]
            raw_o = rep.get(i, {}).get("raw")
            wrap = raw_o is not None and not (0 <= raw_o + d <= 4095)
            print(f"ID {i} ({NAME[i]}): open {FLEX_OPEN}  mentok {best[i]}  -> CLOSE {close} "
                  f"({abs(close - FLEX_OPEN) * 360 / 4096:.0f} deg)"
                  + ("\n   !! rentang jari ini MELEWATI titik 0/4095 encoder: setelah power mati bacaan bisa"
                     " loncat 4096. Pasang ulang horn dengan hold_1700.py" if wrap else ""))

    old = {}
    try:
        old = json.load(open(CALIB))
    except (OSError, ValueError):
        pass
    old.setdefault("flex_limits", {}).update({str(k): v for k, v in limits.items()})
    old.setdefault("rot_home", {}).update({str(i): ROT_TARGET for i in ids if i in ROT_IDS})
    old["offsets"] = {**old.get("offsets", {}), **{str(i): r["offset"] for i, r in rep.items()}}
    old["date"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(old, open(CALIB, "w"), indent=1)
    print(f"\nTersimpan: {CALIB}  (chromasense_gui.py memakai batas ini)")
    print("Torque tetap OFF. Buka GUI: ./run_gripper.sh")
    bus.port.closePort()


if __name__ == "__main__":
    main()
