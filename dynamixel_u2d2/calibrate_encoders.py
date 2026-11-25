#!/usr/bin/env python3
"""
GRIPPER ENCODER RECALIBRATION, TERMINAL VERSION (all 6 servos) - the GUI version is calibrate_gui.py
===================================================================================================
The Homing Offset (EEPROM addr 20) is shifted so the encoder readings have a fixed meaning:

    ROTATION ID 1, 2, 3 : STRAIGHT / home            -> 2048
    FLEXION  ID 4, 5, 6 : finger fully OPEN          -> 1700
    then each finger CLOSED fully by hand          -> the CLOSE limit is measured (minus a margin)

Result (offsets in the servos, permanent) + open / close limits per finger -> calib.json, used by
chromasense_gui.py (FLEX_LIMITS / ROT_HOME). No need to take the mechanics apart.

  ./run_gripper.sh calibrate_encoders.py            all (home + open, then close)
  ./run_gripper.sh calibrate_encoders.py --only 4,5 only these IDs
  ./run_gripper.sh calibrate_encoders.py --read     positions + offsets only (writes nothing)

Close the GUI first (one program per port). The torque is off during calibration: the fingers move freely
by hand. Finger pairs: ROT 1 + FLEX 4, ROT 2 + FLEX 5, ROT 3 + FLEX 6.
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
NAME = {1: "ROT finger 1", 2: "ROT finger 2", 3: "ROT finger 3", 4: "FLEX finger 1", 5: "FLEX finger 2", 6: "FLEX finger 3"}


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
            sys.exit(f"!! cannot open {DEVICENAME} (GUI still running? U2D2 plugged in?)")
        self.lock = threading.Lock()

    def _r4(self, i, addr):
        """None instead of a crash when the port drops out (another program on it / U2D2 unplugged)"""
        try:
            with self.lock:
                v, res, _ = self.pk.read4ByteTxRx(self.port, i, addr)
        except Exception as e:
            if not getattr(self, "_warned", False):
                self._warned = True
                print(f"!! port {DEVICENAME}: {e}\n   -> another program on this port? check: fuser -v {DEVICENAME}")
            return None
        return s32(v) if res == COMM_SUCCESS else None

    def pos(self, i):
        return self._r4(i, ADDR_PRESENT_POSITION)

    def offset(self, i):
        return self._r4(i, ADDR_HOMING_OFFSET)

    def hw_error(self, i):
        """Hardware Error Status (addr 70): 0 = fine. bit0 voltage, bit2 overheat, bit3 ENCODER, bit4 shock,
        bit5 overload. With an error the servo shuts down and may report a FROZEN position."""
        try:
            with self.lock:
                v, res, _ = self.pk.read1ByteTxRx(self.port, i, 70)
        except Exception:
            return None
        return v if res == COMM_SUCCESS else None

    def reboot(self, i):
        with self.lock:
            res, err = self.pk.reboot(self.port, i)
        time.sleep(0.6)
        return res == COMM_SUCCESS

    def torque(self, i, on):
        with self.lock:
            res, err = self.pk.write1ByteTxRx(self.port, i, ADDR_TORQUE, 1 if on else 0)
        return res == COMM_SUCCESS and not err

    def set_offset(self, i, v):
        with self.lock:
            res, err = self.pk.write4ByteTxRx(self.port, i, ADDR_HOMING_OFFSET, int(v) & 0xFFFFFFFF)
        if res != COMM_SUCCESS or err:
            print(f"   !! ID {i}: offset REJECTED ({self.pk.getTxRxResult(res) if res != COMM_SUCCESS else self.pk.getRxPacketError(err)})")
            return False
        return True


def live(bus, ids, stop):
    """one status line, refreshed, while the user moves the fingers"""
    import shutil
    while not stop.is_set():
        vals = [bus.pos(i) for i in ids]
        line = " ".join(f"{i}:{'ERR' if v is None else v}" for i, v in zip(ids, vals))
        w = max(20, shutil.get_terminal_size((80, 20)).columns - 1)
        line = ("position " + line)[:w]                     # never longer than the terminal: \r stays on one line
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
        sys.exit(f"!! ID {missing} do not answer -> ./run_gripper.sh setup_ids.py check")
    if "--read" in args:
        for i in ids:
            print(f"ID {i} ({NAME.get(i, '')}): position {bus.pos(i)}  homing offset {bus.offset(i)}")
        return

    print(__doc__.split("Result")[0])
    for i in ids:
        bus.torque(i, False)
    print("Torque OFF.\n")
    print(">>> STEP 1: turn the ROTATIONS (ID 1-3) STRAIGHT / home, open every finger FULLY <<<")
    print("    (the numbers follow the motion; press ENTER when it is right - NOT Ctrl+C,")
    print("     Ctrl+C cancels without writing anything)")
    wait(bus, ids, "")

    rep = {}
    for i in ids:
        target = ROT_TARGET if i in ROT_IDS else FLEX_OPEN
        pos, old = bus.pos(i), bus.offset(i)
        if pos is None or old is None:
            print(f"ID {i}: !! read failed, skipped")
            continue
        raw = pos - old                                   # what the encoder itself reads
        new = old + (target - pos)
        ok = bus.set_offset(i, new)
        time.sleep(0.1)
        chk = bus.pos(i)
        good = ok and chk is not None and abs(chk - target) <= 5
        if not good and ok:
            bus.set_offset(i, old)                     # do not leave a wrong offset behind
            print(f"   ID {i}: offset put back to {old}; hardware error {bus.hw_error(i)} -> reboot this servo"
                  " (or power-cycle the servos) and repeat")
        rep[i] = dict(raw=raw, offset=new if ok else old, home=target)
        print(f"ID {i} ({NAME.get(i, '')}): read {pos} -> {chk}  (offset {old} -> {new})  "
              f"{'OK' if good else '!! verification failed'}")

    flex = [i for i in ids if i in FLEX_IDS]
    limits = {}
    if flex:
        print("\n>>> STEP 2: CLOSE every finger (ID " + ",".join(map(str, flex)) + ") FULLY by hand, "
              "slowly, to its stop <<<")
        print("    (one at a time is fine; the farthest position of each finger counts. ENTER when done)")
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
                print(f"ID {i}: !! finger hardly closed ({best[i]}) -> the old limit stays")
                continue
            close = int(best[i] - CLOSE_MARGIN * (1 if d > 0 else -1))
            limits[i] = [FLEX_OPEN, close]
            raw_o = rep.get(i, {}).get("raw")
            wrap = raw_o is not None and not (0 <= raw_o + d <= 4095)
            print(f"ID {i} ({NAME[i]}): open {FLEX_OPEN}  stop {best[i]}  -> CLOSE {close} "
                  f"({abs(close - FLEX_OPEN) * 360 / 4096:.0f} deg)"
                  + ("\n   !! this finger's range CROSSES the encoder's 0/4095 point: after a power cycle the reading"
                     " can jump 4096. Refit the horn with hold_1700.py" if wrap else ""))

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
    print(f"\nSaved: {CALIB}  (chromasense_gui.py uses these limits)")
    print("Torque stays OFF. Open the GUI: ./run_gripper.sh")
    bus.port.closePort()


if __name__ == "__main__":
    main()
