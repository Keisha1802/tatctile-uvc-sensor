"""
KALIBRASI FLEX ID 4,5,6 -> POSISI TERBUKA = 1700
=================================================
Method: Homing Offset (addr 20). Shifts the encoder zero in
software, WITHOUT taking the mechanics apart.

CARA PAKAI:
1. Make sure the gripper GUI is CLOSED (the port cannot be shared by 2 programs)
2. Run: python3 calibrate1700.py
3. The script switches the torque of all motors off
4. MOVE ALL THREE FINGERS FULLY OPEN BY HAND
5. Press ENTER -> the script computes & writes the offset, then verifies

After success the GUI can use FLEX_LIMITS 1700-2300 for every finger.
The offset is stored in EEPROM: permanent, even without power.
"""

from dynamixel_sdk import *
import time

import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
BAUDRATE = 57600
FLEX_IDS = [4, 5, 6]
TARGET = 1700              # wanted open position

ADDR_HOMING_OFFSET   = 20   # EEPROM, 4 byte, signed
ADDR_TORQUE_ENABLE   = 64
ADDR_PRESENT_POSITION = 132

port = PortHandler(DEVICENAME)
packet = PacketHandler(2.0)

if not (port.openPort() and port.setBaudRate(BAUDRATE)):
    print("!! Could not open the port"); exit(1)


def s32(v):
    return v - 4294967296 if v > 2147483647 else v


def read_pos(i):
    pos, res, _ = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
    return s32(pos) if res == COMM_SUCCESS else None


def read_offset(i):
    off, res, _ = packet.read4ByteTxRx(port, i, ADDR_HOMING_OFFSET)
    return s32(off) if res == COMM_SUCCESS else None


# 1. Torque OFF (wajib: Homing Offset = EEPROM, tolak ditulis saat torque ON)
for i in FLEX_IDS:
    packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
print("Torque OFF on all flex servos.")
print()
print(">>> MOVE FINGERS A, B, C FULLY OPEN BY HAND <<<")
input(">>> then press ENTER... ")

print()
for i in FLEX_IDS:
    pos = read_pos(i)
    old = read_offset(i)
    if pos is None or old is None:
        print(f"ID {i}: !! read failed, skipped"); continue

    # position read = raw_encoder + old_offset
    # target        = raw_encoder + offset_baru
    # offset_baru   = offset_lama + (TARGET - posisi_terbaca)
    new = old + (TARGET - pos)
    print(f"ID {i}: pos={pos}  offset {old} -> {new}", end="  ")

    res, err = packet.write4ByteTxRx(port, i, ADDR_HOMING_OFFSET,
                                     new & 0xFFFFFFFF)
    if res != COMM_SUCCESS or err != 0:
        print(f"!! TOLAK (res={res} err={err})")
        continue
    time.sleep(0.1)

    check = read_pos(i)
    ok = check is not None and abs(check - TARGET) <= 5
    print(f"verifikasi: {check} {'OK ✅' if ok else '!! FAILED - offset not accepted'}")

print()
print("Done. Close the fingers slowly by hand and read again")
print("to see the real close values, or go straight to the GUI")
print("with FLEX_LIMITS = (1700, 2300).")
port.closePort()
