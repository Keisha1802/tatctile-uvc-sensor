"""
KALIBRASI FLEX ID 4,5,6 -> POSISI TERBUKA = 1700
=================================================
Metode: Homing Offset (addr 20). Menggeser titik nol encoder via
software, TANPA bongkar mekanik.

CARA PAKAI:
1. Pastikan GUI gripper DITUTUP (port tidak boleh dipakai 2 program)
2. Jalankan: python3 calibrate_1700.py
3. Script akan matikan torque semua motor
4. POSISIKAN KETIGA JARI TERBUKA PENUH DENGAN TANGAN
5. Tekan ENTER -> script hitung & tulis offset, lalu verifikasi

Setelah sukses, GUI bisa pakai FLEX_LIMITS 1700-2300 untuk semua jari.
Offset tersimpan di EEPROM: permanen walau power dicabut.
"""

from dynamixel_sdk import *
import time

import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
BAUDRATE = 57600
FLEX_IDS = [4, 5, 6]
TARGET = 1700              # posisi terbuka yang diinginkan

ADDR_HOMING_OFFSET   = 20   # EEPROM, 4 byte, signed
ADDR_TORQUE_ENABLE   = 64
ADDR_PRESENT_POSITION = 132

port = PortHandler(DEVICENAME)
packet = PacketHandler(2.0)

if not (port.openPort() and port.setBaudRate(BAUDRATE)):
    print("!! Gagal buka port"); exit(1)


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
print("Torque OFF semua flex.")
print()
print(">>> POSISIKAN JARI A, B, C TERBUKA PENUH DENGAN TANGAN <<<")
input(">>> lalu tekan ENTER... ")

print()
for i in FLEX_IDS:
    pos = read_pos(i)
    old = read_offset(i)
    if pos is None or old is None:
        print(f"ID {i}: !! gagal baca, skip"); continue

    # posisi terbaca = raw_encoder + offset_lama
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
    print(f"verifikasi: {check} {'OK ✅' if ok else '!! GAGAL — offset tidak diterima'}")

print()
print("Selesai. Tutup jari pelan dengan tangan & jalankan lagi 'baca saja'")
print("kalau mau lihat nilai close aslinya, atau langsung pakai GUI")
print("dengan FLEX_LIMITS = (1700, 2300).")
port.closePort()
