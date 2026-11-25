"""
MOUNTING HELPER — TAHAN FLEX 4,5,6 DI POSISI 1700
==================================================
Untuk pasang ulang jari setelah dibongkar:
1. Tutup GUI dulu (port cuma bisa dipakai 1 program)
2. Jalankan: python3 hold_1700.py
3. Script: reset Homing Offset -> 0, drive servo ke 1700, TAHAN
4. Pasang jari pada horn di ORIENTASI TERBUKA, kencangkan baut
5. Tekan ENTER untuk selesai (torque tetap ON supaya tidak geser)

Setelah itu langsung pakai GUI: 1700 = terbuka, 2300 = menutup.
"""

from dynamixel_sdk import *
import time

DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 57600
FLEX_IDS = [4, 5, 6]
TARGET = 1700

ADDR_HOMING_OFFSET        = 20
ADDR_OPERATING_MODE       = 11
ADDR_TORQUE_ENABLE        = 64
ADDR_GOAL_CURRENT         = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY     = 112
ADDR_GOAL_POSITION        = 116
ADDR_PRESENT_POSITION     = 132

port = PortHandler(DEVICENAME)
packet = PacketHandler(2.0)

if not (port.openPort() and port.setBaudRate(BAUDRATE)):
    print("!! Gagal buka port"); exit(1)


def read_pos(i):
    pos, res, _ = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
    return pos if res == COMM_SUCCESS else None


print("PASTIKAN JARI 4,5,6 SUDAH DILEPAS DARI HORN!")
input("Tekan ENTER untuk lanjut... ")

for i in FLEX_IDS:
    # torque OFF dulu (syarat tulis EEPROM)
    packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
    # reset Homing Offset ke 0 (hapus sisa kalibrasi software sebelumnya)
    packet.write4ByteTxRx(port, i, ADDR_HOMING_OFFSET, 0)
    # mode 5 + arus dibatasi (aman walau ada yang nyangkut)
    packet.write1ByteTxRx(port, i, ADDR_OPERATING_MODE, 5)
    packet.write2ByteTxRx(port, i, ADDR_GOAL_CURRENT, 200)
    packet.write4ByteTxRx(port, i, ADDR_PROFILE_ACCELERATION, 8)
    packet.write4ByteTxRx(port, i, ADDR_PROFILE_VELOCITY, 40)
    # torque ON, kirim ke 1700
    packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 1)
    packet.write4ByteTxRx(port, i, ADDR_GOAL_POSITION, TARGET)
    print(f"ID {i}: offset direset, menuju {TARGET}...")

time.sleep(3)   # tunggu semua sampai

print()
ok_all = True
for i in FLEX_IDS:
    pos = read_pos(i)
    ok = pos is not None and abs(pos - TARGET) <= 10
    ok_all = ok_all and ok
    print(f"ID {i}: pos = {pos}  {'✅' if ok else '!! belum sampai / cek koneksi'}")

print()
if ok_all:
    print("Semua servo DITAHAN di 1700.")
    print(">>> PASANG JARI PADA ORIENTASI TERBUKA, kencangkan baut <<<")
else:
    print("!! Ada servo yang belum di 1700 — cek dulu sebelum pasang jari.")

input("Tekan ENTER kalau sudah selesai pasang (torque akan tetap ON)... ")
print("Selesai. Tutup script ini lalu buka GUI — kalibrasi 1700-2300 kini valid.")
port.closePort()
