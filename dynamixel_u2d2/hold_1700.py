"""
MOUNTING HELPER - HOLD FLEX 4,5,6 AT POSITION 1700
==================================================
To refit a finger after taking it apart:
1. Close the GUI first (only 1 program can use the port)
2. Run: python3 hold_1700.py
3. The script resets Homing Offset -> 0, drives the servos to 1700 and HOLDS them
4. Fit the finger on the horn in the OPEN orientation, tighten the screws
5. Press ENTER when done (torque stays ON so nothing moves)

Then use the GUI: 1700 = open, 2300 = closed.
"""

from dynamixel_sdk import *
import time

import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
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
    print("!! Could not open the port"); exit(1)


def read_pos(i):
    pos, res, _ = packet.read4ByteTxRx(port, i, ADDR_PRESENT_POSITION)
    return pos if res == COMM_SUCCESS else None


print("MAKE SURE FINGERS 4,5,6 ARE OFF THEIR HORNS!")
input("Press ENTER to continue... ")

for i in FLEX_IDS:
    # torque OFF first (required to write EEPROM)
    packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 0)
    # reset Homing Offset to 0 (clears earlier software calibration)
    packet.write4ByteTxRx(port, i, ADDR_HOMING_OFFSET, 0)
    # mode 5 + limited current (safe even if something jams)
    packet.write1ByteTxRx(port, i, ADDR_OPERATING_MODE, 5)
    packet.write2ByteTxRx(port, i, ADDR_GOAL_CURRENT, 200)
    packet.write4ByteTxRx(port, i, ADDR_PROFILE_ACCELERATION, 8)
    packet.write4ByteTxRx(port, i, ADDR_PROFILE_VELOCITY, 40)
    # torque ON, kirim ke 1700
    packet.write1ByteTxRx(port, i, ADDR_TORQUE_ENABLE, 1)
    packet.write4ByteTxRx(port, i, ADDR_GOAL_POSITION, TARGET)
    print(f"ID {i}: offset reset, moving to {TARGET}...")

time.sleep(3)   # wait until all arrive

print()
ok_all = True
for i in FLEX_IDS:
    pos = read_pos(i)
    ok = pos is not None and abs(pos - TARGET) <= 10
    ok_all = ok_all and ok
    print(f"ID {i}: pos = {pos}  {'✅' if ok else '!! not there yet / check the connection'}")

print()
if ok_all:
    print("All servos HELD at 1700.")
    print(">>> FIT THE FINGERS IN THE OPEN ORIENTATION, tighten the screws <<<")
else:
    print("!! A servo is not at 1700 - check it before fitting a finger.")

input("Press ENTER when the fingers are fitted (torque stays ON)... ")
print("Done. Close this script and open the GUI - the 1700-2300 calibration is valid now.")
port.closePort()
