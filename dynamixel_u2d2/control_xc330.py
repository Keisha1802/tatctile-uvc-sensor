from dynamixel_sdk import *
import time


# ==========================
# CONFIGURATION
# ==========================

import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

DXL_IDS = [1, 2, 3]

# Control table XC330 (Protocol 2.0)
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


# ==========================
# INITIALIZE
# ==========================

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)


# Open port
if portHandler.openPort():
    print("Port opened")
else:
    print("Failed to open port")
    quit()


# Set baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Baudrate set:", BAUDRATE)
else:
    print("Failed baudrate")
    quit()


# ==========================
# ENABLE TORQUE
# ==========================

def enable_torque():

    for dxl_id in DXL_IDS:

        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_ENABLE
        )

        if dxl_comm_result == COMM_SUCCESS:
            print("Torque ON ID:", dxl_id)
        else:
            print("Torque error ID:", dxl_id)



# ==========================
# READ POSITION
# ==========================

def read_position(dxl_id):

    position, result, error = packetHandler.read4ByteTxRx(
        portHandler,
        dxl_id,
        ADDR_PRESENT_POSITION
    )

    if result == COMM_SUCCESS:
        return position
    else:
        return None



# ==========================
# SYNC WRITE POSITION
# ==========================

def set_position(goal_positions):

    groupSyncWrite = GroupSyncWrite(
        portHandler,
        packetHandler,
        ADDR_GOAL_POSITION,
        4
    )


    for dxl_id, position in goal_positions.items():

        param_goal_position = [
            DXL_LOBYTE(DXL_LOWORD(position)),
            DXL_HIBYTE(DXL_LOWORD(position)),
            DXL_LOBYTE(DXL_HIWORD(position)),
            DXL_HIBYTE(DXL_HIWORD(position))
        ]

        groupSyncWrite.addParam(
            dxl_id,
            param_goal_position
        )


    result = groupSyncWrite.txPacket()

    if result == COMM_SUCCESS:
        print("Goal position sent")

    else:
        print("Sync write failed")


    groupSyncWrite.clearParam()



# ==========================
# MAIN PROGRAM
# ==========================


enable_torque()


time.sleep(1)


# Contoh posisi target
# range XC330: 0 - 4095

target = {
    1: 2048,
    2: 2048,
    3: 2048,
    4: 2048,
    5: 2048,
    6: 2048
}


set_position(target)


print("\nReading position...\n")


try:

    while True:

        for dxl_id in DXL_IDS:

            pos = read_position(dxl_id)

            print(
                "ID",
                dxl_id,
                "Position:",
                pos
            )

        print("----------------")

        time.sleep(0.2)



except KeyboardInterrupt:

    print("\nStopping...")


    # Disable torque
    for dxl_id in DXL_IDS:

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            TORQUE_DISABLE
        )


    portHandler.closePort()

    print("Torque OFF")
