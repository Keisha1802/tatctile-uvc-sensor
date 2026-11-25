from dynamixel_sdk import *

import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

ADDR_PRESENT_POSITION = 132

IDS = [1, 2, 3]

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

while True:
    for dxl_id in IDS:
        position, result, error = packetHandler.read4ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_PRESENT_POSITION
        )

        if result == COMM_SUCCESS:
            print("ID", dxl_id, "Position:", position)
        else:
            print("ID", dxl_id, "Error")

    print("----------------")

portHandler.closePort()
