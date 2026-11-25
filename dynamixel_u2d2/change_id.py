from dynamixel_sdk import *

import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
BAUDRATE = 57600

OLD_ID = 2
NEW_ID = 3

PROTOCOL_VERSION = 2.0
ADDR_ID = 7

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(
    portHandler,
    OLD_ID,
    ADDR_ID,
    NEW_ID
)

if dxl_comm_result == COMM_SUCCESS:
    print("ID changed from", OLD_ID, "to", NEW_ID)
else:
    print("Failed")

portHandler.closePort()
