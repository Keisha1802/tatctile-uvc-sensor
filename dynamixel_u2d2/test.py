from dynamixel_sdk import *

import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
BAUDRATE=57600
PROTOCOL_VERSION=2.0

port=PortHandler(DEVICENAME)
packet=PacketHandler(PROTOCOL_VERSION)

port.openPort()
port.setBaudRate(BAUDRATE)

ID=4

# torque ON
packet.write1ByteTxRx(port, ID, 64, 1)

# goal position
packet.write4ByteTxRx(port, ID, 116, 2300)

print("done")