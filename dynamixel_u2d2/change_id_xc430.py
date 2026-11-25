from dynamixel_sdk import *

DEVICENAME = "/dev/ttyUSB1"
BAUDRATE = 57600

OLD_ID = 5
NEW_ID = 6

ADDR_ID = 7

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(2.0)

portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

# Disable torque dulu
packetHandler.write1ByteTxRx(
    portHandler,
    OLD_ID,
    64,
    0
)

# Change ID
result, error = packetHandler.write1ByteTxRx(
    portHandler,
    OLD_ID,
    ADDR_ID,
    NEW_ID
)

if result == COMM_SUCCESS and error == 0:
    print("ID changed:", OLD_ID, "->", NEW_ID)
else:
    print("Failed")
    print(packetHandler.getTxRxResult(result))
    print(packetHandler.getRxPacketError(error))

portHandler.closePort()
