from dynamixel_sdk import *

DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Failed to open port")
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set baudrate")
    quit()

print("Scanning Dynamixel...")

for dxl_id in range(0, 255):
    dxl_model, result, error = packetHandler.ping(portHandler, dxl_id)

    if result == COMM_SUCCESS:
        print("Found ID:", dxl_id, "Model:", dxl_model)

    elif error != 0:
        print("ID", dxl_id, "error:", error)

portHandler.closePort()
