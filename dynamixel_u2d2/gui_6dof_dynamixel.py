from dynamixel_sdk import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
import sys


# =========================
# DYNAMIXEL CONFIG
# =========================

import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB1")
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

IDS = [1, 2, 3, 4, 5, 6]

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132


# =========================
# CONNECT
# =========================

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Failed open port")
    exit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Failed baudrate")
    exit()

print("Dynamixel Connected")


# =========================
# FUNCTIONS
# =========================

def torque(enable):

    for dxl_id in IDS:
        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            enable
        )


def move_all():

    groupSyncWrite = GroupSyncWrite(
        portHandler,
        packetHandler,
        ADDR_GOAL_POSITION,
        4
    )

    for dxl_id, slider in zip(IDS, sliders):

        pos = slider.value()

        param = [
            DXL_LOBYTE(DXL_LOWORD(pos)),
            DXL_HIBYTE(DXL_LOWORD(pos)),
            DXL_LOBYTE(DXL_HIWORD(pos)),
            DXL_HIBYTE(DXL_HIWORD(pos))
        ]

        groupSyncWrite.addParam(
            dxl_id,
            param
        )


    result = groupSyncWrite.txPacket()

    if result == COMM_SUCCESS:
        print("Move sent")
    else:
        print("Sync write error")

    groupSyncWrite.clearParam()



def read_position():

    for i, dxl_id in enumerate(IDS):

        pos, result, error = packetHandler.read4ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_PRESENT_POSITION
        )

        if result == COMM_SUCCESS:
            position_label[i].setText(
                f"Joint {i+1} (ID {dxl_id}) : {pos}"
            )



# =========================
# GUI
# =========================

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle(
    "6 DOF Dynamixel Controller"
)

window.resize(600,700)


layout = QVBoxLayout()


title = QLabel(
    "XC430 + XC330 6 Joint Controller"
)

title.setStyleSheet(
    "font-size:18px"
)

layout.addWidget(title)



# Torque button

btn_on = QPushButton("TORQUE ON")
btn_on.clicked.connect(
    lambda: torque(1)
)

btn_off = QPushButton("TORQUE OFF")
btn_off.clicked.connect(
    lambda: torque(0)
)

layout.addWidget(btn_on)
layout.addWidget(btn_off)



# Slider

sliders = []

for i, dxl_id in enumerate(IDS):

    label = QLabel(
        f"Joint {i+1}  (ID {dxl_id})"
    )

    slider = QSlider()

    slider.setOrientation(1)

    slider.setMinimum(0)
    slider.setMaximum(4095)
    slider.setValue(2048)

    slider.setTickInterval(512)

    layout.addWidget(label)
    layout.addWidget(slider)

    sliders.append(slider)



# Move button

move = QPushButton("MOVE ALL JOINT")

move.clicked.connect(
    move_all
)

layout.addWidget(move)



# Encoder feedback

position_label = []

for i, dxl_id in enumerate(IDS):

    label = QLabel(
        f"Joint {i+1} (ID {dxl_id}) : ---"
    )

    position_label.append(label)

    layout.addWidget(label)



# Update encoder

timer = QTimer()

timer.timeout.connect(
    read_position
)

timer.start(200)



window.setLayout(layout)

window.show()

sys.exit(app.exec_())
