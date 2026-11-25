
from dynamixel_sdk import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
import sys


# ==========================
# DYNAMIXEL CONFIG
# ==========================

DEVICENAME = "/dev/ttyUSB1"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

IDS = [1, 2, 3]

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132


# ==========================
# CONNECT DYNAMIXEL
# ==========================

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort():
    print("Failed open port")
    quit()

portHandler.setBaudRate(BAUDRATE)

print("Dynamixel Connected")


# ==========================
# FUNCTIONS
# ==========================

def torque(state):

    for dxl_id in IDS:

        packetHandler.write1ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_TORQUE_ENABLE,
            state
        )


def move_servo():

    for dxl_id, slider in zip(IDS, sliders):

        pos = slider.value()

        packetHandler.write4ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_GOAL_POSITION,
            pos
        )


def read_position():

    for i, dxl_id in enumerate(IDS):

        pos, result, error = packetHandler.read4ByteTxRx(
            portHandler,
            dxl_id,
            ADDR_PRESENT_POSITION
        )

        if result == COMM_SUCCESS:
            labels[i].setText(
                f"ID {dxl_id} Position : {pos}"
            )



# ==========================
# GUI
# ==========================

app = QApplication(sys.argv)

window = QWidget()
window.setWindowTitle("XC330 Dynamixel Controller")
window.resize(500,400)


layout = QVBoxLayout()


title = QLabel(
    "Dynamixel XC330 - U2D2 Controller"
)

layout.addWidget(title)



# Torque Button

btn_on = QPushButton("Torque ON")
btn_on.clicked.connect(
    lambda: torque(1)
)

btn_off = QPushButton("Torque OFF")
btn_off.clicked.connect(
    lambda: torque(0)
)

layout.addWidget(btn_on)
layout.addWidget(btn_off)



# Position slider

sliders = []
labels = []


for dxl_id in IDS:

    label = QLabel(
        f"ID {dxl_id} Position"
    )

    slider = QSlider()

    slider.setOrientation(1)
    slider.setMinimum(0)
    slider.setMaximum(4095)
    slider.setValue(2048)


    layout.addWidget(label)
    layout.addWidget(slider)


    sliders.append(slider)



# Move button

move_btn = QPushButton(
    "MOVE"
)

move_btn.clicked.connect(
    move_servo
)

layout.addWidget(move_btn)



# Feedback encoder

for dxl_id in IDS:

    label = QLabel(
        f"ID {dxl_id} Position : ---"
    )

    labels.append(label)

    layout.addWidget(label)



# Timer realtime read

timer = QTimer()
timer.timeout.connect(read_position)
timer.start(200)



window.setLayout(layout)

window.show()


sys.exit(app.exec_())
