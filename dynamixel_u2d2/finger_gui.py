from dynamixel_sdk import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer
import sys

# =========================
# DYNAMIXEL CONFIG
# =========================
DEVICENAME = "/dev/ttyUSB1"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

# Pasangan jari: (master, slave)
PAIRS = [(1, 4), (2, 5), (3, 6)]
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
            portHandler, dxl_id, ADDR_TORQUE_ENABLE, enable
        )

def pos_to_param(pos):
    return [
        DXL_LOBYTE(DXL_LOWORD(pos)),
        DXL_HIBYTE(DXL_LOWORD(pos)),
        DXL_LOBYTE(DXL_HIWORD(pos)),
        DXL_HIBYTE(DXL_HIWORD(pos))
    ]

def move_all():
    groupSyncWrite = GroupSyncWrite(
        portHandler, packetHandler, ADDR_GOAL_POSITION, 4
    )

    for (master_id, slave_id), slider, mirror_cb in zip(PAIRS, sliders, mirror_checks):
        pos = slider.value()

        # posisi slave: sama atau mirror
        if mirror_cb.isChecked():
            slave_pos = 4095 - pos
        else:
            slave_pos = pos

        groupSyncWrite.addParam(master_id, pos_to_param(pos))
        groupSyncWrite.addParam(slave_id, pos_to_param(slave_pos))

    result = groupSyncWrite.txPacket()
    if result == COMM_SUCCESS:
        print("Move sent")
    else:
        print("Sync write error")

    groupSyncWrite.clearParam()

def read_position():
    for i, dxl_id in enumerate(IDS):
        pos, result, error = packetHandler.read4ByteTxRx(
            portHandler, dxl_id, ADDR_PRESENT_POSITION
        )
        if result == COMM_SUCCESS:
            position_label[i].setText(
                f"ID {dxl_id} : {pos}"
            )

# =========================
# GUI
# =========================
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Finger Robot Controller (3 Pair)")
window.resize(600, 700)

layout = QVBoxLayout()

title = QLabel("Finger Robot : 1-4 / 2-5 / 3-6")
title.setStyleSheet("font-size:18px")
layout.addWidget(title)

# Torque button
btn_on = QPushButton("TORQUE ON")
btn_on.clicked.connect(lambda: torque(1))
btn_off = QPushButton("TORQUE OFF")
btn_off.clicked.connect(lambda: torque(0))
layout.addWidget(btn_on)
layout.addWidget(btn_off)

# Slider per pasangan
sliders = []
mirror_checks = []

for i, (m_id, s_id) in enumerate(PAIRS):
    label = QLabel(f"Finger {i+1}  (ID {m_id} + ID {s_id})")
    slider = QSlider()
    slider.setOrientation(1)  # horizontal
    slider.setMinimum(0)
    slider.setMaximum(4095)
    slider.setValue(2048)
    slider.setTickInterval(512)

    mirror_cb = QCheckBox(f"Mirror ID {s_id} (4095 - pos)")

    layout.addWidget(label)
    layout.addWidget(slider)
    layout.addWidget(mirror_cb)

    sliders.append(slider)
    mirror_checks.append(mirror_cb)

# Move button
move = QPushButton("MOVE ALL FINGER")
move.clicked.connect(move_all)
layout.addWidget(move)

# Encoder feedback
position_label = []
for i, dxl_id in enumerate(IDS):
    label = QLabel(f"ID {dxl_id} : ---")
    position_label.append(label)
    layout.addWidget(label)

# Update encoder
timer = QTimer()
timer.timeout.connect(read_position)
timer.start(200)

window.setLayout(layout)
window.show()
sys.exit(app.exec_())

