from dynamixel_sdk import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt
import sys
import time

# ============================================================
# DYNAMIXEL CONFIG
# ============================================================
DEVICENAME = "/dev/ttyUSB0"
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0

ROT_IDS  = [1, 2, 3]
FLEX_IDS = [4, 5, 6]
IDS = ROT_IDS + FLEX_IDS

# --- Control table (X-series: XM430 / XC430 etc.) ---
ADDR_OPERATING_MODE       = 11    # EEPROM, torque must be OFF to change
ADDR_CURRENT_LIMIT        = 38    # EEPROM, 2 bytes
ADDR_TORQUE_ENABLE        = 64
ADDR_HW_ERROR_STATUS      = 70
ADDR_GOAL_CURRENT         = 102   # 2 bytes
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY     = 112
ADDR_GOAL_POSITION        = 116
ADDR_PRESENT_CURRENT      = 126   # 2 bytes, signed
ADDR_PRESENT_POSITION     = 132

OP_MODE_POSITION          = 3
OP_MODE_CURRENT_POSITION  = 5     # current-based position = anti-overload

# ============================================================
# PER-JOINT CALIBRATION  (FILL IN FROM SWEEP RESULTS!)
# ============================================================
# Format: id: (OPEN, CLOSE) -- OPEN = finger open, CLOSE = finger closed.
# IMPORTANT: CLOSE must NOT go past the mechanical hard stop.
# Original calibration was 1700 (open) -> 2300 (close). Setting CLOSE to
# 2400 made finger 4 push 100 ticks past its hard stop -> permanent stall.
FLEX_LIMITS = {
    4: (1700, 2400),
    5: (1700, 2400),
    6: (1700, 2400),
}

ROT_LIMITS = {
    1: (1400, 2900),
    2: (1400, 2900),
    3: (1400, 2900),
}
ROT_HOME = {1: 2048, 2: 2048, 3: 2048}

# ============================================================
# CURRENT PROTECTION (THE FIX FOR HW ERROR 32)
# ============================================================
# Goal Current = max current the servo may use in mode 5.
# XM430-W350: 1 unit = 2.69 mA. 200 units ~= 540 mA (medium grip).
# NOTE: Goal Current is REJECTED by the servo if it exceeds Current Limit
# (addr 38, default 1193 on XM430-W350). The old code wrote 2000, the servo
# silently refused it, and the current cap you thought you had never applied.
GOAL_CURRENT_FLEX = 200     # finger grip strength (raise if too weak)
GOAL_CURRENT_ROT  = 600     # base rotation needs more (holds gravity)
CURRENT_WARN      = 180     # UI warning threshold (units), keep < goal current

# Stall protection: if current stays >= STALL_CURRENT_RATIO * goal current
# while position barely moves for STALL_TIME seconds, pull the goal back to
# the present position so the servo stops pushing against an obstacle.
STALL_CURRENT_RATIO = 0.9
STALL_TIME          = 2.0     # seconds
STALL_POS_EPS       = 5       # ticks considered "not moving"

PROFILE_VELOCITY     = 50
PROFILE_ACCELERATION = 10

# ============================================================
# PRESETS  format: (name, {rot_id: val}, {flex_id: val})
# Flex preset values are clamped to FLEX_LIMITS per joint, so use the
# symbolic OPEN/CLOSE helpers to stay safe.
# ============================================================
def F_OPEN(i):  return FLEX_LIMITS[i][0]
def F_CLOSE(i): return FLEX_LIMITS[i][1]

PRESETS = [
    ("HOME OPEN",         {1: 2048, 2: 2048, 3: 2048}, {4: F_OPEN(4),  5: F_OPEN(5),  6: F_OPEN(6)}),
    ("HOME CLOSE",        {1: 2048, 2: 2048, 3: 2048}, {4: F_CLOSE(4), 5: F_CLOSE(5), 6: F_CLOSE(6)}),
    ("3F PARALLEL OPEN",  {1: 2754, 2: 1400, 3: 2048}, {4: F_OPEN(4),  5: F_OPEN(5),  6: F_OPEN(6)}),
    ("3F PARALLEL CLOSE", {1: 2754, 2: 1400, 3: 2048}, {4: F_CLOSE(4), 5: F_CLOSE(5), 6: F_CLOSE(6)}),
    ("PUSH",              {1: 2754, 2: 1400, 3: 2048}, {4: F_OPEN(4),  5: F_OPEN(5),  6: F_CLOSE(6)}),
    ("2F PARALLEL OPEN",  {1: 1700, 2: 2400, 3: 2048}, {4: F_OPEN(4),  5: F_OPEN(5),  6: F_OPEN(6)}),
    ("2F PARALLEL CLOSE", {1: 1700, 2: 2400, 3: 2048}, {4: F_CLOSE(4), 5: F_CLOSE(5), 6: F_OPEN(6)}),
]

# ============================================================
# CONNECT
# ============================================================
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

connected = portHandler.openPort() and portHandler.setBaudRate(BAUDRATE)
print("Dynamixel connected" if connected
      else "!! Port failed, GUI running in simulation mode")


# ============================================================
# HELPERS
# ============================================================
def clamp(v, lo, hi):
    # limits may be reversed (open > close), normalize first
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    return max(a, min(b, int(v)))


def flex_range(dxl_id):
    o, c = FLEX_LIMITS[dxl_id]
    return (min(o, c), max(o, c))


def pos_to_param(pos):
    pos = int(pos)
    return [
        DXL_LOBYTE(DXL_LOWORD(pos)),
        DXL_HIBYTE(DXL_LOWORD(pos)),
        DXL_LOBYTE(DXL_HIWORD(pos)),
        DXL_HIBYTE(DXL_HIWORD(pos)),
    ]


def signed16(v):
    return v - 65536 if v > 32767 else v


def checked_write(fn, dxl_id, addr, value, what):
    """Write and verify comm + servo error. The old code ignored both, which
    is how the invalid Goal Current write went unnoticed."""
    res, err = fn(portHandler, dxl_id, addr, value)
    if res != COMM_SUCCESS:
        print(f"[ID {dxl_id}] {what}: comm error "
              f"{packetHandler.getTxRxResult(res)}")
        return False
    if err != 0:
        print(f"[ID {dxl_id}] {what}: servo error "
              f"{packetHandler.getRxPacketError(err)}")
        return False
    return True


def set_slider_silent(slider, value):
    slider.blockSignals(True)
    slider.setValue(value)
    slider.blockSignals(False)
    slider.valueChanged.emit(slider.value())


# ============================================================
# SETUP: MODE 5 + CURRENT LIMIT + PROFILE
# ============================================================
def setup_motor(dxl_id):
    """Configure one motor: mode 5, goal current, smooth profile.
    Called at startup AND after reboot (RAM is wiped by reboot)."""
    if not connected:
        return
    # changing operating mode requires torque OFF
    checked_write(packetHandler.write1ByteTxRx, dxl_id,
                  ADDR_TORQUE_ENABLE, 0, "torque off")
    checked_write(packetHandler.write1ByteTxRx, dxl_id,
                  ADDR_OPERATING_MODE, OP_MODE_CURRENT_POSITION, "op mode")

    # Clamp goal current to the servo's Current Limit; values above it are
    # rejected by the servo and the cap silently never applies.
    limit, res, _ = packetHandler.read2ByteTxRx(portHandler, dxl_id,
                                                ADDR_CURRENT_LIMIT)
    gc = GOAL_CURRENT_FLEX if dxl_id in FLEX_IDS else GOAL_CURRENT_ROT
    if res == COMM_SUCCESS and limit > 0:
        gc = min(gc, limit)
    checked_write(packetHandler.write2ByteTxRx, dxl_id,
                  ADDR_GOAL_CURRENT, gc, "goal current")

    # smooth profile
    checked_write(packetHandler.write4ByteTxRx, dxl_id,
                  ADDR_PROFILE_ACCELERATION, PROFILE_ACCELERATION,
                  "profile accel")
    checked_write(packetHandler.write4ByteTxRx, dxl_id,
                  ADDR_PROFILE_VELOCITY, PROFILE_VELOCITY, "profile vel")


def setup_all_motors():
    for i in IDS:
        setup_motor(i)


def torque(enable):
    if not connected:
        status.setText("Simulation mode (port not connected)")
        return
    if enable:
        setup_all_motors()  # ensure mode 5 + goal current BEFORE torque ON
        for dxl_id in IDS:
            checked_write(packetHandler.write1ByteTxRx, dxl_id,
                          ADDR_TORQUE_ENABLE, 1, "torque on")
        # sync sliders to actual positions so the first MOVE doesn't jump
        sync_sliders_to_actual()
        status.setText("Torque ON — mode 5 (current-limited), overload-safe")
    else:
        for dxl_id in IDS:
            checked_write(packetHandler.write1ByteTxRx, dxl_id,
                          ADDR_TORQUE_ENABLE, 0, "torque off")
        status.setText("Torque OFF")


def sync_sliders_to_actual():
    """Set sliders = present positions so the first goal doesn't jerk."""
    if not connected:
        return
    for i, dxl_id in enumerate(ROT_IDS):
        pos, res, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id,
                                                  ADDR_PRESENT_POSITION)
        if res == COMM_SUCCESS:
            set_slider_silent(rot_sliders[i], clamp(pos, *ROT_LIMITS[dxl_id]))
    for i, dxl_id in enumerate(FLEX_IDS):
        pos, res, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id,
                                                  ADDR_PRESENT_POSITION)
        if res == COMM_SUCCESS:
            set_slider_silent(flex_sliders[i], clamp(pos, *flex_range(dxl_id)))


# ============================================================
# STALL PROTECTION (prevents the "ID 4 stalls forever" case)
# ============================================================
# state per motor: {"since": t or None, "last_pos": int}
stall_state = {i: {"since": None, "last_pos": None} for i in IDS}


def goal_current_of(dxl_id):
    return GOAL_CURRENT_FLEX if dxl_id in FLEX_IDS else GOAL_CURRENT_ROT


def check_stall(dxl_id, pos, cur):
    """If the motor sits at max current without moving for STALL_TIME,
    it's pressing against an obstacle / hard stop. Back the goal off to
    the present position so it stops pushing and cooling down."""
    st = stall_state[dxl_id]
    stalled_now = (cur >= STALL_CURRENT_RATIO * goal_current_of(dxl_id)
                   and st["last_pos"] is not None
                   and abs(pos - st["last_pos"]) <= STALL_POS_EPS)
    st["last_pos"] = pos

    if not stalled_now:
        st["since"] = None
        return

    if st["since"] is None:
        st["since"] = time.time()
        return

    if time.time() - st["since"] < STALL_TIME:
        return

    # Stalled too long: hold at present position instead of pushing on.
    st["since"] = None
    checked_write(packetHandler.write4ByteTxRx, dxl_id,
                  ADDR_GOAL_POSITION, pos, "stall backoff")
    if dxl_id in FLEX_IDS:
        s = flex_sliders[FLEX_IDS.index(dxl_id)]
        set_slider_silent(s, clamp(pos, *flex_range(dxl_id)))
    else:
        s = rot_sliders[ROT_IDS.index(dxl_id)]
        set_slider_silent(s, clamp(pos, *ROT_LIMITS[dxl_id]))
    status.setText(f"ID {dxl_id} stalled — goal pulled back to {pos}. "
                   f"Check CLOSE limit calibration for this joint.")
    print(f"ID {dxl_id} stall backoff at pos {pos}")


# ============================================================
# AUTO-RECOVERY FOR HW ERROR 32
# ============================================================
recovering = set()

def recover_motor(dxl_id):
    """Reboot a servo in hardware error, restore settings, torque back ON."""
    if dxl_id in recovering:
        return
    recovering.add(dxl_id)
    status.setText(f"ID {dxl_id} hardware error — auto rebooting...")
    QApplication.processEvents()

    packetHandler.reboot(portHandler, dxl_id)
    time.sleep(0.8)                       # wait for servo restart

    setup_motor(dxl_id)                   # RAM is lost after reboot!
    checked_write(packetHandler.write1ByteTxRx, dxl_id,
                  ADDR_TORQUE_ENABLE, 1, "torque on (recovery)")

    # Do NOT resend the old goal (that's what caused the error).
    # Hold at the current position instead.
    pos, res, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id,
                                              ADDR_PRESENT_POSITION)
    if res == COMM_SUCCESS:
        checked_write(packetHandler.write4ByteTxRx, dxl_id,
                      ADDR_GOAL_POSITION, pos, "hold position")
        # pull the slider to the actual position so the user doesn't
        # immediately resend the fatal goal
        if dxl_id in FLEX_IDS:
            s = flex_sliders[FLEX_IDS.index(dxl_id)]
            set_slider_silent(s, clamp(pos, *flex_range(dxl_id)))
        else:
            s = rot_sliders[ROT_IDS.index(dxl_id)]
            set_slider_silent(s, clamp(pos, *ROT_LIMITS[dxl_id]))

    status.setText(f"ID {dxl_id} recovered — goal held at safe position")
    recovering.discard(dxl_id)


def check_hw_errors():
    """Called periodically: auto-recover on error 32 / any HW error."""
    if not connected:
        return
    for dxl_id in IDS:
        err, res, _ = packetHandler.read1ByteTxRx(portHandler, dxl_id,
                                                  ADDR_HW_ERROR_STATUS)
        if res == COMM_SUCCESS and err != 0:
            print(f"ID {dxl_id} HW error: {err}")
            recover_motor(dxl_id)


# ============================================================
# MOVE
# ============================================================
def move_all():
    rot_goals  = [clamp(rot_sliders[i].value(), *ROT_LIMITS[ROT_IDS[i]])
                  for i in range(3)]
    flex_goals = [clamp(flex_sliders[i].value(), *flex_range(FLEX_IDS[i]))
                  for i in range(3)]

    if not connected:
        for i in range(3):
            actual[ROT_IDS[i]]  = rot_goals[i]
            actual[FLEX_IDS[i]] = flex_goals[i]
        refresh_feedback_sim()
        return

    # New goal: reset stall timers so backoff doesn't fire on a fresh move.
    for st in stall_state.values():
        st["since"] = None

    sync = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, 4)
    for i in range(3):
        sync.addParam(ROT_IDS[i],  pos_to_param(rot_goals[i]))
        sync.addParam(FLEX_IDS[i], pos_to_param(flex_goals[i]))
    if sync.txPacket() != COMM_SUCCESS:
        print("Sync write error")
    sync.clearParam()


def set_preset(rot_vals, flex_vals):
    for i, dxl_id in enumerate(ROT_IDS):
        set_slider_silent(rot_sliders[i],
                          clamp(rot_vals[dxl_id], *ROT_LIMITS[dxl_id]))
    for i, dxl_id in enumerate(FLEX_IDS):
        set_slider_silent(flex_sliders[i],
                          clamp(flex_vals[dxl_id], *flex_range(dxl_id)))
    move_all()


# ============================================================
# FEEDBACK
# ============================================================
actual = {i: 2048 for i in IDS}


def refresh_feedback_sim():
    for dxl_id in IDS:
        fb_labels[dxl_id].setText(f"ID {dxl_id}: {actual[dxl_id]}  |  --- mA")


def read_position():
    if not connected:
        return
    for dxl_id in IDS:
        pos, res, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id,
                                                  ADDR_PRESENT_POSITION)
        cur_raw, res2, _ = packetHandler.read2ByteTxRx(portHandler, dxl_id,
                                                       ADDR_PRESENT_CURRENT)
        if res == COMM_SUCCESS:
            actual[dxl_id] = pos
            cur = abs(signed16(cur_raw)) if res2 == COMM_SUCCESS else 0
            warn = "!" if cur >= CURRENT_WARN else ""
            fb_labels[dxl_id].setText(f"ID {dxl_id}: {pos}  |  {cur}u {warn}")
            fb_labels[dxl_id].setStyleSheet(
                "color:#f87171; font-weight:bold;" if cur >= CURRENT_WARN
                else "color:#e5e7eb;")
            if res2 == COMM_SUCCESS:
                check_stall(dxl_id, pos, cur)


tick_count = 0
def periodic():
    """Main timer: read positions every 200 ms, check HW errors every 1 s."""
    global tick_count
    tick_count += 1
    read_position()
    if tick_count % 5 == 0:
        check_hw_errors()


# ============================================================
# GUI
# ============================================================
app = QApplication(sys.argv)
app.setStyleSheet("""
QWidget { background:#0f1115; color:#e5e7eb; font-family:Arial; }
QLabel#title { font-size:20px; font-weight:bold; color:#facc15; }
QLabel#status { color:#93c5fd; font-weight:bold; }
QGroupBox { border:1px solid #2a2e37; border-radius:10px; margin-top:12px; padding-top:14px; }
QGroupBox::title { subcontrol-origin: margin; left:12px; color:#93c5fd; }
QSlider::groove:horizontal { height:8px; background:#2a2e37; border-radius:4px; }
QSlider::handle:horizontal { width:20px; margin:-7px 0; border-radius:10px; background:#facc15; }
QSlider::sub-page:horizontal { background:#38bdf8; border-radius:4px; }
QSlider#rot::handle:horizontal { background:#a78bfa; }
QSlider#rot::sub-page:horizontal { background:#7c3aed; }
QPushButton { background:#1f2430; border:1px solid #3a3f4a; border-radius:8px;
              padding:8px 10px; font-weight:bold; }
QPushButton:hover { background:#2a3040; }
QPushButton#on  { background:#14532d; border-color:#22c55e; }
QPushButton#off { background:#7f1d1d; border-color:#ef4444; }
QPushButton#move{ background:#1e3a8a; border-color:#3b82f6; font-size:15px; padding:12px; }
""")

window = QWidget()
window.setWindowTitle("Chromasense 3-Finger Gripper — Anti-Overload")
window.resize(760, 800)
root = QVBoxLayout(window)

title = QLabel("CHROMASENSE GRIPPER — MODE 5 CURRENT-LIMITED")
title.setObjectName("title")
title.setAlignment(Qt.AlignCenter)
root.addWidget(title)

status = QLabel("Ready. Torque ON will set mode 5 + goal current automatically.")
status.setObjectName("status")
status.setAlignment(Qt.AlignCenter)
root.addWidget(status)

# ---- torque ----
trow = QHBoxLayout()
btn_on = QPushButton("TORQUE ON")
btn_on.setObjectName("on")
btn_off = QPushButton("TORQUE OFF")
btn_off.setObjectName("off")
btn_on.clicked.connect(lambda: torque(1))
btn_off.clicked.connect(lambda: torque(0))
trow.addWidget(btn_on)
trow.addWidget(btn_off)
root.addLayout(trow)

# ---- presets ----
preset_box = QGroupBox("Grip Configurations")
pg = QGridLayout(preset_box)
for idx, (text, rv, fv) in enumerate(PRESETS):
    b = QPushButton(text)
    b.clicked.connect(lambda _, r=rv, f=fv: set_preset(r, f))
    pg.addWidget(b, idx // 2, idx % 2)
root.addWidget(preset_box)

# ---- manual control (PER-JOINT limits) ----
finger_names  = ["Finger A", "Finger B", "Finger C"]
finger_colors = ["#facc15", "#38bdf8", "#f87171"]
flex_sliders, rot_sliders = [], []

for i in range(3):
    rid, fid = ROT_IDS[i], FLEX_IDS[i]
    fmin, fmax = flex_range(fid)
    rmin, rmax = ROT_LIMITS[rid]

    box = QGroupBox(f"{finger_names[i]} — rot ID {rid} ({rmin}-{rmax}) | "
                    f"flex ID {fid} ({fmin}-{fmax})")
    lay = QGridLayout(box)

    lay.addWidget(QLabel("Base"), 0, 0)
    rs = QSlider(Qt.Horizontal)
    rs.setObjectName("rot")
    rs.setRange(rmin, rmax)
    rs.setValue(clamp(ROT_HOME[rid], rmin, rmax))
    rl = QLabel(str(rs.value()))
    rl.setStyleSheet("color:#a78bfa; font-weight:bold; min-width:52px;")
    rs.valueChanged.connect(lambda v, l=rl: l.setText(str(v)))
    rs.sliderReleased.connect(move_all)
    lay.addWidget(rs, 0, 1)
    lay.addWidget(rl, 0, 2)

    lay.addWidget(QLabel("Flex"), 1, 0)
    fs = QSlider(Qt.Horizontal)
    fs.setRange(fmin, fmax)
    fs.setValue(clamp(FLEX_LIMITS[fid][0], fmin, fmax))   # start at OPEN
    fl = QLabel(str(fs.value()))
    fl.setStyleSheet(f"color:{finger_colors[i]}; font-weight:bold; min-width:52px;")
    fs.valueChanged.connect(lambda v, l=fl: l.setText(str(v)))
    fs.sliderReleased.connect(move_all)
    lay.addWidget(fs, 1, 1)
    lay.addWidget(fl, 1, 2)

    root.addWidget(box)
    rot_sliders.append(rs)
    flex_sliders.append(fs)

hint = QLabel("Mode 5: the servo pushes with at most Goal Current — even when "
              "blocked it will not overload. Red label = high current "
              "(holding a load). Stall backoff pulls the goal back after "
              f"{STALL_TIME:.0f}s of blocked pushing.")
hint.setStyleSheet("color:#6b7280;")
hint.setWordWrap(True)
hint.setAlignment(Qt.AlignCenter)
root.addWidget(hint)

# ---- move ----
btn_move = QPushButton("MOVE ALL")
btn_move.setObjectName("move")
btn_move.clicked.connect(move_all)
root.addWidget(btn_move)

# ---- feedback ----
fb_box = QGroupBox("Encoder Feedback  (position | current)")
fb_lay = QGridLayout(fb_box)
fb_labels = {}
for i, dxl_id in enumerate(IDS):
    l = QLabel(f"ID {dxl_id}: ---")
    fb_labels[dxl_id] = l
    fb_lay.addWidget(l, i // 3, i % 3)
root.addWidget(fb_box)

timer = QTimer()
timer.timeout.connect(periodic)
timer.start(200)

window.show()
sys.exit(app.exec_())
