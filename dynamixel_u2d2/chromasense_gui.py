from dynamixel_sdk import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, Qt
import sys
import time

# ============================================================
# DYNAMIXEL CONFIG
# ============================================================
import os
DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
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
OP_MODE_EXT_POSITION      = 4     # extended (multi-turn) position: XC430 has no current sensor
ADDR_PWM_LIMIT            = 36    # EEPROM, 2 bytes, 0..885
ADDR_GOAL_PWM             = 100   # 2 bytes: force cap in position modes 3 / 4
GOAL_PWM_ROT              = 600   # ~68 % of full PWM for the rotation servos (XC430)
LOAD_STALL_PWM            = 780   # present LOAD units (0.1 %) used as 'goal current' for the stall check
MOTOR_MODE = {}                   # id -> "current" (XC330, mode 5) / "pwm" (XC430, mode 4)

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

# calibrate_encoders.py writes the measured open / close per finger here (overrides the values above)
try:
    import json as _json
    _cal = _json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")))
    for _k, _v in (_cal.get("flex_limits") or {}).items():
        if int(_k) in FLEX_LIMITS:
            FLEX_LIMITS[int(_k)] = (int(_v[0]), int(_v[1]))
    for _k, _v in (_cal.get("rot_home") or {}).items():
        if int(_k) in ROT_HOME:
            ROT_HOME[int(_k)] = int(_v)
    print(f"calib.json ({_cal.get('date', '')}): FLEX_LIMITS {FLEX_LIMITS}")
except (OSError, ValueError, KeyError, TypeError, IndexError):
    pass

# ============================================================
# CURRENT PROTECTION (THE FIX FOR HW ERROR 32)
# ============================================================
# Goal Current = max current the servo may use in mode 5.
# XM430-W350: 1 unit = 2.69 mA. 200 units ~= 540 mA (medium grip).
# NOTE: Goal Current is REJECTED by the servo if it exceeds Current Limit
# (addr 38, default 1193 on XM430-W350). The old code wrote 2000, the servo
# silently refused it, and the current cap you thought you had never applied.
GOAL_CURRENT_FLEX = 350     # finger grip strength (raise if too weak; GUI slider "Grip current")
                            # 200 stopped the fingers short of the calibrated CLOSE (25 Nov)
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

# calib.json (calibrate_gui.py, software calibration): the rotation values above are written for a straight
# finger at 2048 -> shift the limits and the presets to the measured straight position of each servo
_ROT_SHIFT = {i: ROT_HOME[i] - 2048 for i in ROT_HOME}
if any(_ROT_SHIFT.values()):
    ROT_LIMITS = {i: (ROT_LIMITS[i][0] + _ROT_SHIFT[i], ROT_LIMITS[i][1] + _ROT_SHIFT[i]) for i in ROT_LIMITS}
    PRESETS = [(n, {i: v + _ROT_SHIFT[i] for i, v in rv.items()}, fv) for n, rv, fv in PRESETS]
    print(f"rotation straight = {ROT_HOME} -> ROT_LIMITS {ROT_LIMITS}")

# ============================================================
# CONNECT
# ============================================================
portHandler = PortHandler(DEVICENAME)

# A serial glitch (U2D2 USB hiccup, ModemManager probing the port) raised inside the SDK in the middle of a
# packet and left its 'port in use' flag set -> every later call failed with "Port is in use!" (25 Nov).
# Now a glitch is just a timed-out packet, and the flag is always released.
PORT_ERR = {"n": 0, "t": 0.0}
_orig_read, _orig_write = portHandler.readPort, portHandler.writePort


def _safe_read(length):
    try:
        return _orig_read(length)
    except Exception as e:
        PORT_ERR["n"] += 1
        if time.time() - PORT_ERR["t"] > 5:
            PORT_ERR["t"] = time.time()
            print(f"!! serial glitch on {DEVICENAME}: {e} (another program on the port? ModemManager?)")
        return b""


def _safe_write(packet):
    try:
        return _orig_write(packet)
    except Exception as e:
        PORT_ERR["n"] += 1
        print(f"!! serial write failed on {DEVICENAME}: {e}")
        return 0


portHandler.readPort, portHandler.writePort = _safe_read, _safe_write
packetHandler = PacketHandler(PROTOCOL_VERSION)

try:
    connected = portHandler.openPort() and portHandler.setBaudRate(BAUDRATE)
except Exception as e:                       # no U2D2 plugged in: pyserial raises instead of returning False
    print(f"!! {DEVICENAME}: {e}")
    connected = False
print("Dynamixel connected" if connected
      else "!! Port failed, GUI running in simulation mode")

# ENCODER WRAP: a joint whose calibrated zero sits near the encoder's 0/4095 point (e.g. rotation straight
# at 33) reads ~4000 after a power cycle when it rests just below zero. In the multi-turn mode the GUI uses
# (mode 5) the servo keeps counting from where it IS, so the limits / presets of such a joint are moved by
# whole turns (4096) to the frame the servo reports now -> no 350 deg spin back to "33".
def _unwrap_frames():
    global ROT_LIMITS, PRESETS
    for i in list(ROT_HOME) + list(FLEX_LIMITS):
        pos, res, _ = packetHandler.read4ByteTxRx(portHandler, i, ADDR_PRESENT_POSITION)
        if res != COMM_SUCCESS:
            continue
        pos = pos - 4294967296 if pos > 2147483647 else pos
        ref = ROT_HOME[i] if i in ROT_HOME else FLEX_LIMITS[i][0]
        k = round((pos - ref) / 4096.0)
        if k == 0:
            continue
        d = 4096 * k
        if i in ROT_HOME:
            ROT_HOME[i] += d
            ROT_LIMITS = {**ROT_LIMITS, i: (ROT_LIMITS[i][0] + d, ROT_LIMITS[i][1] + d)}
            PRESETS = [(n, {**rv, i: rv[i] + d} if i in rv else rv, fv) for n, rv, fv in PRESETS]
        else:
            FLEX_LIMITS[i] = (FLEX_LIMITS[i][0] + d, FLEX_LIMITS[i][1] + d)
            PRESETS = [(n, rv, {**fv, i: fv[i] + d} if i in fv else fv) for n, rv, fv in PRESETS]
        print(f"ID {i}: encoder reads {pos}, calibrated zero {ref} -> frame moved {k:+d} turn(s)")


if connected:
    _unwrap_frames()


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
    ok = checked_write(packetHandler.write2ByteTxRx, dxl_id,
                       ADDR_GOAL_CURRENT, gc, "goal current")
    MOTOR_MODE[dxl_id] = "current"
    if not ok:
        # XC430 (the rotation servos) has no current sensor: no current-based position mode. Fall back to
        # EXTENDED POSITION (mode 4, multi-turn like mode 5, so the encoder-wrap handling still holds) with
        # the force capped by Goal PWM instead of Goal Current.
        checked_write(packetHandler.write1ByteTxRx, dxl_id, ADDR_TORQUE_ENABLE, 0, "torque off")
        checked_write(packetHandler.write1ByteTxRx, dxl_id,
                      ADDR_OPERATING_MODE, OP_MODE_EXT_POSITION, "op mode 4")
        plim, res, _ = packetHandler.read2ByteTxRx(portHandler, dxl_id, ADDR_PWM_LIMIT)
        pwm = min(GOAL_PWM_ROT, plim) if res == COMM_SUCCESS and plim > 0 else GOAL_PWM_ROT
        if checked_write(packetHandler.write2ByteTxRx, dxl_id, ADDR_GOAL_PWM, pwm, "goal PWM"):
            MOTOR_MODE[dxl_id] = "pwm"
            print(f"[ID {dxl_id}] no current control on this model -> extended position, force cap PWM {pwm}/885")

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
    if MOTOR_MODE.get(dxl_id) == "pwm":        # XC430: addr 126 is Present LOAD (0.1 %), not current
        return LOAD_STALL_PWM
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
            cw = 0.9 * goal_current_of(dxl_id)        # red when it pushes near its own cap
            warn = "!" if cur >= cw else ""
            fb_labels[dxl_id].setText(f"ID {dxl_id}: {pos}  |  {cur}u {warn}")
            fb_labels[dxl_id].setStyleSheet(
                "color:#f87171; font-weight:bold;" if cur >= cw
                else "color:#e5e7eb;")
            if res2 == COMM_SUCCESS:
                check_stall(dxl_id, pos, cur)


tick_count = 0
def periodic():
    """Main timer: read positions every 200 ms, check HW errors every 1 s."""
    global tick_count
    tick_count += 1
    try:
        read_position()
        if tick_count % 5 == 0:
            check_hw_errors()
    except Exception as e:
        print(f"!! read failed: {e}")
    finally:
        portHandler.is_using = False            # never leave the SDK's port flag stuck


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
flex_ui = {}


def apply_flex_limits(fid, o, c):
    """new OPEN / CLOSE of one finger: slider range, buttons, presets, % - live (SAVE LIMITS = keep)"""
    global PRESETS
    o, c = int(o), int(c)
    if abs(c - o) < 50:
        status.setText(f"ID {fid}: open and close too close together ({o} / {c})")
        return
    oo, oc = FLEX_LIMITS[fid]
    FLEX_LIMITS[fid] = (o, c)
    PRESETS = [(n, rv, {**fv, fid: (o if fv.get(fid) == oo else c if fv.get(fid) == oc else fv.get(fid))}
                if fid in fv else fv) for n, rv, fv in PRESETS]
    U = flex_ui[fid]
    lo, hi = flex_range(fid)
    s_ = U["slider"]
    v = clamp(s_.value(), lo, hi)
    s_.blockSignals(True)
    s_.setRange(lo, hi)
    s_.setValue(v)
    s_.blockSignals(False)
    U["label"].setText(f"{v}  {U['pct'](v):.0f}%")
    U["go"][0].setText(f"OPEN {o}")
    U["go"][1].setText(f"CLOSE {c}")
    nm, rid, rmin, rmax = U["title"]
    U["box"].setTitle(f"{nm} — rot ID {rid} ({rmin}-{rmax}) | flex ID {fid} ({lo}-{hi})")
    status.setText(f"ID {fid}: limits open {o}  close {c}  ({abs(c - o) * 360 / 4096:.0f}°) - click SAVE FLEX LIMITS")


def save_flex_limits():
    import json
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib.json")
    try:
        d = json.load(open(path))
    except (OSError, ValueError):
        d = {}
    d.setdefault("flex_limits", {}).update({str(k): [int(v[0]), int(v[1])] for k, v in FLEX_LIMITS.items()})
    d["date"] = time.strftime("%Y-%m-%d %H:%M")
    json.dump(d, open(path, "w"), indent=1)
    status.setText("flex limits saved in calib.json: " + ", ".join(
        f"ID {k} {v[0]}-{v[1]}" for k, v in sorted(FLEX_LIMITS.items())))

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
    def _pct(v, f=fid):                        # 0 % = calibrated OPEN, 100 % = calibrated CLOSE
        o, c = FLEX_LIMITS[f]
        return 100.0 * (v - o) / (c - o) if c != o else 0.0
    fl = QLabel(f"{fs.value()}  {_pct(fs.value()):.0f}%")
    fl.setStyleSheet(f"color:{finger_colors[i]}; font-weight:bold; min-width:92px;")
    fs.valueChanged.connect(lambda v, l=fl, p=_pct: l.setText(f"{v}  {p(v):.0f}%"))
    fs.sliderReleased.connect(move_all)
    lay.addWidget(fs, 1, 1)
    lay.addWidget(fl, 1, 2)
    # one click to the calibrated OPEN / CLOSE of this finger (calib.json)
    gobtn = []
    for col, (txt, k) in enumerate((("OPEN", 0), ("CLOSE", 1)), 3):
        b = QPushButton(f"{txt} {FLEX_LIMITS[fid][k]}")
        b.clicked.connect(lambda _, s_=fs, f=fid, k=k: (set_slider_silent(s_, FLEX_LIMITS[f][k]), move_all()))
        lay.addWidget(b, 1, col)
        gobtn.append(b)

    # ---- FLEX LIMITS of this finger: type them, or take the present position ----
    lay.addWidget(QLabel("Limits"), 2, 0)
    lim_row = QHBoxLayout()
    sp_o, sp_c = QSpinBox(), QSpinBox()
    for sp_, k in ((sp_o, 0), (sp_c, 1)):
        sp_.setRange(-20000, 20000)
        sp_.setValue(FLEX_LIMITS[fid][k])
        sp_.setKeyboardTracking(False)               # apply on Enter / arrows, not on every digit
    b_o = QPushButton("open = here")
    b_c = QPushButton("close = here")
    b_o.setToolTip("this finger's present encoder position becomes its OPEN limit")
    b_c.setToolTip("this finger's present encoder position becomes its CLOSE limit")
    for w_ in (QLabel("open"), sp_o, QLabel("close"), sp_c, b_o, b_c):
        lim_row.addWidget(w_)
    lay.addLayout(lim_row, 2, 1, 1, 4)
    flex_ui[fid] = dict(slider=fs, label=fl, pct=_pct, go=gobtn, open=sp_o, close=sp_c, box=box,
                        title=(finger_names[i], rid, rmin, rmax))
    sp_o.valueChanged.connect(lambda v, f=fid: apply_flex_limits(f, v, FLEX_LIMITS[f][1]))
    sp_c.valueChanged.connect(lambda v, f=fid: apply_flex_limits(f, FLEX_LIMITS[f][0], v))
    b_o.clicked.connect(lambda _, f=fid: flex_ui[f]["open"].setValue(actual.get(f, FLEX_LIMITS[f][0])))
    b_c.clicked.connect(lambda _, f=fid: flex_ui[f]["close"].setValue(actual.get(f, FLEX_LIMITS[f][1])))

    root.addWidget(box)
    rot_sliders.append(rs)
    flex_sliders.append(fs)

b_save_lim = QPushButton("SAVE FLEX LIMITS (calib.json)")
b_save_lim.clicked.connect(save_flex_limits)
root.addWidget(b_save_lim)

hint = QLabel("Mode 5: the servo pushes with at most Goal Current — even when "
              "blocked it will not overload. Red label = high current "
              "(holding a load). Stall backoff pulls the goal back after "
              f"{STALL_TIME:.0f}s of blocked pushing.")
hint.setStyleSheet("color:#6b7280;")
hint.setWordWrap(True)
hint.setAlignment(Qt.AlignCenter)
root.addWidget(hint)

# ---- grip current (force cap of the flexion servos, mode 5) - live, no restart ----
def set_flex_current(v):
    global GOAL_CURRENT_FLEX
    GOAL_CURRENT_FLEX = int(v)
    gc_label.setText(f"{v}")
    if not connected:
        return
    for fid in FLEX_IDS:
        lim, res, _ = packetHandler.read2ByteTxRx(portHandler, fid, ADDR_CURRENT_LIMIT)
        val = min(int(v), lim) if res == COMM_SUCCESS and lim > 0 else int(v)
        checked_write(packetHandler.write2ByteTxRx, fid, ADDR_GOAL_CURRENT, val, "goal current")

gc_row = QHBoxLayout()
gc_row.addWidget(QLabel("Grip current (flex ID 4-6)"))
gc_slider = QSlider(Qt.Horizontal)
_gc_max = 1000
if connected:
    _l, _r, _ = packetHandler.read2ByteTxRx(portHandler, FLEX_IDS[0], ADDR_CURRENT_LIMIT)
    if _r == COMM_SUCCESS and _l > 0:
        _gc_max = int(_l)
gc_slider.setRange(50, _gc_max)
gc_slider.setValue(min(GOAL_CURRENT_FLEX, _gc_max))
gc_label = QLabel(str(gc_slider.value()))
gc_label.setStyleSheet("font-weight:bold; min-width:52px;")
gc_slider.valueChanged.connect(lambda v: gc_label.setText(str(v)))
gc_slider.sliderReleased.connect(lambda: set_flex_current(gc_slider.value()))
gc_row.addWidget(gc_slider)
gc_row.addWidget(gc_label)
gc_row.addWidget(QLabel(f"(max {_gc_max} = Current Limit)"))
root.addLayout(gc_row)

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
