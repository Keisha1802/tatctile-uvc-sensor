# Gripper finger servos (Dynamixel via U2D2)

Control of the 3-finger gripper's Dynamixel servos (XC430 / XC330 / XM430, Protocol 2.0, 57600 baud) through a ROBOTIS U2D2.

| IDs | Joint |
|---|---|
| 1, 2, 3 | rotation (base of each finger) |
| 4, 5, 6 | flexion (finger open / close) |

## Main scripts

- `chromasense_gui.py`: main gripper GUI (PyQt5). It sets current-based position mode (mode 5) with a current cap, stall protection, per-joint open / close limits (`FLEX_LIMITS`, `ROT_LIMITS`) and presets (home, 3-finger parallel, 2-finger parallel, push).
- `hold_1700.py`: mounting helper. Resets the homing offset of ID 4-6 and holds them at 1700 (open) while a finger is fitted on its horn.
- `calibrate1700.py`: sets the homing offset (EEPROM) so the open finger reads 1700, without taking the hand apart.
- `finger_gui.py`, `gui_6dof_dynamixel.py`, `gui_xc330.py`, `control_xc330.py`: simple slider / test controllers.
- `scan_dxl.py`, `read_all_position.py`, `change_id.py`, `change_id_xc430.py`, `test.py`: quick tools.
- `u2d2_interface.py`, `fake_u2d2_interface.py`, `base_interface.py`, `helpers/`, `examples/`: the U2D2 interface library (MIT, see `LICENSE` and `README.md`).

## Setup

```bash
cd dynamixel_u2d2
python3 -m venv dxl-env && . dxl-env/bin/activate      # (the venv is not in git)
pip install -r requirements.txt PyQt5
python3 helpers/u2d2_port_timer.py --latency-timer 2  # U2D2 USB latency
python3 scan_dxl.py                                    # find the servos
python3 chromasense_gui.py
```

Only one program can use the U2D2 port at a time. Check `DEVICENAME` (`/dev/ttyUSB0` or `/dev/ttyUSB1`) in each script.
