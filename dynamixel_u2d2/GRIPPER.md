# Gripper finger servos (Dynamixel via U2D2)

Control of the 3-finger gripper's Dynamixel servos (XC430 / XC330 / XM430, Protocol 2.0, 57600 baud) through a ROBOTIS U2D2.

| IDs | Joint |
|---|---|
| 1, 2, 3 | rotation (base of each finger) |
| 4, 5, 6 | flexion (finger open / close) |

## Reset the servo IDs (setup_ids.py)

```bash
./run_gripper.sh setup_ids.py check   # scan every baudrate, find duplicate / missing / swapped IDs (writes nothing)
./run_gripper.sh setup_ids.py         # wizard: plug ONE servo at a time, choose its joint -> ID + 57600 written
./run_gripper.sh setup_ids.py one 1   # the single servo on the bus -> ID 1
```
Rotation servos (XC430) become ID 1-3 and flexion servos (XC330) become ID 4-6 (suggested from the model). Torque is switched off before the EEPROM write, and the result is verified.

## Recalibrate the encoders (calibrate_gui.py, or calibrate_encoders.py in the terminal)

`./run_gripper.sh calibrate_gui.py` is the easiest way. It shows live positions and walks through four buttons: 1 TORQUE OFF, 2 SET HOME / OPEN (or `set` per servo), 3 RECORD CLOSE / STOP RECORDING, 4 SAVE. Flex limits can also be edited in the gripper GUI (row "Limits": type them or "open = here" / "close = here", then SAVE FLEX LIMITS).

```bash
./run_gripper.sh calibrate_encoders.py          # torque off -> rotation straight + fingers open -> ENTER
                                                #            -> close each finger fully by hand -> ENTER
./run_gripper.sh calibrate_encoders.py --read   # positions + homing offsets only
./run_gripper.sh calibrate_encoders.py --only 4,5
```
calibrate_gui.py uses a SOFTWARE zero: it sets every homing offset back to 0 and records the encoder reading at rotation straight / finger open as each joint's reference. It writes nothing to EEPROM, because on 25 Nov the servos accepted new offsets but their positions did not change. chromasense_gui.py shifts its limits and presets to these readings. The terminal tool calibrate_encoders.py still writes homing offsets (rotation straight = 2048, flexion open = 1700). The fully closed position is measured per finger and stored 40 ticks short of the hard stop. The result goes to `calib.json`, which `chromasense_gui.py` reads at start (FLEX_LIMITS, ROT_HOME). The tool warns when a finger's range crosses the encoder's 0/4095 point; in that case remount the horn with `hold_1700.py`.

## Main scripts

- `chromasense_gui.py`: main gripper GUI (PyQt5). It sets current-based position mode (mode 5) with a current cap, stall protection, per-joint open / close limits (`FLEX_LIMITS`, `ROT_LIMITS`) and presets (home, 3-finger parallel, 2-finger parallel, push).
- `hold_1700.py`: mounting helper. Resets the homing offset of ID 4-6 and holds them at 1700 (open) while a finger is fitted on its horn.
- `calibrate1700.py`: sets the homing offset (EEPROM) so the open finger reads 1700, without taking the hand apart.
- `finger_gui.py`, `gui_6dof_dynamixel.py`, `gui_xc330.py`, `control_xc330.py`: simple slider / test controllers.
- `scan_dxl.py`, `read_all_position.py`, `change_id.py`, `change_id_xc430.py`, `test.py`: quick tools.
- `u2d2_interface.py`, `fake_u2d2_interface.py`, `base_interface.py`, `helpers/`, `examples/`: the U2D2 interface library (MIT, see `LICENSE` and `README.md`).

## Setup (Linux: Ubuntu / Debian, x86_64 PC or aarch64 Jetson / Raspberry Pi)

```bash
cd dynamixel_u2d2
bash install_linux.sh          # venv dxl-env + dynamixel-sdk + PyQt5 + U2D2 udev rule + dialout group
                               # (log out and in once after the first install)
./run_gripper.sh               # main GUI (chromasense_gui.py)
./run_gripper.sh scan_dxl.py   # any other script, with the venv
DXL_PORT=/dev/ttyUSB1 ./run_gripper.sh     # another port
```

- **`/dev/u2d2`**: the udev rule gives the U2D2 (FTDI 0403:6014) a fixed name and a 1 ms USB latency timer (the default 16 ms slows every read). Every script uses `DXL_PORT`, else `/dev/u2d2`, else its old `/dev/ttyUSB0` / `/dev/ttyUSB1`.
- **No permission on the port**: log out and in after the install (dialout group).
- **`/dev/ttyUSB*` disappears right after plugging in**: `sudo apt remove brltty`.
- **Without sudo**: `bash install_linux.sh --no-udev`, then pass the port with `DXL_PORT=`.
- **No U2D2 plugged in**: `chromasense_gui.py` starts in simulation mode.

Only one program can use the U2D2 port at a time.
