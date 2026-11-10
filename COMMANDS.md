# Command reference

Every command, option and key of the tactile UVC sensor in one place.
Run everything from the repository folder (`cd ~/tatctile-uvc-sensor`).
`./run.sh` picks the right Python (the `.venv` on Ubuntu/Jetson, system Python on
Raspberry Pi OS) and starts `host/run_sensor_fast.py` unless `--script=` says otherwise.
Quit the programs with **q** (not Ctrl+C).

---

## 1. Setup

### Pi Zero W (each camera module)
```bash
sudo bash zero/setup_uvc_zero.sh        # build + install the UVC gadget service, then: sudo reboot
sudo bash zero/tune_zero.sh             # optional: steady 15 fps (CPU fixed at 1 GHz, no overclock,
                                        # uvc-gadget priority, Bluetooth/timers off), then reboot
sudo bash zero/tune_zero.sh --status    # show what is applied
sudo bash zero/tune_zero.sh --undo      # revert, then reboot
```
The Zero's hostname is the camera name (`camera1`, `camera2`). Plug the Zero's **USB** (data) port into the host.

### Host
```bash
bash install/setup_raspi.sh                 # Raspberry Pi 5, Raspberry Pi OS
bash install/setup_raspi.sh --usb-power     # + 1.6 A on the USB ports (Zeros powered by the Pi 5)
bash install/setup_ubuntu.sh                # Ubuntu 22.04/24.04 (PC, Pi 5) and Jetson (JetPack 6):
                                            # .venv + OpenCV contrib + video group + udev rule;
                                            # on a Jetson also CuPy for 'gpu'
bash install/setup_ubuntu.sh --ros          # + ROS 2 (Humble on 22.04, Jazzy on 24.04)
sudo bash install/jetson_max.sh            # Jetson: MAXN + jetson_clocks + USB autosuspend off,
                                            # kept after reboot (--status, --undo)
```
Log out and in once (video group). Update later with `git pull`.

### Checks
```bash
ls /dev/v4l/by-id/ | grep Tactile                     # both cameras present?
./run.sh --script=preview_uvc.py                      # live image + flow check (f, r, s, q)
./run.sh --script=test_fps.py camera1                 # fps per resolution
v4l2-ctl -d /dev/v4l/by-id/usb-camera1_Tactile_UVC_Cam_e19b63bd-video-index0 \
         --stream-mmap --stream-count=300             # raw delivery rate of one camera
./run.sh --script=bench_pipeline.py                   # processing speed of this machine
./run.sh --script=bench_pipeline.py --camera=camera1 --variants   # DIS settings vs speed
```

---

## 2. Run

### One camera
```bash
./run.sh usb                                 # first Tactile camera found
./run.sh usb --usb=camera1                   # pick a Zero by hostname
./run.sh usb --usb=/dev/video8               # or by device
./run.sh usb --usb=camera1 clean prof gpu    # neutral settings + timing + GPU depth (Jetson)
```

### Two cameras (one window, one process per camera)
```bash
./run.sh dual --usb=camera1 --usb2=camera2
./run.sh dual --usb=camera1 --usb2=camera2 clean prof gpu --depth-hz=15    # Jetson
./run.sh dual --usb=camera1 --usb2=camera2 nogui prof --seconds=30         # measure only
```

### Three cameras (one window, one process per camera)
```bash
./run.sh triple --usb=camera1 --usb2=camera2 --usb3=camera3
./run.sh triple                              # the first three Tactile cameras found
./run.sh triple gpu --depth-hz=15 prof       # laptop/Jetson with an NVIDIA GPU
```
All dual options and keys work the same (`r`, `c`, `w`, … act on all cameras).
Plug each Zero into its own USB port (not one hub): three 15 fps YUYV streams need
~220 Mbit/s of the 480 Mbit/s a USB 2.0 hub shares.

### With ROS 2
```bash
source /opt/ros/jazzy/setup.bash            # or humble
./run.sh dual --usb=camera1 --usb2=camera2 ros --gel-mm=20
python3 host/tactile_listener.py camera1 camera2      # example consumer (gripper logic)
```

### Automatic grasp (sensor closes the loop, no ROS)
```bash
# once: teach the two poses (servo_gui.py to move there, then q; servos keep holding)
./run.sh --script=servo_grasp.py teach open
./run.sh --script=servo_grasp.py teach closed          # fully closed, empty hand -> host/servo_grasp.json
# mechanical max per finger (measured with servo_gui): never commanded past it (-1 deg), caps the advance
./run.sh --script=servo_grasp.py limit closed -94.5 -78.5 -87.7
./run.sh --script=servo_grasp.py limit open -2.9 1.7 0.6
# ONE terminal: grasp starts the sensors itself (sensor options after a lone --), q stops both
./run.sh --script=servo_grasp.py --sensor=triple --depth-ref=0.2 -- --usb=camera1 --usb2=camera2 --usb3=camera3
# or two terminals -- terminal 1: sensor with UDP output
./run.sh usb --udp                                     # or: ./run.sh dual ... --udp
./run.sh triple --usb=camera1 --usb2=camera2 --usb3=camera3 --udp   # three sensors: contact = any, depth = deepest
# terminal 2: g = close until touch, then hold depth;  o = open;  h = hold;  q = quit
./run.sh --script=servo_grasp.py --depth-ref=0.3 --speed=8
./run.sh --script=servo_grasp.py --map=1:camera2,2:camera1,3:camera3   # each finger (ID) follows ITS sensor
./run.sh --script=servo_grasp.py --together           # all fingers as one (any contact)
# keys: g grasp, a active (fingers stay, a pushed finger yields/follows), o open, h hold, t tare, q quit
# g/a tare the sensors first (press with nothing touching); contact = depth >= 0.10 mm AND area >= 2 % for 3 frames
# after contact each finger runs a PID on ITS median-filtered depth (--kp=12 --ki=2 --kd=0.3 --vmax=3 --deadband=0.03),
# at most --excursion=6 deg from where it touched; may close up to --overclose=15 deg past the taught closed pose
# default = coordinated grip (PID on the MEAN depth of the touching fingers + zero-sum balance --kbal=6); --independent = one PID per finger
# --slip-boost = raise the reference on a slip lasting --slip-hold=0.6 s (off by default)
# frame drops (shared USB hub): a touching finger keeps going on depth predicted from its own motion (k mm/deg learned online),
#   stops only after --stale=1.0 s without frames; --no-predict = off
# CENTRE PID (adaptive): >= 2 fingers touching -> zero-sum PID on their angles moves the object to the middle until
#   (gain adapts x0.5..x3 to how far the object really moves; note [CENTRE] adaptive gain)
#   ID1-3 have nearly the same degree (--centre-kp=0.4 --centre-ki=0.08 --centre-kd=0.05 --centre-dead=1; --no-centre = off)
# BASE PID: the base pushes the object to the optimal position (contact centred, shear 0, depths even)
#   with a PID on the tactile correction: --base-kp=1.5 --base-ki=0.5 --base-kd=0.15 (status line: BASE track |e|)
# ADVANCE: a finger at its closed pose whose sensor feels nothing moves on forward (<= --advance=10 deg, 3 deg/s, only while load < 0.5 Nm)
#   paused at --advance-load, resumes --advance-hyst=0.12 Nm lower; meanwhile the TOUCHING fingers YIELD
#   (light depth --yield-depth=0.6 x ref, open above --yield-load=0.35 Nm, make room for a paused finger) -> --no-yield = old rigid hold
# SAFETY (servo load, no sensor needed): finger > --grip-load=0.8 Nm backs off; any > --load-stop=1.2 Nm for 0.15 s -> open 5 deg + stop;
#   never past the taught closed pose (--overclose=0); target never > 3 deg ahead of the finger
# BASE (ID4-6, --no-base = off): after the grip settles each base joint is probed +-1.5 deg to learn how it shifts depth
#   between the fingers, then the base evens the fingers out (<= 1 deg/s, <= --base-max=6 deg, base load > 0.6 Nm -> back)
#   + optical flow: the base also drives each finger's gel shear to zero (--base-shear-w=0.01 mm/px, --no-base-shear = off)
#   BASE TACTILE SERVO (default): after a +-1.5 deg probe, the base moves continuously and LINEARLY with the tactile error:
#   contact centre -> gel middle, shear -> 0, equal depth, missing fingers depth -> ref (J refined online). --base-sweep = old search
#   speed: all base joints probed together (4 positions, --probe-wait=0.45 s), --base-speed=2.5 deg/s, --base-gain=1.5
# ML dataset: --object=sponge, keys 1 success / 0 fail / 2 slip label the episode (= last g/a)
#   host/grasp_logs/grasp_<time>.csv (~40 Hz rows: servos, PID, full tactile feature vector per finger;
#   f<ID>_new_frame=1 marks rows with a new camera frame) + host/grasp_logs/episodes.csv (one row per grasp)
# terminal: status line 5x/s (--log-hz=) + every phase/mode change;  CSV of every loop pass -> host/grasp_logs/ (--no-log = off)
./run.sh --script=servo_grasp.py --no-tactile --contact-load=0.6   # without sensor: contact = servo load
```

### Other tools
```bash
./run.sh --script=servo_fw.py                   # lab ESP32 sketch (460800, 6 servos, CSV): which IDs
                                                # answer + angle/load; sends nothing
./run.sh --script=servo_fw.py watch             # move one joint by hand -> '<<' marks its ID
./run.sh --script=servo_fw.py wiggle 3          # ID 3 +-5 deg and back, the other five hold
./run.sh --script=servo_fw.py zero              # current pose = new zero: prints zero_offset[] for the sketch
./run.sh --script=servo_gui.py                  # slow manual control: slider per servo, SPACE send on/off,
                                                # h hold, 0 zero pose, r reset ESP32; stops on ERR / load / lag
./run.sh --script=servo_scan.py                 # bus servos (Waveshare Servo Driver ESP32, ST/SC):
                                                # scan IDs; first press 'Start Serial Forwarding'
                                                # at 192.168.4.1 (Wi-Fi ESP32_DEV / 12345678)
./run.sh --script=servo_scan.py watch           # live position/load per ID: move one finger by hand
./run.sh --script=servo_scan.py wiggle 3        # servo ID 3 moves +-60 steps and back ('all' = each)
./run.sh --script=run_sensor_calib.py usb             # calibration/patent version of the single program
./run.sh --script=record_dual.py ...                  # raw recording of two cameras, no processing
python3 host/mts_calib.py host/calib/logs/depthlog_<camera>_*.csv <mts files> \
        --out=host/calib/depth_calib_<camera>.json    # depth -> mm from an MTS run (--range=0.4,1.2)
python3 host/make_synth_calib.py ...                  # SYNTHETIC calibration, testing only
./run.sh --script=bench_flow.py <burst folder>        # compare flow methods on recorded frames
```

---

## 3. Options

Words without dashes (`clean`, `gpu`, …) and `--name=value` options can be combined in any order.

### Both programs (single and dual)
| Option | Meaning |
|---|---|
| `clean` | neutral settings for this run (deadzone 0, EMA 0, tavg 1, rebaseline off); the saved `sensor_config.json` is not used and not changed. Press `c` to save, start without `clean` to load it |
| `gpu` | depth on the NVIDIA GPU (CuPy, Jetson); checked against the CPU at start (`[GPU] … matches CPU within …`), otherwise falls back to the CPU |
| `prof` | timing per stage (`[PROF]` lines) |
| `nodepth` | flow only |
| `track=adaptive` / `track=single` | reference tracking: paper flow chaining / single reference (key `t`) |
| `detail`, `detail=residual`, `detail=depth` | texture DETAIL panel (key `x`) |
| `--depth-refine=on` | start with depth refinement (default off = calibrated raw depth, key `f`) |
| `--depth-calib=FILE` / `--depth-calib=none` | depth→mm file / relative units (default `host/calib/depth_calib_<camera>.json`, else built-in **synthetic** values) |
| `--flow-vis=3` | flow colour scale: full brightness at N px (fixed scale) |
| `--display-hz=N` | screen refresh (single default 30, dual 10); processing stays at the camera rate |
| `--flow=medium` | flow method: `medium`, `medium_ref`, `farneback`, `pcaflow`, `tvl1`, `ultrafast`, `fast` |
| `--usb-buffers=N` | V4L2 buffers per USB camera (2-10, default 8; OpenCV's own default is 4). More buffers = fewer dropped frames when the host is briefly late |
| `--slip-max=N` | full scale of the \|F\| bar / slip bins (px) |
| `--sensor-w=N`, `--focal=N` | undistort focal length (Zero sensor width 1280 = 640×480 crop) / force focal (px) |
| `ram`, `--pre=N`, `--out=DIR`, `--fmt=npy` | raw recording ('v'): frames in RAM, N s pre-trigger, output folder, format |
| `--window=x0,y0,x1,y1` / `--window=off` | gel window in raw-frame pixels (everything outside is blanked before processing) |

### Adaptive referencing (automatic re-baseline, key `w`)
| Option | Default | Meaning |
|---|---|---|
| `--aref=off` | on | disable (only `r`) |
| `--aref-release-still=S` | 0.4 | fallback stillness before the release reference (s) |
| `--aref-follow=S` | 6 | after a release, keep removing the gel's creep for S seconds (0 = off) |
| `--aref-follow-px=P` | 0.15 | residual (p98, px) that triggers a follow-up reference |
| `--aref-leftover=F` | 0.4 | release = flow below F × peak (lower = later reference) |
| `--aref-shift=1` | off | > 60 % of the gel moves the SAME way by > 0.5 px for 1 s (camera/gel/mount moved) → new reference, also while 'contact' is reported (0 = off) |
| `--aref-idle=2` | off | no contact and flow left (p98 > `--aref-idle-px`, 0.15 px) for 2 s → that flow is noise → new reference (0 = off) |
| `--aref-residue=2,0.1,0.25` | off | during contact, re-reference only RESIDUE: gel still ≥ 2 s, max depth < 0.1 mm, flow < 25 % of the press peak (`off` = never) |
| `--aref-leftover-contact=on` | off | also re-reference while contact is still reported once the flow fell below F × peak (old behaviour; can reset a lightly held object) |
| `--aref-still=S` | 1.0 | stillness needed for idle re-references (s) |
| `--aref-resid=P` | 0.35 | idle residual flow threshold (p90, px) |
| `--aref-period=S` | 20 | periodic idle re-reference (s, 0 = off) |
| `--aref-stuck=S` | 0 | re-reference a contact unchanged for S s (0 = off) |

The terminal shows each reference: `[AREF] new reference: released …`, `release follow-up …`, `residual …`, `periodic …`.

### ROS 2 (`ros`)
| Option | Meaning |
|---|---|
| `ros` | publish under `/tactile/<camera>/` (contact, wrench, contact_point, slip, event, state, depth) |
| `--gel-mm=W` | gel width seen in the ROI (mm) → positions and shear in mm |
| `--force-gain=G` | N per unit indentation volume → force in N |
| `--slip-thr=T` | slip threshold |
| `--ros-img-hz=H` | depth image rate (default 5, 0 = off) |
| `--ros-ns=NAME` | topic prefix (default `/tactile`) |
| `--udp[=HOST:PORT]` | also send the features as JSON over UDP (default `127.0.0.1:9870`, no ROS needed; used by `servo_grasp.py`) |

### Single camera only (`./run.sh usb …`)
| Option | Meaning |
|---|---|
| `--usb=camera1` | camera by hostname or `/dev/...` |
| `--usb-size=384x288`, `--usb-fps=10` | other USB resolution / frame rate (default 640×480 @ 15) |
| `--depth-stride=N` | depth every N frames |
| `3d` / `o3d` | OpenCV 3D depth view / Open3D window |
| `nopardepth` | depth inline instead of its own process |
| `--calib-step=0.1`, `--calib-metric=max\|p99` | depth calibration step (mm) and metric |
| `--lens=checker` | checkerboard lens calibration (CSI `full` profile) |
| `--slip-grid=7x9` | slip feature grid (rows x columns) |
| `turbo`, `fast`, `quality`, `full`, `cam1` | CSI camera profiles on a Pi 5 (not for USB) |

### Dual / triple only (`./run.sh dual …`, `./run.sh triple …`)
| Option | Meaning |
|---|---|
| `--usb=camera1 --usb2=camera2` | the two Zeros (`usb2` alone = second camera found automatically) |
| `triple`, `--usb3=camera3` | three cameras; cameras not named are filled in from those found |
| `--depth-hz=N` | depth rate per camera (default 7.5; 15 or 0 = every frame) |
| `--dis=full` | deltact_plus DIS settings (default in dual: lighter grad 12 / var 2) |
| `nogui`, `--seconds=N` | no window, run N seconds (measurement) |
| `threads` | old single-process mode |
| `nopardepth` | depth inside each camera process |
| `--cv-threads=N` | OpenCV threads per camera process |
| `nocalib` | relative depth units |

---

## 4. Keys

| Key | Action |
|---|---|
| `r` | new baseline (**no contact**) |
| `d` | re-detect the gel ROI (adjust `roi_thresh` first) |
| `u` | clean controls (deadzone/EMA 0, tavg 1, no rebaseline); single + CSI: measure mode |
| `c` | save config (`host/sensor_config.json`): sensor settings + trackbars (roi_thresh per camera, gain, bright, arrows, suppression). Loaded at the next start **without** `clean` |
| `t` | tracking single ↔ adaptive |
| `w` | adaptive referencing on/off |
| `f` | depth refinement on/off |
| `x` | DETAIL panel: off → fine → residual → depth |
| `a` | flow arrows |
| `g` / `k` | slip-feature log start/stop / label static ↔ slip |
| `v` | raw frame recording start/stop |
| `s` / `b` | snapshot / burst of 10 raw frames |
| `l` | per-frame depth log (for `mts_calib.py`) |
| `p` / `z` / `o` | depth calibration: add point / undo / fit & save |
| `e`, `[` `]`, `-` `=` | CSI only: lock exposure/gain, exposure −/+, gain −/+ |
| `m`, `n` | single: no-contact probe recording, noise floor test |
| `q` / ESC | quit |

---

## 5. Typical sessions

```bash
# first start of a sensor
./run.sh usb --usb=camera1 clean          # adjust roi_thresh, d, release everything, r, c

# daily use, Jetson, two sensors
./run.sh dual --usb=camera1 --usb2=camera2 gpu --depth-hz=15

# robot integration
source /opt/ros/humble/setup.bash
./run.sh dual --usb=camera1 --usb2=camera2 gpu ros --gel-mm=20

# depth -> mm calibration with the MTS
./run.sh usb --usb=camera1                # r, then l (log) while the MTS runs, l again
python3 host/mts_calib.py host/calib/logs/depthlog_camera1_*.csv exp1_test1_noreset.txt \
        --out=host/calib/depth_calib_camera1.json

# a camera drops frames: is it the camera or the USB port?
v4l2-ctl -d /dev/v4l/by-id/usb-camera1_Tactile_UVC_Cam_e19b63bd-video-index0 --stream-mmap --stream-count=300
```
Two cameras through one hub share its bandwidth: plug each Zero into its own port on the host.

# ML DATA from the grasp logs (samples / sliding windows with future targets / attempts):
#   cd host && python3 grasp_to_ml.py grasp_logs/grasp_*.csv --out=ml_data   [--win=0.5 --stride=0.1 --horizon=0.5]
#   split train/test BY attempt_uid; label attempts in the GUI with 1 / 0 / 2 for a manual outcome
# baseline models (grouped CV by attempt; <= --cap windows per attempt so one long hold does not dominate):
#   cd host && python3 grasp_train.py --data=ml_data [--save]      (pip install scikit-learn if missing)
# LIVE ML in the grasp (models from grasp_train.py --save; rebuild the data with --overload=0.8 first):
#   ./run.sh --script=servo_grasp.py --sensor=triple ... --ml=host/ml_data/models          (acts)
#   ./run.sh --script=servo_grasp.py --sensor=triple ... --ml=host/ml_data/models --ml-show (only shows / logs)
# OPTIMAL GRIP: lowest grip without slip (x1.3 margin), learned per object, saved in host/grasp_logs/optimal_grip.json
#   ./run.sh --script=servo_grasp.py --sensor=triple ... --optimal --object=cup [--ml=host/ml_data/models]
#   when the optimal grip is reached everything stops adjusting (HOLD); --opt-keep = keep adapting
# SYMMETRIC STOP (default): all fingers touch + ID1-3 within --stop-spread=3 deg of each other + ID4-6 within
#   --stop-spread-rot=6 deg of each other -> HOLD (no time limit); --no-stop keeps searching / adjusting
# TACTILE GRIP (default): grip follows the gel's SENSING - shear grows (tangential load) -> grip harder,
#   slip -> grip harder at once, load gone -> back down; pad-edge contact reported; the base uses contact centre
#   and shear too. --tg-shear-k=0.15 --tg-slip=0.25 --tg-max=1.6, --no-tactile-grip = off
# SENSOR REBASE (default): fingers fully open -> servo_grasp tells the sensor to reset its baseline (UDP 'reset'
#   on 127.0.0.1:9871, same as key r), again 3 s later: no phantom touch from the gel's leftover imprint. --no-rebase = off
# SENSING HOLD (default): HOLD stops searching but keeps the tactile grip (shear / slip -> tighter, <= 2 deg/s)
#   and goes back to GRASP when a held finger loses contact for 0.5 s; --hold-frozen = old frozen HOLD.
#   Edge contact -> +10 % grip; all contacts on the same pad edge -> [TACTILE] hint to place the object differently
# PRE-TOUCH: gel feels >= 0.03 mm -> that finger slows to 2 deg/s (--pre-depth, --pre-speed); contact release needs 3 frames
# base never spreads ID4-6 beyond --stop-spread-rot; centring stuck + depths balanced -> 'most symmetric reachable' HOLD
# while grasping the sensor keeps its reference (UDP 'aref hold' -> adaptive referencing off; 'aref release'
#   restores it when the fingers are open). --no-rebase turns off both the reset and this.
# INDUSTRIAL MOTION: trapezoidal profile (--accel=200 deg/s^2); --fast=45 deg/s in free space (opening and up to the
#   pre-grasp angle = learned contact angle of this --object - 8 deg, else closed - 25 deg), then --speed, then --pre-speed
#   smooth: one velocity ramp per finger, a finger lagging > 4 deg behind is slowed (not stopped) -> no stop-and-go
# ONE-FINGER RE-GRASP (default, --no-regrasp = off): the worst finger opens 8 deg ALONE (others keep holding),
#   its rotation joint (--rot-map=1:4,2:5,3:6) turns +-3 deg, it closes again; kept only if the pose got better.
#   Triggers (2 s): finger presses off its pad, contact at the pad edge (> 0.8), finger angles > --regrasp-spread=10 deg
#   apart; or no HOLD --regrasp-after=10 s after the first contact. <= 2 tries per finger, 4 per grasp.
#   --regrasp-open=8 --regrasp-rot=3 --regrasp-max=2 --regrasp-total=4 --regrasp-wait=2 --regrasp-edge=0.8
