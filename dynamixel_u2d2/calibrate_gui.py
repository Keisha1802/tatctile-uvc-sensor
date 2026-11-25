#!/usr/bin/env python3
"""
GRIPPER ENCODER CALIBRATION - GUI
=================================
  ./run_gripper.sh calibrate_gui.py

1. TORQUE OFF  -> every servo's homing offset back to 0 (the encoder's own reading), fingers move freely by hand
2. set the ROTATIONS straight + fingers fully OPEN  -> click "SET HOME / OPEN"
      the encoder reading NOW is recorded as each joint's zero (one servo: the "set" button in its row)
      - nothing is written to EEPROM (25 Nov: offsets were accepted but the servo position did not change)
3. click "RECORD CLOSE", close every finger fully by hand to its stop, click "STOP RECORDING"
4. click "SAVE"  -> calib.json (used by chromasense_gui.py)

Pairs: ROT 1 + FLEX 4, ROT 2 + FLEX 5, ROT 3 + FLEX 6.
"""
import json
import os
import sys
import time

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                             QPushButton, QVBoxLayout, QWidget)

import calibrate_encoders as ce

TICK_DEG = 360.0 / 4096


class CalibGUI(QWidget):
    def __init__(self, bus):
        super().__init__()
        self.bus = bus
        self.ids = ce.ROT_IDS + ce.FLEX_IDS
        self.raw_open = {}                     # encoder raw at the open pose (wrap check)
        self.best = {}                         # farthest closed reading while recording
        self.rec = False
        self.limits = {}
        self.setWindowTitle("Gripper encoder calibration")
        self.resize(760, 560)
        self.setStyleSheet("QWidget{font-size:15px} QPushButton{padding:8px}")
        v = QVBoxLayout(self)

        self.info = QLabel()
        self.info.setWordWrap(True)
        v.addWidget(self.info)

        g = QGridLayout()
        for c, h in enumerate(["ID", "joint", "position", "", "offset / zero", "close", "", "status"]):
            g.addWidget(QLabel(f"<b>{h}</b>"), 0, c)
        self.w = {}
        for r, i in enumerate(self.ids, 1):
            pos = QLabel("-")
            pos.setStyleSheet("font-size:22px; font-weight:bold; min-width:70px")
            bar = QProgressBar()
            bar.setRange(0, 4095)
            bar.setTextVisible(False)
            off = QLabel("-")
            cl = QLabel("-")
            btn = QPushButton("set")
            btn.setToolTip("record the present reading as this servo's zero (rotation straight / finger open)")
            btn.clicked.connect(lambda _, i=i: self.set_home([i]))
            stt = QPushButton("OK")
            stt.setToolTip("servo hardware error; click = reboot this servo (clears the error, torque stays off)")
            stt.clicked.connect(lambda _, i=i: self.reboot([i]))
            for c, wd in enumerate([QLabel(f"<b>{i}</b>"), QLabel(ce.NAME[i]), pos, bar, off, cl, btn, stt]):
                g.addWidget(wd, r, c)
            self.w[i] = dict(pos=pos, bar=bar, off=off, close=cl, stat=stt)
        v.addLayout(g)

        h = QHBoxLayout()
        self.b_torque = QPushButton("1. TORQUE OFF")
        self.b_torque.clicked.connect(self.torque_off)
        self.b_home = QPushButton("2. SET HOME / OPEN (all)")
        self.b_home.clicked.connect(lambda: self.set_home(self.ids))
        self.b_rec = QPushButton("3. RECORD CLOSE")
        self.b_rec.clicked.connect(self.toggle_rec)
        self.b_save = QPushButton("4. SAVE")
        self.b_save.clicked.connect(self.save)
        for b in (self.b_torque, self.b_home, self.b_rec, self.b_save):
            h.addWidget(b)
        v.addLayout(h)

        self.log = QLabel()
        self.log.setWordWrap(True)
        self.log.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.log.setStyleSheet("font-size:13px; color:#333")
        v.addWidget(self.log)
        self.lines = []
        self.step_text("Click <b>1. TORQUE OFF</b>, then set the rotations STRAIGHT and every finger fully OPEN by hand.")

        self.t = QTimer(self)
        self.t.timeout.connect(self.tick)
        self.t.start(150)
        self.refresh_offsets()

    # ---------------------------------------------------------------- helpers
    def say(self, msg):
        self.lines = (self.lines + [msg])[-7:]
        self.log.setText("<br>".join(self.lines))

    def step_text(self, html):
        self.info.setText(f"<div style='font-size:16px; background:#fff6d5; padding:6px'>{html}</div>")

    def refresh_offsets(self):
        for i in self.ids:
            o = self.bus.offset(i)
            self.w[i]["off"].setText("-" if o is None else str(o))

    ERR_BITS = {0: "VOLTAGE", 2: "HEAT", 3: "ENCODER", 4: "SHOCK", 5: "OVERLOAD"}

    def reboot(self, ids):
        for i in ids:
            ok = self.bus.reboot(i)
            self.bus.torque(i, False)
            e = self.bus.hw_error(i)
            self.say(f"ID {i}: reboot {'OK' if ok else 'failed'} - hardware error now {e}")
        self.refresh_offsets()

    def tick(self):
        self.n_tick = getattr(self, "n_tick", 0) + 1
        if self.n_tick % 7 == 1:                       # ~1 s: hardware error status of every servo
            for i in self.ids:
                e = self.bus.hw_error(i)
                b = self.w[i]["stat"]
                if e:
                    names = [n for k, n in self.ERR_BITS.items() if e >> k & 1] or [str(e)]
                    b.setText("ERR " + "+".join(names) + " (reboot)")
                    b.setStyleSheet("background:#c00; color:white; font-weight:bold")
                else:
                    b.setText("OK" if e == 0 else "?")
                    b.setStyleSheet("")
        for i in self.ids:
            p = self.bus.pos(i)
            W = self.w[i]
            if p is None:
                W["pos"].setText("ERR")
                W["pos"].setStyleSheet("font-size:22px; font-weight:bold; color:#c00; min-width:70px")
                continue
            W["pos"].setText(str(p))
            W["pos"].setStyleSheet("font-size:22px; font-weight:bold; min-width:70px")
            W["bar"].setValue(max(0, min(4095, p)))
            h0 = getattr(self, "home", {}).get(i)
            if h0 is not None:                         # angle from the recorded zero
                W["pos"].setText(f"{p}  ({(p - h0) * TICK_DEG:+.0f}°)")
            if self.rec and i in ce.FLEX_IDS and h0 is not None:
                b = self.best.get(i, h0)
                if abs(p - h0) > abs(b - h0):
                    self.best[i] = p
                b = self.best.get(i, h0)
                W["close"].setText(f"{b}  ({abs(b - h0) * TICK_DEG:.0f}°)")

    # ---------------------------------------------------------------- actions
    def torque_off(self):
        bad = [i for i in self.ids if not self.bus.torque(i, False)]
        self.say("torque OFF" + (f" - failed on ID {bad}" if bad else " - every finger moves freely"))
        # back to the encoder's own reading: homing offset 0 (undoes every earlier offset attempt)
        for i in self.ids:
            o = self.bus.offset(i)
            if o:
                ok = self.bus.set_offset(i, 0)
                self.say(f"ID {i}: homing offset {o} -> 0 {'OK' if ok else 'FAILED'}")
        self.home = {}
        self.refresh_offsets()
        self.step_text("Set <b>rotations ID 1-3 STRAIGHT</b> (circular grip) and <b>all three fingers fully OPEN</b>, "
                       "then click <b>2. SET HOME / OPEN</b> (or 'set' per servo).")

    def set_home(self, ids):
        """SOFTWARE zero: the encoder reading NOW is this joint's reference (rotation straight / finger open).
        Nothing is written to the servo; the GUI maps its values onto these readings via calib.json."""
        if not hasattr(self, "home"):
            self.home = {}
        for i in ids:
            pos, off = self.bus.pos(i), self.bus.offset(i)
            if pos is None:
                self.say(f"ID {i}: read failed")
                continue
            self.home[i] = pos
            self.raw_open[i] = pos - (off or 0)
            e = self.bus.hw_error(i)
            self.w[i]["off"].setText(f"zero {pos}")
            if i in ce.ROT_IDS and not (0 <= pos - 648 and pos + 852 <= 4095):
                # the GUI turns a rotation 648 ticks one way / 852 the other way from straight: that range must
                # not cross the encoder's 0/4095 point, else after a power cycle the reading jumps 4096
                self.say(f"<b style='color:#b60'>ID {i}: zero {pos} is near the encoder's end (0/4095). That is fine: "
                         f"chromasense_gui.py adapts at start ('frame moved'). Cleaner: refit the horn so this "
                         f"position is near 2048.</b>")
            self.say(f"ID {i} ({ce.NAME[i]}): zero = {pos} "
                     + ("<b style='color:green'>OK</b>" if not e else f"<b style='color:#c00'>hardware error {e}"
                        " - reboot this servo (status button), then 'set' again</b>"))
        self.step_text("Click <b>3. RECORD CLOSE</b>, then <b>close every finger fully</b> by hand to its stop "
                       "(one at a time is fine), then click <b>STOP RECORDING</b>.")

    def toggle_rec(self):
        if not self.rec:
            hm = getattr(self, "home", {})
            if not all(i in hm for i in ce.FLEX_IDS):
                self.say("<b>set the open zero of the fingers first (step 2)</b>")
                return
            self.best = {i: hm[i] for i in ce.FLEX_IDS}
            self.rec = True
            self.b_rec.setText("STOP RECORDING")
            self.b_rec.setStyleSheet("background:#f3c; color:white; font-weight:bold")
            self.say("recording: close every finger fully by hand ...")
            return
        self.rec = False
        self.b_rec.setText("3. RECORD CLOSE")
        self.b_rec.setStyleSheet("")
        for i in ce.FLEX_IDS:
            h0 = self.home[i]
            d = self.best[i] - h0
            if abs(d) < 150:
                self.say(f"ID {i}: finger hardly closed ({self.best[i]}) - not used")
                continue
            close = int(self.best[i] - ce.CLOSE_MARGIN * (1 if d > 0 else -1))
            self.limits[i] = [h0, close]
            self.w[i]["close"].setText(f"<b>{close}</b> ({abs(close - h0) * TICK_DEG:.0f}°)")
            ro = self.raw_open.get(i)
            if ro is not None and not (0 <= ro + d <= 4095):
                self.say(f"<b style='color:#b60'>ID {i}: the finger's range crosses the encoder's 0/4095 point - "
                         f"chromasense_gui.py handles it at start; cleaner: refit the horn with hold_1700.py</b>")
            else:
                self.say(f"ID {i}: OPEN {h0}  CLOSE {close} ({abs(close - h0) * TICK_DEG:.0f}°)")
        self.step_text("Click <b>4. SAVE</b>. Then close this window and open the GUI: <b>./run_gripper.sh</b>")

    def save(self):
        old = {}
        try:
            old = json.load(open(ce.CALIB))
        except (OSError, ValueError):
            pass
        hm = getattr(self, "home", {})
        old.setdefault("flex_limits", {}).update({str(k): v for k, v in self.limits.items()})
        old.setdefault("rot_home", {}).update({str(i): hm[i] for i in ce.ROT_IDS if i in hm})
        old["mode"] = "software zero (homing offset 0)"
        old["offsets"] = {**old.get("offsets", {}), **{str(i): self.bus.offset(i) for i in self.ids}}
        old["date"] = time.strftime("%Y-%m-%d %H:%M")
        json.dump(old, open(ce.CALIB, "w"), indent=1)
        self.say(f"<b>saved</b> {ce.CALIB}: " + ", ".join(f"ID {k} {v[0]}-{v[1]}"
                                                         for k, v in sorted(old['flex_limits'].items())))
        QMessageBox.information(self, "Calibration", "Saved. Close this window, then run ./run_gripper.sh")

    def closeEvent(self, e):
        self.t.stop()
        try:
            self.bus.port.closePort()
        except Exception:
            pass
        e.accept()


def main():
    bus = ce.Bus()
    missing = [i for i in ce.ROT_IDS + ce.FLEX_IDS if bus.pos(i) is None]
    app = QApplication(sys.argv)
    if missing:
        QMessageBox.warning(None, "Servo", f"ID {missing} do not answer.\nRun: ./run_gripper.sh setup_ids.py check")
    w = CalibGUI(bus)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
