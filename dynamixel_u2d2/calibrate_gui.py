#!/usr/bin/env python3
"""
KALIBRASI ENCODER GRIPPER - GUI
================================
  ./run_gripper.sh calibrate_gui.py

1. TORQUE OFF  -> jari bebas digerakkan dengan tangan, angka posisi ikut berubah
2. atur ROTASI lurus + jari TERBUKA penuh  -> klik "SET HOME / OPEN"
      (rotasi ID 1-3 jadi 2048, fleksi ID 4-6 jadi 1700; per servo: tombol "set" di barisnya)
3. klik "REKAM CLOSE", tutup tiap jari penuh dengan tangan sampai mentok, klik "STOP REKAM"
4. klik "SIMPAN"  -> calib.json (dipakai chromasense_gui.py)

Homing offset ditulis ke EEPROM servo (permanen). Pasangan: ROT 1 + FLEX 4, ROT 2 + FLEX 5, ROT 3 + FLEX 6.
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
        self.setWindowTitle("Kalibrasi encoder gripper")
        self.resize(760, 560)
        self.setStyleSheet("QWidget{font-size:15px} QPushButton{padding:8px}")
        v = QVBoxLayout(self)

        self.info = QLabel()
        self.info.setWordWrap(True)
        v.addWidget(self.info)

        g = QGridLayout()
        for c, h in enumerate(["ID", "sendi", "posisi", "", "offset", "close", "", "status"]):
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
            btn.setToolTip("jadikan posisi ini 2048 (rotasi) / 1700 (fleksi terbuka) untuk servo ini saja")
            btn.clicked.connect(lambda _, i=i: self.set_home([i]))
            stt = QPushButton("OK")
            stt.setToolTip("hardware error servo; klik = reboot servo ini (hapus error, torque tetap off)")
            stt.clicked.connect(lambda _, i=i: self.reboot([i]))
            for c, wd in enumerate([QLabel(f"<b>{i}</b>"), QLabel(ce.NAME[i]), pos, bar, off, cl, btn, stt]):
                g.addWidget(wd, r, c)
            self.w[i] = dict(pos=pos, bar=bar, off=off, close=cl, stat=stt)
        v.addLayout(g)

        h = QHBoxLayout()
        self.b_torque = QPushButton("1. TORQUE OFF")
        self.b_torque.clicked.connect(self.torque_off)
        self.b_home = QPushButton("2. SET HOME / OPEN (semua)")
        self.b_home.clicked.connect(lambda: self.set_home(self.ids))
        self.b_rec = QPushButton("3. REKAM CLOSE")
        self.b_rec.clicked.connect(self.toggle_rec)
        self.b_save = QPushButton("4. SIMPAN")
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
        self.step_text("Klik <b>1. TORQUE OFF</b>, lalu atur rotasi LURUS dan semua jari TERBUKA penuh dengan tangan.")

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

    ERR_BITS = {0: "tegangan", 2: "panas", 3: "ENCODER", 4: "listrik", 5: "OVERLOAD"}

    def reboot(self, ids):
        for i in ids:
            ok = self.bus.reboot(i)
            self.bus.torque(i, False)
            e = self.bus.hw_error(i)
            self.say(f"ID {i}: reboot {'OK' if ok else 'gagal'} - hardware error sekarang {e}")
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
            if self.rec and i in ce.FLEX_IDS:
                b = self.best.get(i, ce.FLEX_OPEN)
                if abs(p - ce.FLEX_OPEN) > abs(b - ce.FLEX_OPEN):
                    self.best[i] = p
                b = self.best.get(i, ce.FLEX_OPEN)
                W["close"].setText(f"{b}  ({abs(b - ce.FLEX_OPEN) * TICK_DEG:.0f}°)")

    # ---------------------------------------------------------------- actions
    def torque_off(self):
        bad = [i for i in self.ids if not self.bus.torque(i, False)]
        self.say("torque OFF" + (f" - gagal di ID {bad}" if bad else " - semua jari bebas"))
        self.step_text("Atur <b>rotasi ID 1-3 LURUS</b> (grip circular) dan <b>ketiga jari TERBUKA penuh</b>, "
                       "lalu klik <b>2. SET HOME / OPEN</b> (atau 'set' per servo).")

    def set_home(self, ids):
        for i in ids:
            self.bus.torque(i, False)
            target = ce.ROT_TARGET if i in ce.ROT_IDS else ce.FLEX_OPEN
            pos, old = self.bus.pos(i), self.bus.offset(i)
            if pos is None or old is None:
                self.say(f"ID {i}: gagal baca")
                continue
            self.raw_open[i] = pos - old
            new = old + (target - pos)
            ok = self.bus.set_offset(i, new)
            time.sleep(0.08)
            chk = self.bus.pos(i)
            good = ok and chk is not None and abs(chk - target) <= 5
            self.say(f"ID {i} ({ce.NAME[i]}): {pos} -> {chk}  offset {old} -> {new}  "
                     + ("<b style='color:green'>OK</b>" if good else "<b style='color:#c00'>GAGAL</b>"))
            if not good and ok:
                # the offset was accepted but the position did not move: the servo reports a FROZEN position
                # (hardware error, e.g. encoder / overload) -> put the old offset back, reboot it, retry
                self.bus.set_offset(i, old)
                e = self.bus.hw_error(i)
                self.say(f"<b style='color:#c00'>ID {i}: posisi tidak ikut berubah (hardware error {e}) -> offset "
                         f"dikembalikan ke {old}. Klik tombol status / reboot di baris ID {i}, lalu 'set' lagi."
                         f" Masih gagal: matikan-nyalakan power servo.</b>")
        self.refresh_offsets()
        self.step_text("Klik <b>3. REKAM CLOSE</b>, lalu <b>tutup tiap jari penuh</b> dengan tangan sampai mentok "
                       "(satu per satu boleh), lalu klik <b>STOP REKAM</b>.")

    def toggle_rec(self):
        if not self.rec:
            self.best = {i: ce.FLEX_OPEN for i in ce.FLEX_IDS}
            self.rec = True
            self.b_rec.setText("STOP REKAM")
            self.b_rec.setStyleSheet("background:#f3c; color:white; font-weight:bold")
            self.say("merekam: tutup tiap jari penuh dengan tangan ...")
            return
        self.rec = False
        self.b_rec.setText("3. REKAM CLOSE")
        self.b_rec.setStyleSheet("")
        for i in ce.FLEX_IDS:
            d = self.best[i] - ce.FLEX_OPEN
            if abs(d) < 150:
                self.say(f"ID {i}: jari hampir tidak ditutup ({self.best[i]}) - tidak dipakai")
                continue
            close = int(self.best[i] - ce.CLOSE_MARGIN * (1 if d > 0 else -1))
            self.limits[i] = [ce.FLEX_OPEN, close]
            self.w[i]["close"].setText(f"<b>{close}</b> ({abs(close - ce.FLEX_OPEN) * TICK_DEG:.0f}°)")
            ro = self.raw_open.get(i)
            if ro is not None and not (0 <= ro + d <= 4095):
                self.say(f"<b style='color:#c00'>ID {i}: rentang jari melewati titik 0/4095 encoder - setelah power "
                         f"mati bacaan bisa loncat 4096. Pasang ulang horn (hold_1700.py)</b>")
            else:
                self.say(f"ID {i}: CLOSE = {close} ({abs(close - ce.FLEX_OPEN) * TICK_DEG:.0f}° dari open)")
        self.step_text("Klik <b>4. SIMPAN</b>. Setelah itu tutup jendela ini dan buka GUI: <b>./run_gripper.sh</b>")

    def save(self):
        old = {}
        try:
            old = json.load(open(ce.CALIB))
        except (OSError, ValueError):
            pass
        old.setdefault("flex_limits", {}).update({str(k): v for k, v in self.limits.items()})
        old.setdefault("rot_home", {}).update({str(i): ce.ROT_TARGET for i in ce.ROT_IDS})
        old["offsets"] = {**old.get("offsets", {}), **{str(i): self.bus.offset(i) for i in self.ids}}
        old["date"] = time.strftime("%Y-%m-%d %H:%M")
        json.dump(old, open(ce.CALIB, "w"), indent=1)
        self.say(f"<b>tersimpan</b> {ce.CALIB}: " + ", ".join(f"ID {k} {v[0]}-{v[1]}"
                                                         for k, v in sorted(old['flex_limits'].items())))
        QMessageBox.information(self, "Kalibrasi", "Tersimpan. Tutup jendela ini, lalu jalankan ./run_gripper.sh")

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
        QMessageBox.warning(None, "Servo", f"ID {missing} tidak menjawab.\nJalankan: ./run_gripper.sh setup_ids.py check")
    w = CalibGUI(bus)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
