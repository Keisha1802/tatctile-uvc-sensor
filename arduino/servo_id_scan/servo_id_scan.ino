// servo_id_scan.ino — find and identify Waveshare/Feetech ST/SC bus servos (ST3215, STS...)
// Board: Waveshare "Servo Driver with ESP32" (servo bus on GPIO 18 = RX, 19 = TX, 1 Mbit/s).
// If your own sketch used other pins/baud for the servo bus, change S_RXD / S_TXD / BUS_BAUD.
// Library: SCServo (Waveshare/Feetech, "SCServo" in the Arduino library manager, or the zip
// from the Waveshare wiki).  Servos need the board's DC input (USB powers only the ESP32).
//
// Serial Monitor at 115200, line ending "Newline". Commands:
//   s            scan IDs 0..253 (also done at start)
//   w            watch: position/load of every servo found, 5x per second
//                (move or squeeze ONE finger by hand: the ID whose numbers change is that one)
//                any key stops it
//   g <id>       wiggle servo <id>: +-60 steps (~5 deg) and back
//   g all        wiggle each servo in turn
//   t <id> 0|1   torque off / on
// Nothing moves except with 'g'. Check nothing can be pinched before 'g'.
#include <SCServo.h>

#define S_RXD    18
#define S_TXD    19
#define BUS_BAUD 1000000
#define MAX_ID   253
#define WIGGLE   60

SMS_STS st;
int found[MAX_ID + 1];
int nFound = 0;

void scan() {
  nFound = 0;
  Serial.println("[SCAN] IDs 0..253 ...");
  for (int id = 0; id <= MAX_ID; id++) {
    if (st.Ping(id) != -1) {
      found[nFound++] = id;
      int pos = st.ReadPos(id);
      int load = st.ReadLoad(id);
      int volt = st.ReadVoltage(id);
      int temp = st.ReadTemper(id);
      Serial.printf("  ID %3d: found   position %5d  load %+5d  %.1f V  %d C\n",
                    id, pos, load, volt / 10.0, temp);
    }
  }
  if (nFound == 0)
    Serial.println("[SCAN] no servo answered: DC power on? pins/baud right (S_RXD/S_TXD/BUS_BAUD)?");
  else
    Serial.printf("[SCAN] %d servo(s)\n", nFound);
}

void watch() {
  int ref[MAX_ID + 1];
  for (int i = 0; i < nFound; i++) ref[i] = st.ReadPos(found[i]);
  Serial.println("[WATCH] move ONE finger by hand; '<<' marks the one that moved. Any key = stop.");
  while (!Serial.available()) {
    for (int i = 0; i < nFound; i++) {
      int id = found[i];
      int pos = st.ReadPos(id), load = st.ReadLoad(id);
      int d = pos - ref[i];
      Serial.printf("ID%d: %5d (%+5d) load %+5d%s | ", id, pos, d, load,
                    (abs(d) > 15 || abs(load) > 80) ? " <<" : "");
    }
    Serial.println();
    delay(200);
  }
  while (Serial.available()) Serial.read();
}

void wiggle(int id) {
  int p0 = st.ReadPos(id);
  if (p0 < 0) { Serial.printf("  ID %d: no answer\n", id); return; }
  Serial.printf("  ID %d: wiggle around %d\n", id, p0);
  int targets[3] = {p0 + WIGGLE, p0 - WIGGLE, p0};
  for (int k = 0; k < 3; k++) {
    int t = constrain(targets[k], 0, 4095);
    st.WritePosEx(id, t, 600, 50);          // speed 600 steps/s, acceleration 50
    delay(400);
  }
  Serial.printf("  ID %d: back at %d\n", id, st.ReadPos(id));
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(BUS_BAUD, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(1000);
  scan();
  Serial.println("commands: s | w | g <id> | g all | t <id> 0|1");
}

void loop() {
  if (!Serial.available()) return;
  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line == "s") scan();
  else if (line == "w") watch();
  else if (line == "g all") { for (int i = 0; i < nFound; i++) { wiggle(found[i]); delay(2000); } }
  else if (line.startsWith("g ")) wiggle(line.substring(2).toInt());
  else if (line.startsWith("t ")) {
    int sp = line.indexOf(' ', 2);
    int id = line.substring(2, sp).toInt(), on = line.substring(sp + 1).toInt();
    st.EnableTorque(id, on);
    Serial.printf("  ID %d torque %s\n", id, on ? "ON" : "OFF");
  }
  else if (line.length()) Serial.println("commands: s | w | g <id> | g all | t <id> 0|1");
}
