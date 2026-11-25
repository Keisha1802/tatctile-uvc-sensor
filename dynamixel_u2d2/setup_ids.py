#!/usr/bin/env python3
"""
RESET THE GRIPPER SERVOS: ID 1-6, baudrate 57600 (Dynamixel Protocol 2.0, U2D2)
================================================================================
Target:  ID 1, 2, 3 = ROTATION fingers 1-3 (XC430)      ID 4, 5, 6 = FLEXION fingers 1-3 (XC330)

  ./run_gripper.sh setup_ids.py check     scan EVERY baudrate + find duplicate IDs (writes nothing)
  ./run_gripper.sh setup_ids.py           wizard: plug in ONE servo, choose its finger -> ID + baud written
  ./run_gripper.sh setup_ids.py one 1     the servo plugged in on its own -> ID 1
  ./run_gripper.sh setup_ids.py remap 6:4,4:5,5:6
                                          swap the IDs of servos that are ALL plugged in (safe: via temporary IDs),
                                          e.g. so the finger pairs become ROT 1 + FLEX 4, ROT 2 + FLEX 5, ROT 3 + FLEX 6

Why one at a time: two servos with the same ID answer together -> the packet is garbled and that servo
cannot be told apart / changed. With only ONE servo on the bus, any ID and baudrate is found.
The ID and baudrate are stored in the servo's EEPROM (permanent); the torque is switched off first (needed
to write EEPROM).
"""
import os
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler

DEVICENAME = os.environ.get("DXL_PORT") or ("/dev/u2d2" if os.path.exists("/dev/u2d2") else "/dev/ttyUSB0")
TARGET_BAUD = 57600
BAUDS = [57600, 1000000, 115200, 2000000, 3000000, 4000000, 9600]     # most common first
BAUD_CODE = {9600: 0, 57600: 1, 115200: 2, 1000000: 3, 2000000: 4, 3000000: 5, 4000000: 6}

ADDR_ID, ADDR_BAUD, ADDR_TORQUE = 7, 8, 64
JOINTS = {1: "ROTATION finger 1", 2: "ROTATION finger 2", 3: "ROTATION finger 3",
          4: "FLEXION finger 1", 5: "FLEXION finger 2", 6: "FLEXION finger 3"}
MODELS = {1060: "XL430-W250", 1070: "XC430-W150", 1080: "XC430-W240", 1020: "XM430-W350", 1030: "XM430-W210",
          1190: "XC330-M077", 1200: "XC330-M181", 1240: "XC330-M288", 1210: "XC330-T181", 1220: "XC330-T288"}


def model_name(m):
    return MODELS.get(m, f"model {m}")


def kind(m):
    """rotation (430 series) or flexion (330 series)"""
    name = MODELS.get(m, "")
    return "rot" if "430" in name else ("flex" if "330" in name else "?")


class Bus:
    def __init__(self, dev=DEVICENAME):
        self.port = PortHandler(dev)
        self.pk = PacketHandler(2.0)
        try:
            ok = self.port.openPort()
        except Exception as e:
            ok = False
            print(f"!! {dev}: {e}")
        if not ok:
            sys.exit(f"!! cannot open {dev} (U2D2 plugged in? another program on the port? DXL_PORT=...)")
        self.baud = None
        print(f"U2D2: {dev}")

    def set_baud(self, b):
        if b != self.baud:
            self.port.setBaudRate(b)
            self.baud = b
            time.sleep(0.02)

    def scan(self, baud):
        """{id: model} at this baudrate (broadcast ping; fallback: ping 0-252 one by one)"""
        self.set_baud(baud)
        found = {}
        try:
            data, res = self.pk.broadcastPing(self.port)
            if res == COMM_SUCCESS:
                found = {int(i): int(v[0]) for i, v in data.items()}
                return found
        except Exception:
            pass
        for i in range(0, 253):
            m, res, err = self.pk.ping(self.port, i)
            if res == COMM_SUCCESS:
                found[i] = m
        return found

    def scan_all(self, quiet=False):
        """[(baud, id, model)] over every baudrate"""
        out = []
        for b in BAUDS:
            f = self.scan(b)
            if not quiet:
                print(f"  {b:>8} baud: " + (", ".join(f"ID {i} ({model_name(m)})" for i, m in sorted(f.items()))
                                            if f else "-"))
            out += [(b, i, m) for i, m in sorted(f.items())]
        return out

    def write1(self, i, addr, v, what):
        res, err = self.pk.write1ByteTxRx(self.port, i, addr, v)
        if res != COMM_SUCCESS or err:
            print(f"!! ID {i} {what}: {self.pk.getTxRxResult(res) if res != COMM_SUCCESS else self.pk.getRxPacketError(err)}")
            return False
        return True

    def comm_errors(self, i, n=10):
        """how many of n reads fail: > 0 at a found ID = two servos answering together (duplicate ID)"""
        bad = 0
        for _ in range(n):
            _, res, _ = self.pk.read1ByteTxRx(self.port, i, ADDR_ID)
            bad += res != COMM_SUCCESS
        return bad

    def assign(self, baud, old_id, new_id):
        """one servo at (baud, old_id) -> new_id at TARGET_BAUD, verified"""
        self.set_baud(baud)
        if not self.write1(old_id, ADDR_TORQUE, 0, "torque off"):
            return False
        if old_id != new_id:
            if not self.write1(old_id, ADDR_ID, new_id, f"ID {old_id} -> {new_id}"):
                return False
            time.sleep(0.05)
        if baud != TARGET_BAUD:
            if not self.write1(new_id, ADDR_BAUD, BAUD_CODE[TARGET_BAUD], f"baud -> {TARGET_BAUD}"):
                return False
            time.sleep(0.05)
        self.set_baud(TARGET_BAUD)
        m, res, _ = self.pk.ping(self.port, new_id)
        ok = res == COMM_SUCCESS
        print(f"   verify: ID {new_id} @ {TARGET_BAUD} -> {'OK ' + model_name(m) if ok else '!! no answer'}")
        return ok

    def close(self):
        self.port.closePort()


def check(bus):
    print("\nScanning every baudrate (a few seconds) ...")
    found = bus.scan_all()
    print()
    if not found:
        print("!! no servo at all: check the servo power supply and the 3-pin cable from the U2D2")
        return False
    ids_57 = {i: m for b, i, m in found if b == TARGET_BAUD}
    problems = []
    for b, i, m in found:
        if b != TARGET_BAUD:
            problems.append(f"ID {i} ({model_name(m)}) at {b} baud, not {TARGET_BAUD}")
    bus.set_baud(TARGET_BAUD)
    for i in ids_57:
        e = bus.comm_errors(i)
        if e:
            problems.append(f"ID {i}: {e}/10 reads failed -> probably TWO servos use ID {i}")
    for i in range(1, 7):
        if i not in ids_57:
            problems.append(f"ID {i} ({JOINTS[i]}) missing at {TARGET_BAUD} baud")
        elif (i <= 3 and kind(ids_57[i]) == "flex") or (i >= 4 and kind(ids_57[i]) == "rot"):
            problems.append(f"ID {i} adalah {model_name(ids_57[i])}, padahal {JOINTS[i]} "
                            f"({'XC430' if i <= 3 else 'XC330'}) -> IDs swapped?")
    for i in ids_57:
        if not 1 <= i <= 6:
            problems.append(f"ID {i} ({model_name(ids_57[i])}) outside 1-6")
    n_rot = sum(1 for _, _, m in found if kind(m) == "rot")
    n_flex = sum(1 for _, _, m in found if kind(m) == "flex")
    print(f"Found {len(found)} servos: {n_rot} rotation (430), {n_flex} flexion (330); expected 3 + 3.")
    if n_rot + n_flex < 6:
        print("   (servos with a DUPLICATE ID show up as one - fewer than 6 can mean that)")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        print("\n-> run the wizard:  ./run_gripper.sh setup_ids.py   (one servo plugged in per step)")
        return False
    print("\nOK: ID 1-6 complete, all at 57600, no duplicates, models match.")
    return True


def find_single(bus):
    """the ONE servo on the bus -> (baud, id, model) or None"""
    found = bus.scan_all(quiet=True)
    if not found:
        print("!! no servo detected: check the power + cable of this servo")
        return None
    if len(found) > 1:
        print("!! more than one servo detected: " + ", ".join(f"ID {i} @ {b}" for b, i, _ in found)
              + "\n   unplug the others, leave ONE servo on the U2D2")
        return None
    b, i, m = found[0]
    bus.set_baud(b)
    if bus.comm_errors(i, 5):
        print(f"!! ID {i} answers unreliably: maybe two servos are still on the bus")
        return None
    return found[0]


def ask(prompt, valid):
    while True:
        a = input(prompt).strip().lower()
        if a in valid:
            return a


def wizard(bus):
    print(__doc__.split("Why")[0])
    print("Steps: unplug ALL servos from the U2D2 (servo power may stay ON), then plug in ONE servo,")
    print("choose its finger, repeat for all six servos. At the end plug them all in -> checked.\n")
    done = {}
    while len(done) < 6:
        left = [j for j in range(1, 7) if j not in done]
        print(f"Not done yet: " + ", ".join(f"ID {j} ({JOINTS[j]})" for j in left))
        a = ask("Plug in ONE servo, then press ENTER   (s = finish / skip the rest): ", ("", "s"))
        if a == "s":
            break
        r = find_single(bus)
        if r is None:
            continue
        b, i, m = r
        k = kind(m)
        sug = [j for j in left if (k == "rot" and j <= 3) or (k == "flex" and j >= 4)] or left
        print(f"   found: ID {i} @ {b} baud, {model_name(m)} "
              f"({'ROTATION servo' if k == 'rot' else 'FLEXION servo' if k == 'flex' else 'unknown type'})")
        opts = "/".join(str(j) for j in sug)
        a = ask(f"   Which joint is this servo?  new ID [{opts}]  (ENTER = {sug[0]}, x = skip): ",
                tuple(str(j) for j in left) + ("", "x"))
        if a == "x":
            continue
        new = int(a) if a else sug[0]
        if (new <= 3 and k == "flex") or (new >= 4 and k == "rot"):
            if ask(f"   !! {model_name(m)} is usually {'FLEXION' if k == 'flex' else 'ROTATION'}, really make it ID {new}? [y/n]: ",
                   ("y", "n")) != "y":
                continue
        print(f"   write: ID {i} @ {b} -> ID {new} @ {TARGET_BAUD} ({JOINTS[new]})")
        if bus.assign(b, i, new):
            done[new] = m
            print(f"   >> label this servo: ID {new} = {JOINTS[new]}\n")
    print("\nNow plug ALL servos back into the U2D2.")
    input("ENTER for the final check ... ")
    return check(bus)


def remap(bus, spec):
    """rename several servos at once, all on the bus at TARGET_BAUD, also cyclic (6->4, 4->5, 5->6):
    every servo first goes to a temporary ID (200+), then to its final ID -> never two with the same ID"""
    try:
        pairs = [(int(a), int(b)) for a, b in (x.split(":") for x in spec.split(","))]
    except ValueError:
        print("!! format: remap OLD:NEW,OLD:NEW,...  e.g. remap 6:4,4:5,5:6")
        return False
    olds, news = [a for a, _ in pairs], [b for _, b in pairs]
    if len(set(olds)) != len(olds) or len(set(news)) != len(news):
        print("!! every old / new ID may appear only once")
        return False
    present = bus.scan(TARGET_BAUD)
    missing = [a for a in olds if a not in present]
    if missing:
        print(f"!! ID {missing} missing at {TARGET_BAUD} baud (run 'check' first)")
        return False
    taken = [b for b in news if b in present and b not in olds]
    if taken:
        print(f"!! new ID {taken} is already used by another servo that is not part of the swap")
        return False
    for a in olds:
        if bus.comm_errors(a, 5):
            print(f"!! ID {a} answers unreliably (two servos with ID {a}?) -> use the wizard, one at a time")
            return False
    print("Plan:")
    for a, b in pairs:
        print(f"   ID {a} ({model_name(present[a])}) -> ID {b}" + (f"  = {JOINTS[b]}" if b in JOINTS else ""))
        if b in JOINTS and ((b <= 3 and kind(present[a]) == "flex") or (b >= 4 and kind(present[a]) == "rot")):
            print(f"   !! {model_name(present[a])} as {JOINTS[b]}?")
    if ask("Write to the servos' EEPROM? [y/n]: ", ("y", "n")) != "y":
        return False
    bus.set_baud(TARGET_BAUD)
    tmp = {a: 200 + k for k, a in enumerate(olds)}
    for a in olds:                                        # 1) all to temporary IDs
        if not (bus.write1(a, ADDR_TORQUE, 0, "torque off") and bus.write1(a, ADDR_ID, tmp[a], f"ID {a} -> {tmp[a]}")):
            print("!! stopped; servos already moved are at ID 200+ -> run 'check', then repeat the remap")
            return False
        time.sleep(0.05)
    for a, b in pairs:                                    # 2) temporary -> final
        if not bus.write1(tmp[a], ADDR_ID, b, f"ID {tmp[a]} -> {b}"):
            print("!! stopped; run 'check'")
            return False
        time.sleep(0.05)
    ok = True
    now = bus.scan(TARGET_BAUD)
    for a, b in pairs:
        good = now.get(b) == present[a]
        ok = ok and good
        print(f"   verify ID {b}: {'OK ' + model_name(now[b]) if good else '!! mismatch / no answer'}")
    if ok:
        print("Done. Check: move ID 4 in the GUI -> the finger of rotation ID 1 must close.")
    return ok


def main():
    args = [a for a in sys.argv[1:]]
    bus = Bus()
    try:
        if args and args[0] == "check":
            ok = check(bus)
        elif args and args[0] == "remap" and len(args) > 1:
            ok = remap(bus, args[1])
        elif args and args[0] == "one" and len(args) > 1 and args[1].isdigit() and 1 <= int(args[1]) <= 6:
            r = find_single(bus)
            ok = bool(r) and bus.assign(r[0], r[1], int(args[1]))
        else:
            ok = wizard(bus)
    finally:
        bus.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
