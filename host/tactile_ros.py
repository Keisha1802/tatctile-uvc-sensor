#!/usr/bin/env python3
"""
tactile_ros.py  —  turn flow + depth into robot-usable tactile signals, publish on ROS 2
==========================================================================================
Used by run_sensor_fast.py / run_sensor_dual.py with the argument  ros :
  source /opt/ros/<distro>/setup.bash
  python3 run_sensor_fast.py usb ros
  python3 run_sensor_fast.py dual --usb=camera1 --usb2=camera2 ros
Options:
  --gel-mm=W        width of the gel area seen in the ROI, in mm  -> positions/shear in mm
  --force-gain=G    newtons per unit of indentation volume (from a force-gauge fit) -> force in N
  --slip-thr=S      slip threshold: speed of the mean gel shear in the contact, px/s (default 6)
  --ros-img-hz=H    rate of the depth image topic (default 5, 0 = off)
  --ros-ns=NAME     topic prefix (default /tactile)
  --udp[=HOST:PORT] ALSO (or instead of ROS) send the features as one JSON line per frame over UDP
                    (default 127.0.0.1:9870). No ROS needed; servo_grasp.py listens to this.
                    {"cam": "camera1", "t": <unix time>, "events": [...], "contact": 1, "depth_max_mm": .., ...}

Topics per camera  (<ns>/<camera>/..., camera = camera1, camera2, csi):
  contact        std_msgs/Bool                 touching or not (hysteresis on contact area)
  wrench         geometry_msgs/WrenchStamped   force.z  = normal load (N if --force-gain,
                                                          else indentation volume, proxy)
                                               force.x/y = shear (mean gel displacement in
                                                          the contact, mm or px)
                                               torque.z = twist of the contact (rad)
  contact_point  geometry_msgs/PointStamped    x, y = contact centre (mm from ROI centre if
                                               --gel-mm, else normalised -1..1), z = max depth mm
  slip           std_msgs/Bool                 slipping (shear changing fast while in contact)
  event          std_msgs/String               touch | release | slip_start | slip_end
  state          std_msgs/Float32MultiArray    all numbers in one message (labels in layout):
                                               contact, area_frac, cx, cy, depth_max_mm,
                                               volume, force, shear_x, shear_y, twist,
                                               slip_score, slip, E, dE, fps
  depth          sensor_msgs/Image (32FC1, mm) depth map, --ros-img-hz
Frame: x = right, y = down in the image of the gel (frame_id tactile_<camera>).

Without ROS 2 installed the features are still computed and shown in the window;
only publishing is skipped (a message says how to enable it).
"""
import math
import os
import time

import numpy as np

LABELS = ["contact", "area_frac", "cx", "cy", "depth_max_mm", "volume", "force",
          "shear_x", "shear_y", "twist", "slip_score", "slip", "E", "dE", "fps"]


class TactileFeatures:
    """Per-camera feature extractor: contact, load, shear, twist, slip, events."""

    def __init__(self, gel_mm=None, force_gain=None, slip_thr=None,
                 on_frac=0.006, off_frac=0.003, contact_mm=0.02, flow_contact_px=1.0):
        self.gel_mm = gel_mm
        self.force_gain = force_gain
        self.slip_thr = slip_thr if slip_thr is not None else 6.0
        self.on_frac, self.off_frac = on_frac, off_frac
        self.contact_mm = contact_mm
        self.flow_contact_px = flow_contact_px
        self.step = 2                       # feature grid: every 2nd pixel
        self.reset()

    def reset(self):
        self.contact = False
        self.slip = False
        self._prev_shear = None
        self._prev_t = None
        self._score = 0.0
        self._prev_load = None      # normal load of the previous frame (slip gating)
        self._slip_n = 0
        self.last = None

    def update(self, flow, depth_mm, E=float("nan"), dE=0.0, fps=0.0, t=None):
        """flow: HxWx2 px (vs baseline). depth_mm: HxW mm (0 outside contact) or None.
        Returns (dict of features, list of events)."""
        t = time.monotonic() if t is None else t
        h, w = flow.shape[:2]
        # Every 2nd pixel: flow and depth are smooth (DIS + density blur), so contact, load,
        # shear and twist are the same to well below a pixel, at ~1/4 of the CPU time.
        st = self.step
        fl = flow[::st, ::st]
        mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
        if depth_mm is not None:
            d = depth_mm
            if d.shape[:2] != (h, w):
                import cv2
                d = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            d = d[::st, ::st]
            mask = d > self.contact_mm
        else:
            d = None
            mask = mag > self.flow_contact_px            # no depth: contact from flow
        area = int(mask.sum()) * st * st                  # in full-resolution pixels
        frac = float(mask.mean())

        events = []
        was = self.contact
        self.contact = frac > (self.off_frac if was else self.on_frac)
        if self.contact and not was:
            events.append("touch")
        elif was and not self.contact:
            events.append("release")
            if self.slip:
                events.append("slip_end")
            self.slip, self._slip_n, self._score = False, 0, 0.0
            self._prev_shear = None
            self._prev_load = None

        px_mm = (self.gel_mm / float(w)) if self.gel_mm else None
        cx = cy = 0.0
        dmax = volume = force = sx = sy = twist = 0.0
        if self.contact and area > 0:
            yi, xi = np.nonzero(mask)                         # sub-grid indices
            ys, xs = yi * st, xi * st                         # full-resolution pixels
            wts = d[yi, xi] if d is not None else mag[yi, xi]
            wsum = float(wts.sum()) or 1.0
            mx = float((xs * wts).sum() / wsum); my = float((ys * wts).sum() / wsum)
            fx = fl[yi, xi, 0]; fy = fl[yi, xi, 1]
            sx, sy = float(fx.mean()), float(fy.mean())             # shear (px)
            spx = (sx, sy)                                           # slip is judged in px/s
            rx, ry = xs - mx, ys - my                                # twist around the centre
            r2 = float((rx * rx + ry * ry).sum())
            twist = float(((rx * (fy - sy) - ry * (fx - sx)).sum()) / r2) if r2 > 0 else 0.0
            if d is not None:
                dmax = float(depth_mm.max()) if depth_mm is not None else float(d[yi, xi].max())
                volume = float(d[yi, xi].sum()) * st * st             # mm * px^2
                if px_mm:
                    volume *= px_mm * px_mm                          # mm^3
            if px_mm:
                cx, cy = (mx - w / 2.0) * px_mm, (my - h / 2.0) * px_mm
                sx, sy = sx * px_mm, sy * px_mm
            else:
                cx, cy = (mx - w / 2.0) / (w / 2.0), (my - h / 2.0) / (h / 2.0)
            force = volume * self.force_gain if self.force_gain else volume

            # slip: shear changing fast while the contact holds
            if self._prev_shear is not None and self._prev_t is not None:
                dt = max(t - self._prev_t, 1e-3)
                v = math.hypot(spx[0] - self._prev_shear[0], spx[1] - self._prev_shear[1]) / dt
                # pressing / releasing changes the mean shear too -> NOT slip. Only count
                # shear motion while the normal load is steady (< 8 % change per frame).
                load = volume if volume > 0 else float(area)
                if self._prev_load is not None and self._prev_load > 0:
                    dl = abs(load - self._prev_load) / self._prev_load
                    if dl > 0.08:
                        v = 0.0
                if frac < 3.0 * self.on_frac:     # tiny contact (touch starting/ending):
                    v = 0.0                       # mean shear of few pixels is mostly noise
                self._prev_load = load
                self._score = 0.6 * self._score + 0.4 * v
            self._prev_shear = spx
            if self._score > self.slip_thr:
                self._slip_n += 1
            else:
                self._slip_n = 0
            new_slip = self._slip_n >= 3 or (self.slip and self._score > 0.5 * self.slip_thr)
            if new_slip and not self.slip:
                events.append("slip_start")
            elif self.slip and not new_slip:
                events.append("slip_end")
            self.slip = new_slip
        self._prev_t = t

        f = dict(contact=float(self.contact), area_frac=frac, cx=cx, cy=cy, depth_max_mm=dmax,
                 volume=volume, force=force, shear_x=sx, shear_y=sy, twist=twist,
                 slip_score=self._score, slip=float(self.slip),
                 E=float(E) if E == E else float("nan"), dE=float(dE), fps=float(fps))
        self.last = f
        return f, events

    def summary(self):
        """Short text for the GUI."""
        f = self.last
        if f is None:
            return "", False
        if not f["contact"]:
            return "no contact", False
        u = "mm" if self.gel_mm else "px"
        fu = "N" if self.force_gain else ("mm3" if self.gel_mm else "vol")
        txt = (f"CONTACT F {f['force']:.2f}{fu}  shear {f['shear_x']:+.2f},{f['shear_y']:+.2f}{u}"
               f"  twist {math.degrees(f['twist']):+.1f}deg" + ("  SLIP" if f["slip"] else ""))
        return txt, bool(f["slip"])


class TactileROS:
    """One ROS 2 node, one set of topics per camera. Publishing is thread-safe."""

    def __init__(self, cameras, ns="/tactile", img_hz=5.0, node_name="tactile_sensor", udp=None, ros=True):
        self.ok = False
        self.img_dt = (1.0 / img_hz) if img_hz and img_hz > 0 else None
        self.udp = None
        if udp:
            import socket
            host, _, port = udp.rpartition(":")
            self.udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host or "127.0.0.1", int(port)))
            print(f"[UDP] sending tactile features of {cameras} to {self.udp[1][0]}:{self.udp[1][1]}")
        if not ros:
            return
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from std_msgs.msg import Bool, String, Float32MultiArray, MultiArrayDimension
            from geometry_msgs.msg import WrenchStamped, PointStamped
            from sensor_msgs.msg import Image
        except Exception as e:
            print(f"[ROS] rclpy not available ({e}). Features are shown, NOT published. "
                  f"Enable: source /opt/ros/<distro>/setup.bash, then run again with 'ros'.")
            return
        self.M = dict(Bool=Bool, String=String, Arr=Float32MultiArray, Dim=MultiArrayDimension,
                      Wrench=WrenchStamped, Point=PointStamped, Image=Image)
        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(node_name)
        ns = "/" + ns.strip("/")
        self.pubs, self._t_img = {}, {}
        for cam in cameras:
            b = f"{ns}/{cam}"
            self.pubs[cam] = dict(
                contact=self.node.create_publisher(Bool, f"{b}/contact", 10),
                slip=self.node.create_publisher(Bool, f"{b}/slip", 10),
                event=self.node.create_publisher(String, f"{b}/event", 10),
                state=self.node.create_publisher(Float32MultiArray, f"{b}/state", 10),
                wrench=self.node.create_publisher(WrenchStamped, f"{b}/wrench", 10),
                point=self.node.create_publisher(PointStamped, f"{b}/contact_point", 10),
                depth=self.node.create_publisher(Image, f"{b}/depth", qos_profile_sensor_data))
            self._t_img[cam] = 0.0
            print(f"[ROS] publishing {b}/{{contact,wrench,contact_point,slip,event,state,depth}}")
        self.ok = True

    def _header(self, msg, cam):
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = f"tactile_{cam}"
        return msg

    def _send_udp(self, cam, f, events):
        import json
        msg = {"cam": cam, "t": time.time(), "events": list(events), "src": os.getpid()}
        for k in LABELS:
            v = float(f[k])
            msg[k] = v if v == v else 0.0
        try:
            self.udp[0].sendto(json.dumps(msg).encode(), self.udp[1])
        except OSError:
            pass                                   # nobody listening / network down: never block the sensor

    def publish(self, cam, f, events, depth_mm=None):
        if self.udp is not None:
            self._send_udp(cam, f, events)
        if not self.ok or cam not in self.pubs:
            return
        M, P = self.M, self.pubs[cam]
        P["contact"].publish(M["Bool"](data=bool(f["contact"])))
        P["slip"].publish(M["Bool"](data=bool(f["slip"])))
        for ev in events:
            P["event"].publish(M["String"](data=ev))
        a = M["Arr"]()
        a.layout.dim = [M["Dim"](label=",".join(LABELS), size=len(LABELS), stride=len(LABELS))]
        a.data = [float(f[k]) if f[k] == f[k] else 0.0 for k in LABELS]
        P["state"].publish(a)
        wr = self._header(M["Wrench"](), cam)
        wr.wrench.force.x, wr.wrench.force.y, wr.wrench.force.z = f["shear_x"], f["shear_y"], f["force"]
        wr.wrench.torque.z = f["twist"]
        P["wrench"].publish(wr)
        pt = self._header(M["Point"](), cam)
        pt.point.x, pt.point.y, pt.point.z = f["cx"], f["cy"], f["depth_max_mm"]
        P["point"].publish(pt)
        if depth_mm is not None and self.img_dt is not None:
            now = time.monotonic()
            if now - self._t_img[cam] >= self.img_dt:
                self._t_img[cam] = now
                d = np.ascontiguousarray(depth_mm, dtype=np.float32)
                im = self._header(M["Image"](), cam)
                im.height, im.width = d.shape[:2]
                im.encoding, im.is_bigendian, im.step = "32FC1", 0, d.shape[1] * 4
                im.data = d.tobytes()
                P["depth"].publish(im)

    def close(self):
        if self.ok:
            try:
                self.node.destroy_node()
                if self.rclpy.ok():
                    self.rclpy.shutdown()
            except Exception:
                pass
            self.ok = False


def parse_args(argv):
    """Common options for both scripts -> dict."""
    o = dict(on=False, ros=False, udp=None, gel_mm=None, force_gain=None, slip_thr=None, img_hz=5.0, ns="/tactile")
    for a in argv:
        al = a.lower()
        if al in ("ros", "--ros"):
            o["on"] = o["ros"] = True
        elif al == "--udp" or al.startswith("--udp="):
            o["on"] = True
            o["udp"] = a.split("=", 1)[1] if "=" in a else "127.0.0.1:9870"
            if ":" not in o["udp"]:
                o["udp"] = "127.0.0.1:" + o["udp"]
        elif al.startswith("--gel-mm="):
            o["gel_mm"] = float(al.split("=", 1)[1])
        elif al.startswith("--force-gain="):
            o["force_gain"] = float(al.split("=", 1)[1])
        elif al.startswith("--slip-thr="):
            o["slip_thr"] = float(al.split("=", 1)[1])
        elif al.startswith("--ros-img-hz="):
            o["img_hz"] = float(al.split("=", 1)[1])
        elif al.startswith("--ros-ns="):
            o["ns"] = a.split("=", 1)[1]
    return o


def make_features(o):
    return TactileFeatures(gel_mm=o["gel_mm"], force_gain=o["force_gain"], slip_thr=o["slip_thr"])
