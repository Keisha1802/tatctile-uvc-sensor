#!/usr/bin/env python3
"""
tactile_listener.py  —  example ROS 2 node that USES the tactile topics
=======================================================================
Simple grasp logic for a gripper with the tactile sensor(s) on the fingers:
  * no contact            -> close (command +1)
  * contact, load < target -> keep closing slowly
  * load >= target         -> hold (0)
  * slip                   -> squeeze harder (raise the target by 20 %)
  * release (object gone)  -> reset the target
Publishes the command on /gripper/command (std_msgs/Float32: +1 close, 0 hold, -1 open).
Replace that topic with your gripper driver's.

  source /opt/ros/<distro>/setup.bash
  python3 tactile_listener.py                       # listens to camera1 and camera2
  python3 tactile_listener.py camera1 --target=8    # one sensor, load target 8 (units of force.z)
"""
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from geometry_msgs.msg import WrenchStamped


class GraspDemo(Node):
    def __init__(self, cams, target):
        super().__init__("tactile_grasp_demo")
        self.target0 = self.target = target
        self.load = {c: 0.0 for c in cams}
        self.contact = {c: False for c in cams}
        self.cmd = self.create_publisher(Float32, "/gripper/command", 10)
        for c in cams:
            self.create_subscription(WrenchStamped, f"/tactile/{c}/wrench",
                                     lambda m, c=c: self.on_wrench(c, m), 10)
            self.create_subscription(String, f"/tactile/{c}/event",
                                     lambda m, c=c: self.on_event(c, m), 10)
        self.create_timer(0.05, self.step)                    # 20 Hz control loop
        self.get_logger().info(f"listening to {cams}, load target {target}")

    def on_wrench(self, cam, m):
        self.load[cam] = m.wrench.force.z
        self.contact[cam] = m.wrench.force.z > 0.0

    def on_event(self, cam, m):
        self.get_logger().info(f"{cam}: {m.data}")
        if m.data == "slip_start":
            self.target *= 1.2
            self.get_logger().warn(f"slip on {cam} -> squeeze harder, target {self.target:.2f}")
        elif m.data == "release":
            self.target = self.target0

    def step(self):
        load = max(self.load.values())
        if not any(self.contact.values()):
            u = 1.0                       # nothing touched yet: close
        elif load < self.target:
            u = 0.3                       # touching, not firm enough: close slowly
        else:
            u = 0.0                       # firm grip: hold
        self.cmd.publish(Float32(data=u))


def main():
    cams = [a for a in sys.argv[1:] if not a.startswith("--")] or ["camera1", "camera2"]
    target = 5.0
    for a in sys.argv[1:]:
        if a.startswith("--target="):
            target = float(a.split("=", 1)[1])
    rclpy.init()
    node = GraspDemo(cams, target)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
