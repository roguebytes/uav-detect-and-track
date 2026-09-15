#!/usr/bin/env python3
"""Log the UAV's ground-truth odometry to CSV at full rate, for the cinematic replay pass.

    python3 scripts/record_trajectory.py --out runs/x/trajectory.csv [--topic /uav/gz_odom]

Columns: t x y z qx qy qz qw (sim seconds, world ENU metres, body orientation).
"""
import argparse
import os

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class TrajectoryLogger(Node):
    def __init__(self, topic, out):
        super().__init__("record_trajectory")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        self.f = open(out, "w")
        self.f.write("t x y z qx qy qz qw\n")
        self.n = 0
        self.create_subscription(Odometry, topic, self.cb, QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.get_logger().info(f"logging {topic} to {out}")

    def cb(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.f.write(f"{t:.3f} {p.x:.4f} {p.y:.4f} {p.z:.4f} {q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n")
        self.n += 1
        if self.n % 500 == 0:
            self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/uav/gz_odom")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rclpy.init()
    node = TrajectoryLogger(a.topic, a.out)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.f.close()
        node.get_logger().info(f"{node.n} poses written")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
