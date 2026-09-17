#!/usr/bin/env python3
"""Save N frames from a ROS 2 image topic as PNG files, then exit.

    python3 scripts/grab_frames.py --topic /uav/camera --count 2 --out-dir /tmp/frames

Needs the system ROS 2 Python (rclpy, cv_bridge), not the venv.
"""

__author__ = "Frank Loewenich"
import argparse
import os
import sys

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class Grabber(Node):
    """Save a fixed number of frames from an image topic, then exit."""
    def __init__(self, topic, count, out_dir):
        """Subscribe to the image topic."""
        super().__init__("grab_frames")
        self.bridge, self.count, self.out_dir, self.n = CvBridge(), count, out_dir, 0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, topic, self.cb, qos)

    def cb(self, msg):
        """Write the frame to disk and stop once enough frames have been saved."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        path = os.path.join(self.out_dir, f"frame_{self.n:04d}.png")
        cv2.imwrite(path, frame)
        print(f"saved {path} {frame.shape[1]}x{frame.shape[0]} stamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}", flush=True)
        self.n += 1
        if self.n >= self.count:
            raise SystemExit(0)


def main():
    """Parse the command line and run the grabber."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/uav/camera")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rclpy.init()
    node = Grabber(a.topic, a.count, a.out_dir)
    try:
        rclpy.spin_until_future_complete(node, rclpy.task.Future(), timeout_sec=a.timeout)
    except (SystemExit, KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if node.n >= a.count else 1)


if __name__ == "__main__":
    main()
