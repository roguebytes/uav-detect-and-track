"""ROS 2 node: subscribe to an image topic, run detect-and-track, publish the
normalised target offset (geometry_msgs/PointStamped; z carries the track id) and
an annotated image. Run inside a ROS 2 + ultralytics environment:

    python ros2/detect_track_node.py --ros-args \
        -p image_topic:=/camera/image_raw -p model:=yolo11n.pt
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image

from uav_dt.detector import YoloDetector
from uav_dt.pipeline import Pipeline


class DetectTrackNode(Node):
    def __init__(self):
        super().__init__("detect_track")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("model", "yolo11n.pt")
        self.declare_parameter("conf", 0.25)

        topic = self.get_parameter("image_topic").value
        self.bridge = CvBridge()
        self.pipeline = Pipeline(YoloDetector(
            self.get_parameter("model").value,
            float(self.get_parameter("conf").value)))
        self.idx = 0

        self.sub = self.create_subscription(Image, topic, self.on_image, 10)
        self.offset_pub = self.create_publisher(PointStamped, "~/target_offset", 10)
        self.image_pub = self.create_publisher(Image, "~/annotated", 10)
        self.get_logger().info(f"detect_track subscribed to {topic}")

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self.pipeline.process(frame, self.idx)
        self.idx += 1

        tg = result["target"]
        if tg is not None:
            pt = PointStamped()
            pt.header = msg.header
            pt.point.x, pt.point.y = tg["offset"]
            pt.point.z = float(tg["id"])
            self.offset_pub.publish(pt)

        from detect_track import draw  # reuse the CLI annotator
        annotated = self.bridge.cv2_to_imgmsg(draw(frame, result), encoding="bgr8")
        annotated.header = msg.header
        self.image_pub.publish(annotated)


def main():
    rclpy.init()
    node = DetectTrackNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
