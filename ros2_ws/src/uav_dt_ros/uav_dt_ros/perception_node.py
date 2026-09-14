"""Perception node: camera frames + vehicle pose -> bowl detections, geolocated tracks, annotated image.

Publishes
  ~/detections   vision_msgs/Detection2DArray   image-space boxes for the current frame
  ~/tracks       vision_msgs/Detection3DArray   confirmed tracks in the local ENU frame ("map");
                                                detection id is the track id, hypothesis score the best confidence
  ~/annotated    sensor_msgs/Image              quarter-resolution frame with boxes and track ids
  ~/ground_points std_msgs/String               JSON list of [x, y] ground points for the current frame
Subscribes /mission/verdicts (JSON {track_id: bool}) and records the verdict on each track for scoring.
Logs every frame to a JSONL file for scripts/score.py.

Pose source: 'mavros' uses /mavros/local_position/pose (what a real aircraft has),
'gz' uses the Gazebo ground-truth model odometry bridged on /uav/gz_odom (CI, and for isolating
perception error from estimator error).
"""
from __future__ import annotations

import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import (BoundingBox2D, Detection2D, Detection2DArray, Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)

from uav_dt.detector import GroundTruthDetector, TiledYoloDetector
from uav_dt.geolocate import CameraModel
from uav_dt.geotracker import GeoTracker
from uav_dt.pipeline import GeoPipeline


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception")
        p = self.declare_parameters("", [
            ("image_topic", "/uav/camera"),
            ("camera_info_topic", "/uav/camera_info"),
            ("pose_source", "gz"),                 # gz | mavros
            ("gz_odom_topic", "/uav/gz_odom"),
            ("mavros_pose_topic", "/mavros/local_position/pose"),
            ("detector", "gt"),                    # gt | yolo
            ("weights", "models/scratch_best.pt"),
            ("conf", 0.25),
            ("tile", 640),
            ("overlap", 0.2),
            ("device", ""),
            ("half", False),
            ("world_manifest", ""),                # JSON from tools/make_world.py, needed by the gt detector
            ("gt_miss_rate", 0.05),
            ("gt_pixel_noise", 1.0),
            ("ground_z", 0.0),
            ("gate_m", 1.5),
            ("log_path", "runs/perception.jsonl"),
            ("publish_annotated", True),
            ("hfov_deg", 70.0),
        ])
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.bridge = CvBridge()
        self.cam = None                                   # set from camera_info
        self.hfov = math.radians(float(g("hfov_deg")))
        self.pose = None                                  # (position, orientation_xyzw, stamp_s)
        self.pipeline = None
        self.frames = 0
        os.makedirs(os.path.dirname(g("log_path")) or ".", exist_ok=True)
        self.log = open(g("log_path"), "w")
        self.get_logger().info(f"logging frames to {g('log_path')}")

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, g("camera_info_topic"), self.on_camera_info, qos)
        self.create_subscription(Image, g("image_topic"), self.on_image, qos)
        if g("pose_source") == "gz":
            self.create_subscription(Odometry, g("gz_odom_topic"), self.on_odom, qos)
            self.get_logger().info(f"pose from Gazebo odometry {g('gz_odom_topic')}")
        else:
            self.create_subscription(PoseStamped, g("mavros_pose_topic"), self.on_pose, qos)
            self.get_logger().info(f"pose from {g('mavros_pose_topic')}")
        self.det_pub = self.create_publisher(Detection2DArray, "~/detections", 10)
        self.track_pub = self.create_publisher(Detection3DArray, "~/tracks", 10)
        self.img_pub = self.create_publisher(Image, "~/annotated", qos)
        self.gp_pub = self.create_publisher(String, "~/ground_points", 10)   # JSON [[x, y], ...] per frame, for the verify dwell
        self.create_subscription(String, "/mission/verdicts", self.on_verdicts, 10)

    # ---- inputs -------------------------------------------------------------------------------
    def on_camera_info(self, msg: CameraInfo):
        if self.cam is not None:
            return
        fx = msg.k[0]
        hfov = 2 * math.atan((msg.width / 2) / fx) if fx > 0 else self.hfov
        self.cam = CameraModel(msg.width, msg.height, hfov)
        self.get_logger().info(f"camera {msg.width}x{msg.height} hfov {math.degrees(hfov):.1f} deg")
        self._build_pipeline()

    def on_odom(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = ((p.x, p.y, p.z), (q.x, q.y, q.z, q.w), msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    def on_verdicts(self, msg: String):
        for tid, ok in json.loads(msg.data).items():
            tr = self.pipeline.tracker.get(int(tid)) if self.pipeline else None
            if tr is not None:
                tr.verified = bool(ok)

    def on_pose(self, msg: PoseStamped):
        p, q = msg.pose.position, msg.pose.orientation
        self.pose = ((p.x, p.y, p.z), (q.x, q.y, q.z, q.w), msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)

    def _build_pipeline(self):
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        if g("detector") == "yolo":
            det = TiledYoloDetector(g("weights"), conf=float(g("conf")), tile=int(g("tile")), overlap=float(g("overlap")),
                                    device=(g("device") or None), half=bool(g("half")))
            self.get_logger().info(f"tiled YOLO detector from {g('weights')}")
        else:
            manifest = json.load(open(g("world_manifest")))
            bowls = [(b["x"], b["y"]) for b in manifest["bowls"]]
            det = GroundTruthDetector(self.cam, bowls, ground_z=float(g("ground_z")), miss_rate=float(g("gt_miss_rate")),
                                      pixel_noise=float(g("gt_pixel_noise")))
            self.get_logger().info(f"ground-truth detector with {len(bowls)} bowls from {g('world_manifest')}")
        self.pipeline = GeoPipeline(det, self.cam, GeoTracker(gate_m=float(g("gate_m"))), ground_z=float(g("ground_z")))

    # ---- main path ----------------------------------------------------------------------------
    def on_image(self, msg: Image):
        if self.pipeline is None or self.pose is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        position, orientation, _ = self.pose
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        res = self.pipeline.process(frame, t, position, orientation)
        self.frames += 1
        self.log.write(json.dumps(res.to_record()) + "\n")
        self.log.flush()
        self.publish(msg.header, frame, res)
        if self.frames % 5 == 1:
            self.get_logger().info(f"frame {self.frames}: {len(res.detections)} dets, {len(res.tracks)} confirmed tracks, "
                                   f"inference {res.inference_s:.2f}s at {position[2]:.1f} m")

    def publish(self, header, frame, res):
        d2 = Detection2DArray(header=header)
        for d in res.detections:
            det = Detection2D(header=header)
            det.bbox = BoundingBox2D()
            det.bbox.center.position.x = float((d[0] + d[2]) / 2)
            det.bbox.center.position.y = float((d[1] + d[3]) / 2)
            det.bbox.size_x, det.bbox.size_y = float(d[2] - d[0]), float(d[3] - d[1])
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = "bowl", float(d[4])
            det.results.append(hyp)
            d2.detections.append(det)
        self.det_pub.publish(d2)
        self.gp_pub.publish(String(data=json.dumps([[round(g[0], 3), round(g[1], 3)] for g in res.ground_points if g is not None])))

        d3 = Detection3DArray(header=header)
        d3.header.frame_id = "map"
        for tr in res.tracks:
            det = Detection3D(header=d3.header)
            det.id = str(tr.id)
            det.bbox.center.position.x, det.bbox.center.position.y = float(tr.x), float(tr.y)
            det.bbox.center.position.z = float(self.pipeline.ground_z)
            det.bbox.size.x = det.bbox.size.y = 0.16
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = "bowl", float(tr.score)
            hyp.pose.pose.position.x, hyp.pose.pose.position.y = float(tr.x), float(tr.y)
            det.results.append(hyp)
            d3.detections.append(det)
        self.track_pub.publish(d3)

        if self.get_parameter("publish_annotated").value:
            scale = 0.25
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            for d, g in zip(res.detections, res.ground_points):
                x1, y1, x2, y2 = (int(v * scale) for v in d[:4])
                cv2.rectangle(small, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 255), 1)
                label = f"{d[4]:.2f}"
                if g is not None:
                    tr = min(res.tracks, key=lambda tr: math.hypot(tr.x - g[0], tr.y - g[1]), default=None)
                    if tr is not None and math.hypot(tr.x - g[0], tr.y - g[1]) < self.pipeline.tracker.gate_m:
                        label = f"#{tr.id} {d[4]:.2f}"
                cv2.putText(small, label, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(small, f"{res.position[2]:.1f} m AGL  {len(res.tracks)} tracks", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            out = self.bridge.cv2_to_imgmsg(small, encoding="bgr8")
            out.header = header
            self.img_pub.publish(out)


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
