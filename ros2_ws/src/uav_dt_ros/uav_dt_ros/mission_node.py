"""Mission node: survey the field at 40 m, then verify each confirmed track at 11 m, then land.

Consumes ~/tracks and ~/detections-derived ground points from the perception node, drives the
autopilot through the FlightController interface (MAVROS + PX4 offboard today), and publishes
verdicts on /mission/verdicts (std_msgs/String, JSON {track_id: bool}) which the perception
node folds into its log for scoring. Mission state goes to /mission/state.
"""
from __future__ import annotations

import json
import math
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from uav_dt.geolocate import CameraModel
from uav_dt.mission.controller import MavrosPx4Controller
from uav_dt.mission.state_machine import State, SurveyVerifyMission
from uav_dt.mission.survey import lawnmower, path_length
from uav_dt.mission.verify import QuadDescendVerify


class MissionNode(Node):
    def __init__(self):
        super().__init__("mission")
        self.declare_parameters("", [
            ("field_w", 120.0), ("field_h", 80.0), ("survey_alt", 40.0), ("verify_alt", 11.0),
            ("side_overlap", 0.1), ("hfov_deg", 70.0), ("image_w", 4032), ("image_h", 3024),
            ("dwell_s", 4.0), ("verify_radius", 1.0), ("verify_ratio", 0.5), ("max_verify", 0),
            ("wp_tol", 1.5), ("tick_hz", 5.0), ("log_path", "runs/mission.jsonl"),
            ("tracks_topic", "/perception/tracks"), ("ground_points_topic", "/perception/ground_points"),
        ])
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        cam = CameraModel(int(g("image_w")), int(g("image_h")), math.radians(float(g("hfov_deg"))))
        wps = lawnmower(float(g("field_w")), float(g("field_h")), cam, float(g("survey_alt")), float(g("side_overlap")))
        self.get_logger().info(f"survey: {len(wps)} waypoints, {path_length(wps):.0f} m at {g('survey_alt')} m")
        self.ctl = MavrosPx4Controller(self)
        self.mission = SurveyVerifyMission(self.ctl, wps, float(g("survey_alt")), float(g("verify_alt")),
                                           QuadDescendVerify(float(g("dwell_s"))), wp_tol=float(g("wp_tol")),
                                           verify_radius=float(g("verify_radius")), verify_ratio=float(g("verify_ratio")),
                                           max_verify=(int(g("max_verify")) or None))
        self.tracks, self.observations = [], None
        self.create_subscription(Detection3DArray, g("tracks_topic"), self.on_tracks, 10)
        self.create_subscription(String, g("ground_points_topic"), self.on_ground_points, 10)
        self.state_pub = self.create_publisher(String, "/mission/state", 10)
        self.verdict_pub = self.create_publisher(String, "/mission/verdicts", 10)
        os.makedirs(os.path.dirname(g("log_path")) or ".", exist_ok=True)
        self.log = open(g("log_path"), "w")
        self._last_state, self._last_verdicts = None, {}
        self.create_timer(1.0 / float(g("tick_hz")), self.tick)

    def on_tracks(self, msg: Detection3DArray):
        self.tracks = [(int(d.id), d.bbox.center.position.x, d.bbox.center.position.y) for d in msg.detections]

    def on_ground_points(self, msg: String):
        self.observations = [tuple(p) for p in json.loads(msg.data)]

    def tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.mission.tick(now, self.tracks, self.observations)
        st = self.mission.state.value
        if st != self._last_state:
            self._last_state = st
            self.state_pub.publish(String(data=st))
            pos = self.ctl.position()
            self.get_logger().info(f"state -> {st} at {pos and tuple(round(v, 1) for v in pos)}, mode {self.ctl.mode()}")
            self.log.write(json.dumps({"t": now, "state": st, "position": pos, "verdicts": self.mission.verdicts}) + "\n")
            self.log.flush()
        if self.mission.verdicts != self._last_verdicts:
            self._last_verdicts = dict(self.mission.verdicts)
            self.verdict_pub.publish(String(data=json.dumps(self._last_verdicts)))
            stats = getattr(self.mission, "dwell_stats", {})
            self.get_logger().info(f"verdicts {self._last_verdicts} (frames, hits per track: {stats})")
        if self.mission.done:
            self.get_logger().info("mission done")
            self.log.close()
            raise SystemExit(0)


def main():
    rclpy.init()
    node = MissionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
