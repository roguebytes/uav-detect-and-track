"""Keep the follow_cam model behind and above the UAV with a level attitude, for video.

Reads the UAV's ground-truth odometry (/uav/gz_odom), computes a camera pose `distance` metres
behind the nose (using the UAV's yaw) and `height` metres above it, pitched down by `pitch_deg`
with zero roll, and sets it through the world's set_pose service at `rate_hz`. The service is
called through tools/gz_set_pose (a small persistent gz-transport client fed over a pipe): the
ros_gz service bridge on this Humble build never answers, and the gz CLI costs 0.3 s per call.
Because the camera does not roll or pitch with the airframe, altitude changes and the descent to
the verify altitude read clearly in the footage.
"""
from __future__ import annotations

import math
import os
import subprocess

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy


class FollowCamNode(Node):
    def __init__(self):
        super().__init__("follow_cam")
        self.declare_parameters("", [("world", "bowl_field_sparse"), ("entity", "follow_cam"), ("odom_topic", "/uav/gz_odom"),
                                     ("distance", 10.0), ("height", 4.0), ("pitch_deg", 22.0), ("rate_hz", 10.0),
                                     ("yaw_smoothing", 0.15)])
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.entity, self.d, self.h = g("entity"), float(g("distance")), float(g("height"))
        self.pitch, self.alpha = math.radians(float(g("pitch_deg"))), float(g("yaw_smoothing"))
        exe = os.environ.get("GZ_SET_POSE") or os.path.join(os.environ.get("UAV_DT_REPO", "."), "build", "gz_set_pose", "gz_set_pose")
        if not os.path.exists(exe):
            raise SystemExit(f"{exe} not found; build it with scripts/build_tools.sh")
        self.client = subprocess.Popen([exe, g("world")], stdin=subprocess.PIPE, text=True, bufsize=1)
        self.odom, self.yaw_f = None, None
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, g("odom_topic"), self.on_odom, qos)
        self.create_timer(1.0 / float(g("rate_hz")), self.tick)
        self.get_logger().info(f"following UAV with '{self.entity}' at {self.d} m back, {self.h} m up")

    def on_odom(self, msg):
        self.odom = msg

    def tick(self):
        if self.odom is None or self.client.poll() is not None:
            return
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        if self.yaw_f is None:
            self.yaw_f = yaw
        else:   # low-pass the yaw so the camera swings smoothly through turns
            d = math.atan2(math.sin(yaw - self.yaw_f), math.cos(yaw - self.yaw_f))
            self.yaw_f += self.alpha * d
        cx, cy, cz = p.x - self.d * math.cos(self.yaw_f), p.y - self.d * math.sin(self.yaw_f), p.z + self.h
        # orientation: yaw toward the UAV, pitch down, no roll (ZYX)
        cy2, sy2 = math.cos(self.yaw_f / 2), math.sin(self.yaw_f / 2)
        cp2, sp2 = math.cos(self.pitch / 2), math.sin(self.pitch / 2)
        qw, qx, qy, qz = cy2 * cp2, -sy2 * sp2, cy2 * sp2, sy2 * cp2
        try:
            self.client.stdin.write(f"{self.entity} {cx:.3f} {cy:.3f} {cz:.3f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")
        except BrokenPipeError:
            self.get_logger().error("gz_set_pose client exited")

    def destroy_node(self):
        try:
            self.client.stdin.close()
            self.client.wait(timeout=2)
        except Exception:  # noqa: BLE001
            self.client.kill()
        super().destroy_node()


def main():
    rclpy.init()
    node = FollowCamNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
