#!/usr/bin/env python3
"""Plain flight through the FlightController interface: take off, fly a square, land.

    ros2 launch uav_dt_ros sim.launch.py world:=bowl_field_sparse model:=x500_nadir_cam_lite headless:=true
    python3 scripts/plain_flight.py --alt 10 --side 40

Exercises MAVROS offboard on PX4 exactly as the mission node does. Exits 0 on success.
"""

__author__ = "Frank Loewenich"
import argparse
import math
import sys
import time

import rclpy
from rclpy.node import Node

from uav_dt.mission.controller import MavrosPx4Controller


def main():
    """Take off, fly a square and land through the flight controller, then exit with the result."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", type=float, default=10.0)
    ap.add_argument("--side", type=float, default=40.0)
    ap.add_argument("--timeout", type=float, default=240.0)
    a = ap.parse_args()
    rclpy.init()
    node = Node("plain_flight")
    ctl = MavrosPx4Controller(node)
    t0 = time.time()
    legs = [(0, 0), (a.side, 0), (a.side, a.side), (0, a.side), (0, 0)]
    stage, leg, ok = "connect", 0, False
    last_log = 0.0
    try:
        while rclpy.ok() and time.time() - t0 < a.timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.time()
            if now - last_log > 3:
                last_log = now
                p = ctl.position()
                node.get_logger().info(f"{stage} leg {leg} mode {ctl.mode()} armed {ctl.armed()} pos "
                                       f"{p and tuple(round(v, 1) for v in p)}")
            if stage == "connect" and ctl.connected():
                ctl.takeoff(a.alt)
                stage = "takeoff"
            elif stage == "takeoff" and ctl.armed() and ctl.reached(1.0):
                stage = "square"
                ctl.goto(*legs[leg], a.alt, yaw=0.0)
            elif stage == "square" and ctl.reached(1.5):
                leg += 1
                if leg < len(legs):
                    x, y = legs[leg]
                    ctl.goto(x, y, a.alt, yaw=math.atan2(y - legs[leg - 1][1], x - legs[leg - 1][0]))
                else:
                    ctl.land()
                    stage = "land"
            elif stage == "land" and not ctl.armed():
                ok = True
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print("PLAIN FLIGHT", "PASSED" if ok else f"FAILED in stage {stage}", f"after {time.time() - t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
