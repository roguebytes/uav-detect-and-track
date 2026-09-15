"""The control action behind an interface, so the autopilot and airframe can be swapped.

FlightController is what the mission state machine talks to. Positions are local ENU metres
relative to the takeoff point, altitudes are metres above the takeoff point (AGL over flat
ground). Calls are non-blocking: the caller polls `reached()` from its own timer.

MavrosPx4Controller  PX4 offboard mode through MAVROS (simulation now, and any PX4 aircraft).
                     A later MavrosArduPilotController will use GUIDED mode with the same calls.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class FlightController(ABC):
    @abstractmethod
    def connected(self) -> bool: ...

    @abstractmethod
    def position(self) -> tuple[float, float, float] | None:
        """Current local ENU position, or None before the estimator is up."""

    @abstractmethod
    def armed(self) -> bool: ...

    @abstractmethod
    def takeoff(self, altitude: float) -> None:
        """Arm and climb to `altitude` over the current position."""

    @abstractmethod
    def goto(self, x: float, y: float, z: float, yaw: float | None = None) -> None:
        """Fly to local ENU (x, y, z). Yaw in radians (ENU, 0 = east), None keeps the current yaw."""

    @abstractmethod
    def land(self) -> None: ...

    def reached(self, tol: float = 1.0) -> bool:
        p, t = self.position(), self.target()
        return p is not None and t is not None and math.dist(p, t) < tol

    @abstractmethod
    def target(self) -> tuple[float, float, float] | None: ...


class MavrosPx4Controller(FlightController):
    """PX4 offboard control through MAVROS 2 on ROS 2 Humble.

    Streams the current position setpoint at 20 Hz (PX4 needs a stream before and during
    OFFBOARD). Composition over inheritance: pass in the rclpy node that owns the timers.
    """

    def __init__(self, node, setpoint_hz: float = 20.0, ns: str = "/mavros"):
        from geometry_msgs.msg import PoseStamped
        from mavros_msgs.msg import State
        from mavros_msgs.srv import CommandBool, CommandTOL, ParamSetV2, SetMode
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        self.node, self._PoseStamped = node, PoseStamped
        self._state, self._pose, self._target, self._yaw = None, None, None, 0.0
        self._pending_offboard, self._pending_arm = False, False
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        node.create_subscription(State, f"{ns}/state", self._on_state, 10)
        node.create_subscription(PoseStamped, f"{ns}/local_position/pose", self._on_pose, qos)
        self._sp_pub = node.create_publisher(PoseStamped, f"{ns}/setpoint_position/local", 10)
        self._arm_cli = node.create_client(CommandBool, f"{ns}/cmd/arming")
        self._mode_cli = node.create_client(SetMode, f"{ns}/set_mode")
        self._land_cli = node.create_client(CommandTOL, f"{ns}/cmd/land")
        self._param_cli = node.create_client(ParamSetV2, f"{ns}/param/set")
        # survey-friendly PX4 limits: 5 m/s, gentle acceleration, 20 deg max tilt (a DJI-class survey profile)
        self.px4_params = {"MPC_XY_VEL_MAX": 5.0, "MPC_ACC_HOR": 2.0, "MPC_TILTMAX_AIR": 20.0, "MPC_Z_VEL_MAX_DN": 2.0}
        self._params_pending = dict(self.px4_params)     # not yet confirmed by the FCU
        self._param_futures = {}
        self._param_last_try = 0.0
        self._timer = node.create_timer(1.0 / setpoint_hz, self._tick)
        self._ticks = 0

    # ---- ROS callbacks ---------------------------------------------------------------------
    def _on_state(self, msg):
        self._state = msg

    def _on_pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        self._pose = (p.x, p.y, p.z)
        self._yaw_now = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def _tick(self):
        self._send_params()
        if self._target is None:
            return
        msg = self._PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self._target
        msg.pose.orientation.z, msg.pose.orientation.w = math.sin(self._yaw / 2), math.cos(self._yaw / 2)
        self._sp_pub.publish(msg)
        self._ticks += 1
        # after a short setpoint stream, request OFFBOARD then arming, retrying every second
        if self._ticks % 20 == 0 and self._state is not None:
            if self._pending_offboard and self._state.mode != "OFFBOARD":
                self._call_mode("OFFBOARD")
            elif self._pending_arm and not self._state.armed:
                self._call_arm(True)

    def _call_mode(self, mode):
        from mavros_msgs.srv import SetMode
        if self._mode_cli.service_is_ready():
            req = SetMode.Request(); req.custom_mode = mode
            self._mode_cli.call_async(req)

    def _send_params(self):
        """Set PX4 params through MAVROS, retrying until the FCU confirms each one.

        MAVROS refuses sets until its initial parameter pull has finished, which takes longer than
        our takeoff, so a single fire-and-forget request silently fails."""
        from mavros_msgs.srv import ParamSetV2
        from rcl_interfaces.msg import ParameterValue
        if not self._params_pending or not self._param_cli.service_is_ready():
            return
        for name, fut in list(self._param_futures.items()):
            if fut.done():
                res = fut.result()
                if res is not None and res.success:
                    self._params_pending.pop(name, None)
                    self.node.get_logger().info(f"PX4 {name} = {res.value.double_value:.2f} confirmed")
                del self._param_futures[name]
        now = self.node.get_clock().now().nanoseconds * 1e-9
        if now - self._param_last_try < 2.0:
            return
        self._param_last_try = now
        for name, value in self._params_pending.items():
            if name in self._param_futures:
                continue
            req = ParamSetV2.Request()
            req.param_id = name
            req.value = ParameterValue(type=3, double_value=float(value))   # 3 = PARAMETER_DOUBLE -> MAV_PARAM_TYPE_REAL32
            self._param_futures[name] = self._param_cli.call_async(req)

    def _call_arm(self, value):
        from mavros_msgs.srv import CommandBool
        if self._arm_cli.service_is_ready():
            req = CommandBool.Request(); req.value = value
            self._arm_cli.call_async(req)

    # ---- FlightController ------------------------------------------------------------------
    def connected(self):
        return self._state is not None and self._state.connected and self._pose is not None

    def position(self):
        return self._pose

    def armed(self):
        return bool(self._state and self._state.armed)

    def mode(self):
        return self._state.mode if self._state else ""

    def target(self):
        return self._target

    def takeoff(self, altitude):
        x, y, _ = self._pose
        self._yaw = getattr(self, "_yaw_now", 0.0)
        self._target = (float(x), float(y), float(altitude))
        self._ticks = 0
        self._pending_offboard, self._pending_arm = True, True

    def goto(self, x, y, z, yaw=None):
        if yaw is not None:
            self._yaw = float(yaw)
        self._target = (float(x), float(y), float(z))

    def land(self):
        from mavros_msgs.srv import CommandTOL
        self._pending_offboard = self._pending_arm = False
        self._target = None
        self._call_mode("AUTO.LAND")
