"""Pixel to ground geolocation for a nadir-ish camera on a UAV, and the inverse projection.

Frames (all right-handed):
  world   ENU, origin at the takeoff point. Ground is the plane z = ground_z.
  body    FLU (x forward, y left, z up), as published by MAVROS local_position/pose.
  sensor  Gazebo camera convention: x along the optical axis, y left, z up in the image.
  image   u right, v down, origin top-left, principal point at the centre.

The survey camera in sim/models/x500_nadir_cam is mounted with pitch +pi/2 in the body frame,
so its optical axis points down and, at zero yaw, image right is world -y and image down is
world -x. `CameraModel.mount` holds that rotation so the maths stays generic.

Numpy only, so it is testable without ROS.
"""

from __future__ import annotations

__author__ = "Frank Loewenich"

from dataclasses import dataclass, field
import math

import numpy as np

EARTH_R = 6371000.0


def quat_to_rot(q) -> np.ndarray:
    """Quaternion (x, y, z, w) to a 3x3 rotation matrix (ROS ordering)."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation from body to world for ZYX Euler angles (yaw about z, then pitch about y, then roll about x)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


@dataclass
class CameraModel:
    """Pinhole camera with square pixels, a horizontal field of view and a mount rotation."""
    width: int
    height: int
    hfov: float                                   # radians
    mount: np.ndarray = field(default_factory=lambda: rot_rpy(0.0, math.pi / 2, 0.0))  # sensor -> body

    @classmethod
    def dji_mini4pro_still(cls) -> "CameraModel":
        """Camera model of a DJI Mini 4 Pro 4:3 still: 4032 x 3024 px, 70 degree horizontal field of view."""
        return cls(4032, 3024, math.radians(70.0))

    @property
    def fx(self) -> float:
        """Focal length in pixels along the image width."""
        return (self.width / 2.0) / math.tan(self.hfov / 2.0)

    @property
    def fy(self) -> float:
        """Focal length in pixels along the image height (square pixels)."""
        return self.fx

    @property
    def cx(self) -> float:
        """Principal point column, the image centre."""
        return self.width / 2.0

    @property
    def cy(self) -> float:
        """Principal point row, the image centre."""
        return self.height / 2.0

    def ground_sample_distance(self, height_agl: float) -> float:
        """Metres per pixel at nadir for the given height above ground."""
        return height_agl / self.fx

    def ray_sensor(self, u: float, v: float) -> np.ndarray:
        """Unit-free direction of pixel (u, v) in the Gazebo sensor frame (x forward, y left, z up)."""
        return np.array([1.0, -(u - self.cx) / self.fx, -(v - self.cy) / self.fy])


def geolocate(cam: CameraModel, u: float, v: float, position, orientation_q, ground_z: float = 0.0):
    """World ENU point where the ray through pixel (u, v) meets the plane z = ground_z.

    position: (x, y, z) of the body in world ENU. orientation_q: body->world quaternion (x, y, z, w).
    Returns None if the ray does not hit the ground (points at or above the horizon).
    """
    r_wb = quat_to_rot(orientation_q)
    d_w = r_wb @ cam.mount @ cam.ray_sensor(u, v)
    p = np.asarray(position, dtype=float)
    if d_w[2] >= -1e-9:
        return None
    t = (ground_z - p[2]) / d_w[2]
    return p + t * d_w


def project(cam: CameraModel, point, position, orientation_q):
    """Inverse of geolocate: pixel (u, v) of a world point, or None if behind the camera."""
    r_wb = quat_to_rot(orientation_q)
    d_s = cam.mount.T @ r_wb.T @ (np.asarray(point, dtype=float) - np.asarray(position, dtype=float))
    if d_s[0] <= 1e-9:
        return None
    u = cam.cx - cam.fx * d_s[1] / d_s[0]
    v = cam.cy - cam.fy * d_s[2] / d_s[0]
    return float(u), float(v)


def enu_to_latlon(x: float, y: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """Convert local ENU metres to latitude and longitude with a flat-earth approximation."""
    lat = origin_lat + math.degrees(y / EARTH_R)
    lon = origin_lon + math.degrees(x / (EARTH_R * math.cos(math.radians(origin_lat))))
    return lat, lon


def latlon_to_enu(lat: float, lon: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    """Convert latitude and longitude to local ENU metres with a flat-earth approximation."""
    y = math.radians(lat - origin_lat) * EARTH_R
    x = math.radians(lon - origin_lon) * EARTH_R * math.cos(math.radians(origin_lat))
    return x, y
