__author__ = "Frank Loewenich"

import math

import numpy as np
import pytest

from uav_dt.geolocate import CameraModel, enu_to_latlon, geolocate, latlon_to_enu, project, rot_rpy


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


CAM = CameraModel.dji_mini4pro_still()
IDENT = (0.0, 0.0, 0.0, 1.0)


def test_centre_pixel_hits_nadir_point():
    p = geolocate(CAM, CAM.cx, CAM.cy, (3.0, -2.0, 40.0), IDENT)
    assert p is not None
    assert np.allclose(p, [3.0, -2.0, 0.0], atol=1e-6)


def test_image_axes_map_to_world_axes_at_zero_yaw():
    # matches the convention verified against Gazebo renders on 2026-09-14:
    # image right is world -y, image down is world -x
    half_w = 40.0 * math.tan(CAM.hfov / 2)
    right = geolocate(CAM, CAM.width, CAM.cy, (0, 0, 40.0), IDENT)
    assert np.allclose(right, [0.0, -half_w, 0.0], atol=1e-6)
    down = geolocate(CAM, CAM.cx, CAM.height, (0, 0, 40.0), IDENT)
    half_h = 40.0 * (CAM.cy / CAM.fy)
    assert np.allclose(down, [-half_h, 0.0, 0.0], atol=1e-6)


def test_yaw_rotates_footprint():
    # yaw +90 deg (nose to world +y): image down (body -x) now points to world -y
    down = geolocate(CAM, CAM.cx, CAM.height, (0, 0, 40.0), quat_from_yaw(math.pi / 2))
    half_h = 40.0 * (CAM.cy / CAM.fy)
    assert np.allclose(down, [0.0, -half_h, 0.0], atol=1e-6)


def test_project_inverts_geolocate():
    q = quat_from_yaw(0.7)
    pos = (12.0, -5.0, 38.5)
    for u, v in [(100, 200), (2000, 1500), (4000, 3000), (CAM.cx, CAM.cy)]:
        g = geolocate(CAM, u, v, pos, q, ground_z=0.5)
        uv = project(CAM, g, pos, q)
        assert uv is not None and np.allclose(uv, (u, v), atol=1e-6)


def test_tilted_camera_still_hits_ground():
    # 10 deg roll: the footprint shifts but every pixel still lands on the ground
    r = rot_rpy(math.radians(10), 0, 0)
    # rotation matrix -> quaternion via a trusted path
    w = math.sqrt(1 + np.trace(r)) / 2
    q = ((r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w)
    g = geolocate(CAM, 0, 0, (0, 0, 40.0), q)
    assert g is not None and abs(g[2]) < 1e-9


def test_horizon_returns_none():
    # pitch the body nose-up by 90 deg so the camera looks forward along the horizon
    q = (0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))
    assert geolocate(CAM, CAM.cx, CAM.cy, (0, 0, 40.0), q) is None


def test_gsd_and_bowl_pixels():
    # 16 cm bowl should be about 12 px at 40 m and 44 px at 11 m, as measured on the renders
    assert 0.16 / CAM.ground_sample_distance(40.0) == pytest.approx(11.5, abs=1.0)
    assert 0.16 / CAM.ground_sample_distance(11.0) == pytest.approx(42.0, abs=3.0)


def test_latlon_roundtrip():
    lat0, lon0 = 47.397971, 8.546164
    lat, lon = enu_to_latlon(50.0, -30.0, lat0, lon0)
    x, y = latlon_to_enu(lat, lon, lat0, lon0)
    assert (x, y) == pytest.approx((50.0, -30.0), abs=1e-6)
