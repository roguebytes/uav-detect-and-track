import math

import pytest

from uav_dt.geolocate import CameraModel
from uav_dt.mission.survey import footprint, lawnmower, path_length


def test_footprint_at_40m():
    w, l = footprint(CameraModel.dji_mini4pro_still(), 40.0)
    assert 55 < w < 57 and 41 < l < 43


def test_lawnmower_footprint_touches_the_boundary():
    cam = CameraModel.dji_mini4pro_still()
    wps = lawnmower(120, 80, cam, 40.0, side_overlap=0.3)
    swath, along = footprint(cam, 40.0)
    xs = sorted({wp[0] for wp in wps}); ys = sorted({wp[1] for wp in wps})
    # leg ends: the footprint's leading edge reaches the field edge, the aircraft does not
    assert xs[0] == pytest.approx(-60 + along / 2) and xs[-1] == pytest.approx(60 - along / 2)
    # outer legs: the footprint's side edge reaches the side of the field
    assert ys[0] == pytest.approx(-40 + swath / 2) and ys[-1] == pytest.approx(40 - swath / 2)
    assert len(ys) == 2                                        # 80 m field, 56 m swath: two legs 24 m apart
    assert all(b - a <= swath * 0.7 + 1e-6 for a, b in zip(ys, ys[1:]))
    assert wps[0][3] == 0.0 and wps[2][3] == math.pi          # alternating leg direction
    assert len(wps) == 2 * len(ys)


def test_lawnmower_small_field_single_midline_leg():
    cam = CameraModel.dji_mini4pro_still()
    wps = lawnmower(30, 20, cam, 40.0)
    assert len(wps) == 2 and wps[0][0] == wps[1][0] == 0.0 and wps[0][1] == 0.0
