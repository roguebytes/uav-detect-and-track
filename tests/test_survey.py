__author__ = "Frank Loewenich"

import math

import pytest

from uav_dt.geolocate import CameraModel
from uav_dt.mission.survey import footprint, lawnmower


def test_footprint_at_40m():
    width, length = footprint(CameraModel.dji_mini4pro_still(), 40.0)
    assert 55 < width < 57 and 41 < length < 43


def test_lawnmower_sized_field_gives_a_symmetric_sweep():
    cam = CameraModel.dji_mini4pro_still()
    swath, along = footprint(cam, 40.0)
    fh = swath + 0.9 * swath                                   # two passes at 10% overlap fit exactly
    wps = lawnmower(120, fh, cam, 40.0)
    xs = sorted({wp[0] for wp in wps})
    ys = sorted({wp[1] for wp in wps})
    assert xs[0] == pytest.approx(-60 + along / 2) and xs[-1] == pytest.approx(60 - along / 2)
    assert len(ys) == 2 and ys[0] == pytest.approx(-ys[1]) and ys[1] - ys[0] == pytest.approx(0.9 * swath)
    assert ys[0] - swath / 2 == pytest.approx(-fh / 2) and ys[1] + swath / 2 == pytest.approx(fh / 2)
    assert wps[0][3] == 0.0 and wps[2][3] == math.pi


def test_lawnmower_edge_aligned_overshoots_a_short_field():
    cam = CameraModel.dji_mini4pro_still()
    swath, _ = footprint(cam, 40.0)
    ys = sorted({wp[1] for wp in lawnmower(120, 80, cam, 40.0)})
    assert len(ys) == 2 and ys[0] == pytest.approx(-40 + swath / 2) and ys[1] + swath / 2 > 40
    cys = sorted({wp[1] for wp in lawnmower(120, 80, cam, 40.0, edge_aligned=False)})
    assert cys[0] == pytest.approx(-40 + swath / 2) and cys[1] == pytest.approx(40 - swath / 2)


def test_lawnmower_small_field_single_midline_leg():
    cam = CameraModel.dji_mini4pro_still()
    wps = lawnmower(30, 20, cam, 40.0)
    assert len(wps) == 2 and wps[0][0] == wps[1][0] == 0.0 and wps[0][1] == 0.0
