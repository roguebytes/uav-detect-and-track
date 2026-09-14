import math

from uav_dt.geolocate import CameraModel
from uav_dt.mission.survey import footprint, lawnmower, path_length


def test_footprint_at_40m():
    w, l = footprint(CameraModel.dji_mini4pro_still(), 40.0)
    assert 55 < w < 57 and 41 < l < 43


def test_lawnmower_covers_field():
    cam = CameraModel.dji_mini4pro_still()
    wps = lawnmower(120, 80, cam, 40.0, side_overlap=0.3)
    swath, _ = footprint(cam, 40.0)
    ys = sorted({wp[1] for wp in wps})
    assert len(wps) == 2 * len(ys) and len(ys) == 3          # 80 m field, 39 m spacing -> 3 legs
    assert all(b - a <= swath * 0.7 + 1e-6 for a, b in zip(ys, ys[1:]))
    assert ys[0] - swath / 2 <= -40 and ys[-1] + swath / 2 >= 40   # edges covered
    assert wps[0][3] == 0.0 and wps[2][3] == math.pi          # alternating leg direction
    assert path_length(wps) > 3 * 120
