import numpy as np

from uav_dt.geotracker import GeoTracker


def test_same_bowl_seen_in_consecutive_frames_keeps_one_id():
    tr = GeoTracker(gate_m=1.5)
    assert tr.update([[10.0, 5.0, 0.6, 40]], t=0.0) == []            # first sighting is tentative
    conf = tr.update([[10.3, 4.8, 0.5, 40]], t=1.0)                  # 0.36 m away: same bowl
    assert len(conf) == 1 and conf[0].id == 1 and conf[0].hits == 2
    assert abs(conf[0].x - 10.15) < 1e-6 and abs(conf[0].y - 4.9) < 1e-6


def test_two_bowls_beyond_gate_get_two_ids():
    tr = GeoTracker(gate_m=1.5)
    tr.update([[0.0, 0.0, 0.6, 40], [3.0, 0.0, 0.6, 40]], t=0.0)
    conf = tr.update([[0.1, 0.0, 0.6, 40], [3.1, 0.0, 0.6, 40]], t=1.0)
    assert sorted(t.id for t in conf) == [1, 2]


def test_low_confidence_recovers_but_does_not_spawn():
    tr = GeoTracker(gate_m=1.5, high_thresh=0.4, low_thresh=0.1)
    tr.update([[0.0, 0.0, 0.6, 40]], t=0.0)
    tr.update([[0.2, 0.0, 0.2, 40], [50.0, 50.0, 0.2, 40]], t=1.0)   # weak hit on the bowl, weak noise elsewhere
    assert len(tr.tracks) == 1 and tr.tracks[0].hits == 2


def test_misses_only_count_when_in_view():
    tr = GeoTracker(gate_m=1.5, max_misses=2)
    tr.update([[0.0, 0.0, 0.6, 40]], t=0.0)
    for k in range(5):
        tr.update([], t=1.0 + k, in_view=lambda x, y: False)          # flown away: no penalty
    assert len(tr.tracks) == 1
    for k in range(3):
        tr.update([], t=10.0 + k, in_view=lambda x, y: True)          # in view and absent: dropped
    assert len(tr.tracks) == 0


def test_confirmed_track_survives_misses():
    tr = GeoTracker(gate_m=1.5, max_misses=1)
    tr.update([[0.0, 0.0, 0.6, 40]], t=0.0)
    tr.update([[0.0, 0.0, 0.6, 40]], t=1.0)
    for k in range(5):
        tr.update([], t=2.0 + k)
    assert len(tr.confirmed()) == 1
