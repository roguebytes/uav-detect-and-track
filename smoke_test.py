"""Offline end-to-end smoke test: numpy only, no weights, no ROS, no Gazebo.

Flies a synthetic 40 m lawnmower over a few bowls with the ground-truth oracle detector, then
checks that the pipeline geolocates each bowl to within the tracker gate and that track IDs stay
stable across passes.

    python smoke_test.py
"""
from __future__ import annotations

import math

import numpy as np

from uav_dt.detector import GroundTruthDetector
from uav_dt.geolocate import CameraModel
from uav_dt.geotracker import GeoTracker
from uav_dt.pipeline import GeoPipeline

BOWLS = [(-20.0, 10.0), (5.0, -3.0), (18.0, 12.0), (-2.0, -14.0)]


def lawnmower(x0, x1, y0, y1, spacing, step):
    """Yield (x, y, yaw) along east-west legs."""
    y, leg = y0, 0
    while y <= y1:
        xs = np.arange(x0, x1 + 1e-6, step) if leg % 2 == 0 else np.arange(x1, x0 - 1e-6, -step)
        for x in xs:
            yield float(x), float(y), 0.0 if leg % 2 == 0 else math.pi
        y += spacing
        leg += 1


def main():
    cam = CameraModel.dji_mini4pro_still()
    det = GroundTruthDetector(cam, BOWLS, pixel_noise=1.0, miss_rate=0.1, false_positives=0, seed=3)
    pipe = GeoPipeline(det, cam, GeoTracker(gate_m=1.5))
    frame = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)   # oracle ignores pixels
    swath = 2 * 40.0 * math.tan(cam.hfov / 2) * 0.7                  # 30% side overlap
    t, n_frames = 0.0, 0
    for x, y, yaw in lawnmower(-40, 40, -25, 25, swath, 5.0):
        q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        pipe.process(frame, t, (x, y, 40.0), q)
        t += 1.0
        n_frames += 1
    tracks = pipe.tracker.confirmed()
    assert len(tracks) == len(BOWLS), f"expected {len(BOWLS)} confirmed tracks, got {len(tracks)}"
    errs = []
    for bx, by in BOWLS:
        d, tr = min((math.hypot(tr.x - bx, tr.y - by), tr) for tr in tracks)
        assert d < 0.5, f"bowl at ({bx},{by}) geolocated {d:.2f} m off"
        errs.append(d)
    ids = sorted(tr.id for tr in tracks)
    assert ids == list(range(1, len(BOWLS) + 1)), f"track ids not contiguous: {ids}"
    print(f"frames={n_frames} tracks={len(tracks)} mean geolocation error={np.mean(errs):.3f} m "
          f"max={max(errs):.3f} m hits={[tr.hits for tr in tracks]}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
