"""GeoPipeline: frame + vehicle pose -> image detections -> ground points -> tracks.

Pure Python so it runs offline (unit tests, replay of saved frames) and inside the ROS node.
"""

from __future__ import annotations

__author__ = "Frank Loewenich"

from dataclasses import dataclass, field

import numpy as np

from .detector import EMPTY
from .geolocate import CameraModel, geolocate, project
from .geotracker import GeoTracker


@dataclass
class FrameResult:
    """Detections, ground points and confirmed tracks produced from one frame."""
    t: float
    position: tuple
    orientation: tuple
    detections: np.ndarray                      # (N, 6) image boxes
    ground_points: list                         # per detection: (x, y) ENU or None
    tracks: list = field(default_factory=list)  # confirmed GeoTrack objects after this frame
    inference_s: float = 0.0

    def to_record(self) -> dict:
        """Serialise the result for the JSONL log."""
        return {
            "t": self.t,
            "position": [float(v) for v in self.position],
            "orientation": [float(v) for v in self.orientation],
            "detections": [
                {"box": [float(v) for v in d[:4]], "score": float(d[4]),
                 "ground": None if g is None else [float(g[0]), float(g[1])]}
                for d, g in zip(self.detections, self.ground_points)
            ],
            "tracks": [
                {"id": tr.id, "x": tr.x, "y": tr.y, "score": tr.score, "hits": tr.hits, "verified": tr.verified}
                for tr in self.tracks
            ],
            "inference_s": self.inference_s,
        }


class GeoPipeline:
    """Detector, geolocation and tracker chained for one frame at a time."""
    def __init__(self, detector, camera: CameraModel, tracker: GeoTracker | None = None, ground_z: float = 0.0):
        """Store the detector, camera, tracker and ground height."""
        self.detector, self.cam, self.ground_z = detector, camera, ground_z
        self.tracker = tracker or GeoTracker()

    def footprint_test(self, position, orientation):
        """Callable (x, y) -> bool: is the ground point inside the camera frame for this pose?"""
        w, h, margin = self.cam.width, self.cam.height, 0.05

        def in_view(x, y):
            uv = project(self.cam, (x, y, self.ground_z), position, orientation)
            return uv is not None and margin * w <= uv[0] <= (1 - margin) * w and margin * h <= uv[1] <= (1 - margin) * h
        return in_view

    def process(self, frame, t: float, position, orientation) -> FrameResult:
        """Detect in a frame, geolocate the detections and update the tracks."""
        import time
        if hasattr(self.detector, "set_pose"):
            self.detector.set_pose(position, orientation)
        t0 = time.perf_counter()
        dets = self.detector.detect(frame)
        dt = time.perf_counter() - t0
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6) if len(dets) else EMPTY
        height_agl = float(position[2]) - self.ground_z
        ground, rows = [], []
        for d in dets:
            u, v = (d[0] + d[2]) / 2.0, (d[1] + d[3]) / 2.0
            g = geolocate(self.cam, u, v, position, orientation, self.ground_z)
            ground.append(None if g is None else (float(g[0]), float(g[1])))
            if g is not None:
                rows.append([g[0], g[1], d[4], height_agl])
        tracks = self.tracker.update(rows, t, in_view=self.footprint_test(position, orientation))
        return FrameResult(t, tuple(position), tuple(orientation), dets, ground, tracks, dt)
