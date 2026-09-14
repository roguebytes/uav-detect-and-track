"""Multi-object tracking on the ground plane.

At 40 m, 1 Hz and 5 m/s a bowl moves about 360 px between frames while its box is 14 px wide,
so image-space IoU association cannot work. Each detection is geolocated first and tracks live
in world ENU coordinates. Association is by ground distance with the same two-stage idea as
ByteTrack (high-confidence detections first, then low-confidence ones recover unmatched tracks).
Tracks are stationary targets, so the state is a running mean of observed positions.

Numpy only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import greedy_match


@dataclass
class GeoTrack:
    id: int
    x: float
    y: float
    score: float
    hits: int = 1
    misses: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0
    observations: list = field(default_factory=list)   # (t, x, y, score, height_agl)
    verified: bool | None = None                        # None until the verify stage runs

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2

    def update(self, x: float, y: float, score: float, t: float, height_agl: float) -> None:
        self.observations.append((t, x, y, score, height_agl))
        n = len(self.observations)
        self.x = self.x + (x - self.x) / n
        self.y = self.y + (y - self.y) / n
        self.score = max(self.score, score)
        self.hits += 1
        self.misses = 0
        self.last_seen = t


class GeoTracker:
    """detections: (N, 4) rows [x, y, score, height_agl] in world ENU metres.

    gate_m: max ground distance for association. Two bowls closer than this merge into one
    track, so keep it below the world generator's minimum bowl spacing.
    """

    def __init__(self, gate_m: float = 1.5, high_thresh: float = 0.4, low_thresh: float = 0.1,
                 min_hits: int = 2, max_misses: int = 50):
        self.gate_m, self.high_thresh, self.low_thresh = gate_m, high_thresh, low_thresh
        self.min_hits, self.max_misses = min_hits, max_misses
        self.tracks: list[GeoTrack] = []
        self._next_id = 1

    def _affinity(self, tracks: list[GeoTrack], dets: np.ndarray) -> np.ndarray:
        """Affinity in (0, 1], 1 at zero distance, 0 beyond the gate. Shaped for greedy_match."""
        if not tracks or len(dets) == 0:
            return np.zeros((len(tracks), len(dets)), dtype=np.float32)
        tp = np.array([[t.x, t.y] for t in tracks], dtype=np.float32)
        d = np.linalg.norm(tp[:, None, :] - dets[None, :, :2], axis=2)
        return np.where(d < self.gate_m, 1.0 - d / self.gate_m, 0.0).astype(np.float32)

    def update(self, detections, t: float, in_view=None) -> list[GeoTrack]:
        """Associate one frame of geolocated detections. Returns the confirmed tracks.

        in_view: optional callable (x, y) -> bool saying whether a ground point is inside the
        current camera footprint. Tracks in view but unmatched count a miss; tracks out of view
        are left alone, since not seeing them tells us nothing."""
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 4)
        high = dets[dets[:, 2] >= self.high_thresh]
        low = dets[(dets[:, 2] >= self.low_thresh) & (dets[:, 2] < self.high_thresh)]
        eps = 1e-6

        m1, un_t, un_high = greedy_match(self._affinity(self.tracks, high), eps)
        for ti, di in m1:
            self.tracks[ti].update(float(high[di][0]), float(high[di][1]), float(high[di][2]), t, float(high[di][3]))

        remaining = [self.tracks[i] for i in un_t]
        m2, un_rem, _ = greedy_match(self._affinity(remaining, low), eps)
        for ti, di in m2:
            remaining[ti].update(float(low[di][0]), float(low[di][1]), float(low[di][2]), t, float(low[di][3]))

        for i in un_rem:
            tr = remaining[i]
            if in_view is None or in_view(tr.x, tr.y):
                tr.misses += 1

        for di in un_high:
            x, y, s, h = (float(v) for v in high[di])
            tr = GeoTrack(self._next_id, x, y, s, first_seen=t, last_seen=t)
            tr.observations.append((t, x, y, s, h))
            self.tracks.append(tr)
            self._next_id += 1

        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_misses or tr.confirmed]
        return [tr for tr in self.tracks if tr.hits >= self.min_hits]

    def confirmed(self) -> list[GeoTrack]:
        return [tr for tr in self.tracks if tr.hits >= self.min_hits]

    def get(self, track_id: int) -> GeoTrack | None:
        return next((tr for tr in self.tracks if tr.id == track_id), None)
