"""ByteTrackLite — a compact ByteTrack-style multi-object tracker.

Implements ByteTrack's core idea (two-stage association: match tracks to
high-confidence detections first, then recover with low-confidence ones) using
IoU only — no deep features, no Kalman filter. Dependency-free (numpy). For
production accuracy, swap in the reference ByteTrack / OC-SORT or Ultralytics'
built-in tracker behind the same interface.
"""
from __future__ import annotations

import numpy as np

from .geometry import greedy_match, iou_matrix


class Track:
    __slots__ = ("id", "box", "score", "cls", "age", "hits", "state")

    def __init__(self, tid, box, score, cls):
        self.id = tid
        self.box = np.asarray(box, dtype=np.float32)
        self.score = float(score)
        self.cls = int(cls)
        self.age = 0          # frames since last successful update
        self.hits = 1         # total successful updates
        self.state = "tentative"


class ByteTrackLite:
    def __init__(self, track_thresh=0.5, low_thresh=0.1, match_thresh=0.3,
                 max_age=30, min_hits=3):
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_id = 1
        self.tracks: list[Track] = []

    def update(self, detections, frame_hw=None):
        """detections: (N,6) array [x1,y1,x2,y2,score,cls]. Returns confirmed,
        currently-visible tracks as list of (id, box, score, cls)."""
        dets = np.asarray(detections, dtype=np.float32).reshape(-1, 6) \
            if len(detections) else np.zeros((0, 6), dtype=np.float32)
        high = dets[dets[:, 4] >= self.track_thresh]
        low = dets[(dets[:, 4] >= self.low_thresh) & (dets[:, 4] < self.track_thresh)]

        track_boxes = np.array([t.box for t in self.tracks], dtype=np.float32) \
            if self.tracks else np.zeros((0, 4), dtype=np.float32)

        # Stage 1: existing tracks vs high-confidence detections.
        m1, un_t, un_high = greedy_match(iou_matrix(track_boxes, high[:, :4]), self.match_thresh)
        for ti, di in m1:
            self._update_track(self.tracks[ti], high[di])

        # Stage 2: still-unmatched tracks vs low-confidence detections.
        rem = [self.tracks[i] for i in un_t]
        rem_boxes = np.array([t.box for t in rem], dtype=np.float32) \
            if rem else np.zeros((0, 4), dtype=np.float32)
        m2, un_rem, _ = greedy_match(iou_matrix(rem_boxes, low[:, :4]), self.match_thresh)
        for ti, di in m2:
            self._update_track(rem[ti], low[di])

        # Age out tracks unmatched in both stages.
        for i in un_rem:
            rem[i].age += 1

        # Spawn new tracks from unmatched high-confidence detections.
        for di in un_high:
            self._spawn(high[di])

        # Drop stale tracks.
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        return [(t.id, t.box.copy(), t.score, t.cls)
                for t in self.tracks if t.state == "confirmed" and t.age == 0]

    def _update_track(self, t: Track, det):
        t.box = det[:4].astype(np.float32)
        t.score = float(det[4])
        t.cls = int(det[5])
        t.age = 0
        t.hits += 1
        if t.hits >= self.min_hits:
            t.state = "confirmed"

    def _spawn(self, det):
        self.tracks.append(Track(self._next_id, det[:4], det[4], det[5]))
        self._next_id += 1
