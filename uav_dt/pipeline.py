"""Orchestration: frame -> detections -> tracks -> selected target offset."""
from __future__ import annotations

from .geometry import box_area, normalized_offset
from .tracker import ByteTrackLite


def select_target(tracks, frame_hw, policy="best"):
    """Pick one target from the confirmed tracks and report its normalised offset.

    policy: 'best' (highest score), 'largest' (biggest box), or an int track id to lock."""
    if not tracks:
        return None
    if isinstance(policy, int):
        chosen = next((t for t in tracks if t[0] == policy), None)
    elif policy == "largest":
        chosen = max(tracks, key=lambda t: box_area(t[1]))
    else:  # "best"
        chosen = max(tracks, key=lambda t: t[2])
    if chosen is None:
        return None
    tid, box, score, cls = chosen
    ox, oy = normalized_offset(box, frame_hw)
    return {"id": int(tid), "cls": int(cls), "score": float(score),
            "box": [float(v) for v in box], "offset": [float(ox), float(oy)]}


class Pipeline:
    def __init__(self, detector, tracker=None, target_policy="best"):
        self.detector = detector
        self.tracker = tracker or ByteTrackLite()
        self.target_policy = target_policy

    def process(self, frame, frame_idx, frame_hw=None):
        if frame_hw is None:
            frame_hw = frame.shape[:2]
        dets = self.detector.detect(frame)
        tracks = self.tracker.update(dets, frame_hw)
        target = select_target(tracks, frame_hw, self.target_policy)
        return {
            "frame": int(frame_idx),
            "tracks": [
                {"id": int(i), "cls": int(c), "score": float(s),
                 "box": [float(v) for v in b]}
                for (i, b, s, c) in tracks
            ],
            "target": target,
        }
