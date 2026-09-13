"""Geometry helpers: IoU, greedy IoU matching, target offset. Numpy only."""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4), xyxy. Returns (N,M)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (br - tl).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_match(iou: np.ndarray, thresh: float):
    """Greedy highest-IoU-first matching. Returns (matches, unmatched_rows, unmatched_cols)."""
    matches = []
    if iou.size:
        work = iou.copy()
        while True:
            r, c = np.unravel_index(np.argmax(work), work.shape)
            if work[r, c] < thresh:
                break
            matches.append((int(r), int(c)))
            work[r, :] = -1.0
            work[:, c] = -1.0
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    unmatched_r = [r for r in range(iou.shape[0]) if r not in matched_r]
    unmatched_c = [c for c in range(iou.shape[1]) if c not in matched_c]
    return matches, unmatched_r, unmatched_c


def box_area(box) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def normalized_offset(box, frame_hw):
    """Offset of a box centre from image centre, normalised to [-1, 1] per axis."""
    h, w = frame_hw
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return ((cx - w / 2.0) / (w / 2.0), (cy - h / 2.0) / (h / 2.0))
