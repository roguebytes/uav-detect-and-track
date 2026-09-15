#!/usr/bin/env python3
"""Judder check for a clip: fraction of duplicated consecutive frames and how uneven the motion is.

    python3 tools/video_smoothness.py clip.mp4 [--max-frames 1800]

Reports duplicates (frames identical to the previous one), the longest duplicate run, and the
alternation index: median |d[i+1] - d[i]| / median d over moving frames, where d is the mean
absolute pixel change between consecutive frames. Smooth motion gives an index well under 0.5;
motion quantised at a lower rate than the frame rate alternates big and small steps and scores
above 1.
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--max-frames", type=int, default=1800)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev, diffs, k = None, [], 0
    while k < a.max_frames:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (320, 180)).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev, k = g, k + 1
    d = np.array(diffs)
    dup = d < 0.05
    runs, c = [], 0
    for v in dup:
        c = c + 1 if v else 0
        runs.append(c)
    moving = d[~dup]
    alt = np.median(np.abs(np.diff(moving))) / max(np.median(moving), 1e-6) if len(moving) > 2 else float("nan")
    print(f"{a.clip}: {fps:.0f} fps, {k} frames, duplicates {dup.sum()} ({100 * dup.mean():.1f}%), "
          f"longest duplicate run {max(runs) if runs else 0} frames, alternation index {alt:.2f}")


if __name__ == "__main__":
    main()
