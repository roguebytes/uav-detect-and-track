"""Offline end-to-end smoke test — numpy only, no model/weights/network.

Drives the pipeline with a StubDetector that scripts an object moving left->right
(plus a second object appearing later) and asserts the tracker produces stable IDs
and a target offset that crosses image centre.

    python smoke_test.py
"""
from __future__ import annotations

import numpy as np

from uav_dt.detector import StubDetector
from uav_dt.pipeline import Pipeline
from uav_dt.tracker import ByteTrackLite

W, H, N = 640, 480, 16


def make_script():
    frames = []
    for k in range(N):
        dets = []
        cx = 100 + 440 * k / (N - 1)            # primary object: 100 -> 540 px
        dets.append([cx - 30, 240 - 30, cx + 30, 240 + 30, 0.9, 0])
        if k >= 8:                               # second object appears top-left
            bx = 120 + (k - 8) * 5
            dets.append([bx - 25, 100 - 25, bx + 25, 100 + 25, 0.8, 2])
        frames.append(np.array(dets, dtype=np.float32))
    return frames


def main():
    pipe = Pipeline(StubDetector(make_script()), ByteTrackLite(min_hits=3, max_age=30))
    frame = np.zeros((H, W, 3), dtype=np.float32)
    results = [pipe.process(frame, k, (H, W)) for k in range(N)]

    targeted = [r for r in results if r["target"] is not None]
    assert targeted, "no target ever selected"

    ids = [r["target"]["id"] for r in targeted]
    assert len(set(ids[:5])) == 1, f"primary target id not stable: {ids[:5]}"

    first_x = targeted[0]["target"]["offset"][0]
    last_x = targeted[-1]["target"]["offset"][0]
    assert first_x < 0 < last_x, f"target offset did not cross centre: {first_x} -> {last_x}"

    all_ids = {t["id"] for r in results for t in r["tracks"]}
    assert len(all_ids) >= 2, f"expected >=2 distinct track ids, got {all_ids}"

    print(f"frames={len(results)} targets={len(targeted)} ids={sorted(all_ids)} "
          f"offset_x {first_x:+.2f} -> {last_x:+.2f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
