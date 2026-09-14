import math

import numpy as np

from uav_dt.detector import GroundTruthDetector, StubDetector, drop_edge_boxes, nms, tile_origins
from uav_dt.geolocate import CameraModel


def test_tile_grid_covers_frame_with_overlap():
    grid = tile_origins(3024, 4032, 640, 0.2)
    ys = sorted({y for y, _ in grid}); xs = sorted({x for _, x in grid})
    assert ys[0] == 0 and xs[0] == 0
    assert ys[-1] + 640 == 3024 and xs[-1] + 640 == 4032
    assert all(b - a <= 512 for a, b in zip(ys, ys[1:]))
    assert len(grid) == 48


def test_nms_merges_duplicates_across_tile_overlap():
    dets = np.array([[100, 100, 114, 112, 0.6, 0], [101, 100, 115, 113, 0.5, 0], [500, 500, 514, 512, 0.4, 0]], dtype=np.float32)
    kept = nms(dets, 0.5)
    assert len(kept) == 2 and kept[0, 4] == 0.6


def test_edge_boxes_dropped():
    dets = np.array([[0, 10, 14, 22, 0.9, 0], [100, 100, 114, 112, 0.9, 0]], dtype=np.float32)
    assert len(drop_edge_boxes(dets, 3024, 4032, 4)) == 1


def test_stub_detector_sequence():
    d = StubDetector([[[0, 0, 1, 1, 0.9, 0]], []])
    assert d.detect().shape == (1, 6) and d.detect().shape == (0, 6) and d.detect().shape == (0, 6)


def test_ground_truth_detector_projects_bowls_in_footprint():
    cam = CameraModel.dji_mini4pro_still()
    bowls = [(0.0, 0.0), (10.0, 5.0), (100.0, 100.0)]   # third is far outside the 40 m footprint
    d = GroundTruthDetector(cam, bowls, pixel_noise=0.0, seed=1)
    d.set_pose((0.0, 0.0, 40.0), (0.0, 0.0, 0.0, 1.0))
    dets = d.detect(np.zeros((cam.height, cam.width, 3), dtype=np.uint8))
    assert len(dets) == 2
    centre = dets[np.argmin(np.abs(dets[:, 0] - cam.cx))]
    assert math.isclose((centre[0] + centre[2]) / 2, cam.cx, abs_tol=1e-6)
    assert 10 <= centre[2] - centre[0] <= 13     # about 11.5 px wide at 40 m
