"""Detectors behind one interface: detect(frame_bgr) -> (N, 6) float32 [x1, y1, x2, y2, score, cls].

TiledYoloDetector  an Ultralytics YOLOv9-C trained on the paper's imagery, run SAHI-style on 640 px tiles.
                   Downscaling a 12 MP frame to 640 px finds nothing (tested 2026-09-14).
GroundTruthDetector  projects known bowl positions through the camera model using the true
                   vehicle pose. Used by the CI smoke run so it needs no weights or GPU.
StubDetector       scripted detections for the offline unit tests.
"""

from __future__ import annotations

__author__ = "Frank Loewenich"

import numpy as np

from .geometry import iou_matrix

EMPTY = np.zeros((0, 6), dtype=np.float32)


def tile_origins(h: int, w: int, tile: int, overlap: float) -> list[tuple[int, int]]:
    """Top-left corners of tiles covering an h x w frame, last row and column snapped to the edge."""
    step = max(1, int(tile * (1.0 - overlap)))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    return [(y, x) for y in ys for x in xs]


def nms(dets: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    """Greedy non-maximum suppression on (N, 6) detections, highest score first. Numpy only."""
    if len(dets) == 0:
        return EMPTY
    order = np.argsort(-dets[:, 4])
    dets = dets[order]
    keep = []
    alive = np.ones(len(dets), dtype=bool)
    ious = iou_matrix(dets[:, :4], dets[:, :4])
    for i in range(len(dets)):
        if not alive[i]:
            continue
        keep.append(i)
        alive &= ious[i] < iou_thresh
        alive[i] = False
    return dets[keep]


def drop_edge_boxes(dets: np.ndarray, h: int, w: int, margin: int) -> np.ndarray:
    """Remove boxes touching the frame border: a cut-off bowl is seen whole in the next frame."""
    if len(dets) == 0:
        return dets
    ok = (dets[:, 0] >= margin) & (dets[:, 1] >= margin) & (dets[:, 2] <= w - margin) & (dets[:, 3] <= h - margin)
    return dets[ok]


class StubDetector:
    """Returns scripted detections per frame, for offline tests without a model."""

    def __init__(self, frames_dets):
        """Store the scripted detections, one array per frame."""
        self._frames = [np.asarray(d, dtype=np.float32).reshape(-1, 6) for d in frames_dets]
        self._i = 0

    def detect(self, frame=None) -> np.ndarray:
        """Return the next frame's scripted detections, or none once the script is exhausted."""
        det = self._frames[self._i] if self._i < len(self._frames) else EMPTY
        self._i += 1
        return det


class TiledYoloDetector:
    """Ultralytics YOLO on overlapping tiles at native pixel scale, merged with NMS.

    Imported lazily so the package works without torch. Tiles are batched into one predict call.
    """

    def __init__(self, weights: str, conf: float = 0.25, tile: int = 640, overlap: float = 0.2,
                 imgsz: int = 640, device=None, half: bool = False, edge_margin: int = 4,
                 nms_iou: float = 0.5, batch: int = 16):
        """Load the weights and store the tiling and inference settings."""
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf, self.tile, self.overlap, self.imgsz = conf, tile, overlap, imgsz
        self.device, self.half, self.edge_margin, self.nms_iou, self.batch = device, half, edge_margin, nms_iou, batch
        self.last_tile_count = 0

    def detect(self, frame) -> np.ndarray:
        """Detect on overlapping tiles, merge with NMS and drop boxes touching the frame edge."""
        h, w = frame.shape[:2]
        if h <= self.tile and w <= self.tile:
            grid = [(0, 0)]
        else:
            grid = tile_origins(h, w, self.tile, self.overlap)
        self.last_tile_count = len(grid)
        out = []
        for start in range(0, len(grid), self.batch):
            chunk = grid[start:start + self.batch]
            crops = [frame[y:y + self.tile, x:x + self.tile] for y, x in chunk]
            results = self.model.predict(crops, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                         half=self.half, verbose=False)
            for (y, x), r in zip(chunk, results):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                b = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                b[:, [0, 2]] += x
                b[:, [1, 3]] += y
                s = r.boxes.conf.cpu().numpy()[:, None].astype(np.float32)
                c = r.boxes.cls.cpu().numpy()[:, None].astype(np.float32)
                out.append(np.concatenate([b, s, c], axis=1))
        dets = np.concatenate(out, axis=0) if out else EMPTY
        return drop_edge_boxes(nms(dets, self.nms_iou), h, w, self.edge_margin)


class GroundTruthDetector:
    """Oracle detector for CI: projects the manifest's bowls into the frame using the true pose.

    Call set_pose(position_enu, orientation_xyzw) before each detect(). Adds Gaussian pixel noise,
    drops a fixed fraction of visible bowls, and can inject false positives, all seeded, so the
    tracker and scoring paths are exercised without weights.
    """

    def __init__(self, camera, bowls_xy, ground_z: float = 0.0, bowl_diameter_m: float = 0.16,
                 pixel_noise: float = 1.0, miss_rate: float = 0.0, false_positives: int = 0,
                 score: float = 0.9, seed: int = 0, edge_margin: int = 4):
        """Store the camera model, the bowl positions and the noise settings."""
        from .geolocate import project
        self._project = project
        self.cam, self.bowls, self.ground_z = camera, [tuple(b) for b in bowls_xy], ground_z
        self.diameter, self.noise, self.miss_rate = bowl_diameter_m, pixel_noise, miss_rate
        self.false_positives, self.score, self.edge_margin = false_positives, score, edge_margin
        self.rng = np.random.default_rng(seed)
        self.position, self.orientation = None, None

    def set_pose(self, position, orientation_xyzw) -> None:
        """Set the vehicle pose used to project the bowls for the next frame."""
        self.position, self.orientation = position, orientation_xyzw

    def detect(self, frame=None) -> np.ndarray:
        """Project the visible bowls into the frame with noise, dropouts and false positives."""
        if self.position is None:
            return EMPTY
        h, w = (frame.shape[:2] if frame is not None else (self.cam.height, self.cam.width))
        height_agl = max(float(self.position[2]) - self.ground_z, 0.5)
        radius_px = 0.5 * self.diameter / self.cam.ground_sample_distance(height_agl)
        rows = []
        for x, y in self.bowls:
            uv = self._project(self.cam, (x, y, self.ground_z), self.position, self.orientation)
            if uv is None:
                continue
            u, v = uv
            if not (0 <= u < w and 0 <= v < h) or self.rng.random() < self.miss_rate:
                continue
            u += self.rng.normal(0, self.noise)
            v += self.rng.normal(0, self.noise)
            rows.append([u - radius_px, v - radius_px, u + radius_px, v + radius_px, self.score, 0])
        for _ in range(self.false_positives):
            u, v = self.rng.uniform(0, w), self.rng.uniform(0, h)
            rows.append([u - radius_px, v - radius_px, u + radius_px, v + radius_px, 0.3, 0])
        dets = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        return drop_edge_boxes(dets, h, w, self.edge_margin)
