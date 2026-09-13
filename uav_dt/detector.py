"""Detectors behind a common interface: detect(frame) -> (N,6) [x1,y1,x2,y2,score,cls]."""
from __future__ import annotations

import numpy as np

EMPTY = np.zeros((0, 6), dtype=np.float32)


class StubDetector:
    """Returns scripted detections per frame — for offline tests, no model needed."""

    def __init__(self, frames_dets):
        self._frames = [np.asarray(d, dtype=np.float32).reshape(-1, 6) for d in frames_dets]
        self._i = 0

    def detect(self, frame=None) -> np.ndarray:
        det = self._frames[self._i] if self._i < len(self._frames) else EMPTY
        self._i += 1
        return det


class YoloDetector:
    """Ultralytics YOLO wrapper. Imported lazily so the package works without torch."""

    def __init__(self, model="yolo11n.pt", conf=0.25, classes=None, device=None):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf
        self.classes = classes
        self.device = device

    def detect(self, frame) -> np.ndarray:
        res = self.model.predict(frame, conf=self.conf, classes=self.classes,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return EMPTY
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()[:, None]
        cls = res.boxes.cls.cpu().numpy()[:, None]
        return np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
