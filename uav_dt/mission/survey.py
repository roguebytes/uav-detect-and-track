"""Lawnmower survey planning from the camera footprint."""
from __future__ import annotations

import math

from ..geolocate import CameraModel


def footprint(cam: CameraModel, altitude: float) -> tuple[float, float]:
    """Ground footprint (across-track width, along-track length) in metres at nadir.

    Image right is the body's left, so image width maps across-track when the body flies
    along its x axis."""
    width = 2 * altitude * math.tan(cam.hfov / 2)
    length = 2 * altitude * (cam.cy / cam.fy)
    return width, length


def lawnmower(field_w: float, field_h: float, cam: CameraModel, altitude: float, side_overlap: float = 0.3,
              margin: float = 0.0, centre=(0.0, 0.0)) -> list[tuple[float, float, float, float]]:
    """Waypoints (x, y, z, yaw) covering a field_w x field_h rectangle centred on `centre`.

    Legs run east-west (along x), spaced by the across-track footprint minus the side overlap.
    Yaw points the nose along each leg so the camera's along-track axis matches the body's.
    The leg ends extend half a footprint length past the field edge so the corners are seen."""
    swath, along = footprint(cam, altitude)
    spacing = swath * (1.0 - side_overlap)
    x0, x1 = centre[0] - field_w / 2 - margin, centre[0] + field_w / 2 + margin
    y0, y1 = centre[1] - field_h / 2 - margin, centre[1] + field_h / 2 + margin
    n_legs = max(1, math.ceil((y1 - y0) / spacing))
    ys = [y0 + spacing / 2 + i * spacing for i in range(n_legs)]
    if ys[-1] > y1:
        ys = [y0 + (y1 - y0) * (i + 0.5) / n_legs for i in range(n_legs)]   # spread evenly instead
    wps = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            wps += [(x0, y, altitude, 0.0), (x1, y, altitude, 0.0)]
        else:
            wps += [(x1, y, altitude, math.pi), (x0, y, altitude, math.pi)]
    return wps


def path_length(wps) -> float:
    return sum(math.dist(a[:2], b[:2]) for a, b in zip(wps, wps[1:]))
