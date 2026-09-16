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

    Legs run east-west (along x). The camera footprint, not the aircraft, is what has to reach the
    boundary: each leg ends when the footprint's leading edge touches the field edge (endpoints
    inset by half the along-track footprint), the first and last legs are inset by half the swath
    so their footprints touch the side edges, and the legs in between are spaced evenly at no more
    than swath * (1 - side_overlap). Yaw points the nose along each leg so the camera's along-track
    axis matches the body's. `margin` widens the field before planning."""
    swath, along = footprint(cam, altitude)
    x0, x1 = centre[0] - field_w / 2 - margin, centre[0] + field_w / 2 + margin
    y0, y1 = centre[1] - field_h / 2 - margin, centre[1] + field_h / 2 + margin
    xa, xb = x0 + along / 2, x1 - along / 2                    # leg endpoints: footprint touches the ends
    if xb < xa:                                                 # field shorter than one footprint: fly its midline
        xa = xb = (x0 + x1) / 2
    ya, yb = y0 + swath / 2, y1 - swath / 2                     # outer leg centre lines: footprint touches the sides
    max_spacing = swath * (1.0 - side_overlap)
    if yb <= ya:
        ys = [(y0 + y1) / 2]                                    # one leg covers the field's width
    else:
        n_legs = 1 + math.ceil((yb - ya) / max_spacing - 1e-9)
        ys = [ya + (yb - ya) * i / (n_legs - 1) for i in range(n_legs)]
    wps = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            wps += [(xa, y, altitude, 0.0), (xb, y, altitude, 0.0)]
        else:
            wps += [(xb, y, altitude, math.pi), (xa, y, altitude, math.pi)]
    return wps


def path_length(wps) -> float:
    return sum(math.dist(a[:2], b[:2]) for a, b in zip(wps, wps[1:]))
