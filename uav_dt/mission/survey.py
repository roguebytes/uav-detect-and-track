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
              margin: float = 0.0, centre=(0.0, 0.0), edge_aligned: bool = False) -> list[tuple[float, float, float, float]]:
    """Waypoints (x, y, z, yaw) covering a field_w x field_h rectangle centred on `centre`.

    The camera footprint, not the aircraft, is what has to reach the boundary (Loewenich et al.
    2026, section 3.2.4: a pass spans only the centres of its first and last footprints). Each leg
    therefore ends when the footprint's leading edge touches the field edge, with the endpoints
    inset by half the along-track footprint.

    Default placement (used for the demo): the first and last passes are inset by half the swath
    so their footprints touch the two sides exactly, and the passes between are spread evenly at no
    more than swath * (1 - side_overlap). The overlap between neighbours is then whatever the
    even spacing gives, never less than side_overlap.

    edge_aligned=True follows the paper's sweep model instead: passes from one side at exactly
    swath * (1 - side_overlap), ceil of the remaining width over the spacing further passes, so a
    partial final pass may overshoot the far side.

    Legs run east-west (along x), yaw along each leg so the camera's along-track axis matches the
    body's. `margin` widens the field before planning."""
    swath, along = footprint(cam, altitude)
    x0, x1 = centre[0] - field_w / 2 - margin, centre[0] + field_w / 2 + margin
    y0, y1 = centre[1] - field_h / 2 - margin, centre[1] + field_h / 2 + margin
    xa, xb = x0 + along / 2, x1 - along / 2                    # leg endpoints: footprint touches the ends
    if xb < xa:                                                 # field shorter than one footprint: fly its midline
        xa = xb = (x0 + x1) / 2
    spacing = swath * (1.0 - side_overlap)
    n_legs = 1 + max(0, math.ceil(((y1 - y0) - swath) / spacing - 1e-9))
    mid = (y0 + y1) / 2
    if n_legs == 1:
        ys = [mid]                                              # one pass covers the width: fly the midline
    elif edge_aligned:
        ys = [y0 + swath / 2 + i * spacing for i in range(n_legs)]
    else:
        ya, yb = y0 + swath / 2, y1 - swath / 2                 # outer passes: footprints touch the sides
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
