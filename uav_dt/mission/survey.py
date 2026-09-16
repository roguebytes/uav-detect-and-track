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


def lawnmower(field_w: float, field_h: float, cam: CameraModel, altitude: float, side_overlap: float = 0.1,
              margin: float = 0.0, centre=(0.0, 0.0), centre_passes: bool = False) -> list[tuple[float, float, float, float]]:
    """Waypoints (x, y, z, yaw) covering a field_w x field_h rectangle centred on `centre`.

    Follows the sweep geometry of Loewenich et al. 2026, section 3.2.4: a pass spans only the
    centres of its first and last camera footprints, so the leg endpoints are inset by half the
    along-track footprint and the aircraft turns when the footprint's edge reaches the boundary.
    Passes run from one side of the field, the first footprint touching that edge, spaced at
    swath * (1 - side_overlap) (the paper's grid model uses one swath; 10% overlap is the mission
    default), with ceil of the remaining width over the spacing further passes, so a partial final
    pass may overshoot the far side. `centre_passes` shares that overshoot between both sides
    instead. Legs run east-west (along x), yaw along each leg so the camera's along-track axis
    matches the body's. `margin` widens the field before planning."""
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
    elif centre_passes:
        ys = [mid + (i - (n_legs - 1) / 2) * spacing for i in range(n_legs)]
    else:
        ys = [y0 + swath / 2 + i * spacing for i in range(n_legs)]
    wps = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            wps += [(xa, y, altitude, 0.0), (xb, y, altitude, 0.0)]
        else:
            wps += [(xb, y, altitude, math.pi), (xa, y, altitude, math.pi)]
    return wps


def path_length(wps) -> float:
    return sum(math.dist(a[:2], b[:2]) for a, b in zip(wps, wps[1:]))
