"""Geometry helpers for road-segment LineStrings."""

from __future__ import annotations

from shapely.geometry import LineString


def representative_point(coords: list[list[float]]) -> tuple[float, float]:
    """Return a (lon, lat) point representative of the segment.

    Uses the point halfway along the line by arc length (more meaningful than a
    raw coordinate average for curved/uneven segments). Coordinates are GeoJSON
    order: [lon, lat].
    """
    if not coords:
        raise ValueError("empty coordinate list")
    if len(coords) == 1:
        lon, lat = coords[0][0], coords[0][1]
        return lon, lat
    line = LineString([(c[0], c[1]) for c in coords])
    mid = line.interpolate(0.5, normalized=True)
    return mid.x, mid.y


def heading_at_midpoint(coords: list[list[float]]) -> float:
    """Approximate compass heading (degrees, 0=N, 90=E) of travel at the
    midpoint, so the Street View camera can be aimed along the road.
    """
    import math

    if len(coords) < 2:
        return 0.0
    line = LineString([(c[0], c[1]) for c in coords])
    half = line.length / 2.0
    # sample two close points straddling the midpoint
    p1 = line.interpolate(max(half - line.length * 0.01, 0.0))
    p2 = line.interpolate(min(half + line.length * 0.01, line.length))
    dlon = p2.x - p1.x
    dlat = p2.y - p1.y
    # bearing from north, clockwise
    bearing = math.degrees(math.atan2(dlon, dlat)) % 360
    return bearing
