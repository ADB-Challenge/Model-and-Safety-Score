"""Join the 2024 Thailand accident records to road segments by location.

The GeoJSON network has no Route ID/KM key that lines up with the crash file, so
each crash point is attached to the NEAREST road segment within a capped radius
(``ACCIDENT_MATCH_RADIUS_M``). Crashes with no segment inside the radius stay
unmatched and are reported honestly rather than forced onto a far-away road.

Only two crash causes are kept — "Speeding" and "Pedestrian/vehicle/animal
cutting in front suddenly"; every other cause is dropped. Per segment we tally:
  * n_crashes        — total of the two kept causes
  * n_speeding       — cause == "Speeding"
  * n_cutting        — cause == "Pedestrian/vehicle/animal cutting in front..."
  * n_fatalities     — deaths on those crashes
Because each crash has exactly one cause, n_speeding + n_cutting == n_crashes.

Output: output/_pipeline/segment_accidents.csv  (one row per matched OBJECTID).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from . import config

_M_PER_DEG_LAT = 110_574.0  # ~constant


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _load_segments() -> tuple[list, list[int], dict]:
    """Return (LineStrings, OBJECTIDs, {objectid: road_length_km}) for analysable segments."""
    gj = json.load(open(config.GEOJSON_PATH, encoding="utf-8"))
    geoms, oids, length_km = [], [], {}
    for f in gj["features"]:
        p = f["properties"]
        if p.get("AnalysisStatus") != "Valid":
            continue
        if p.get("PercentOverLimit") is None or p.get("F85thPercentileSpeed") is None:
            continue
        coords = f["geometry"]["coordinates"]
        if not coords:
            continue
        if isinstance(coords[0][0], list):  # MultiLineString -> flatten
            coords = [pt for part in coords for pt in part]
        if len(coords) < 2:
            continue
        geoms.append(LineString([(c[0], c[1]) for c in coords]))
        oid = int(p["OBJECTID"])
        oids.append(oid)
        rl = p.get("RoadLength")
        length_km[oid] = float(rl) if rl else max(geoms[-1].length * _M_PER_DEG_LAT / 1000.0, 0.05)
    return geoms, oids, length_km


def _load_crashes() -> pd.DataFrame:
    df = pd.read_csv(config.ACCIDENT_CSV, low_memory=False)
    df = df[df["Latitude"].notna() & df["Longitude"].notna()].copy()
    df["lat"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df[df["lat"].notna() & df["lon"].notna()]
    # Keep ONLY the two requested causes; everything else is dropped entirely.
    cause = df["Presumed Cause of Accident"].astype(str).str.strip()
    df = df[cause.isin(config.ACCIDENT_KEEP_CAUSES)].copy()
    cause = df["Presumed Cause of Accident"].astype(str).str.strip()
    df["is_speeding"] = (cause == config.CAUSE_SPEEDING).astype(int)
    df["is_cutting"] = (cause == config.CAUSE_CUTTING).astype(int)
    df["n_fatal"] = _num(df["Number of fatalities (Deaths)"])
    return df


def run() -> pd.DataFrame:
    config.ensure_dirs()
    geoms, oids, length_km = _load_segments()
    print(f"[accidents] analysable segments: {len(geoms)}")
    crashes = _load_crashes()
    print(f"[accidents] crash records with coordinates: {len(crashes)}")

    tree = STRtree(geoms)
    # cosine factor so a degree of longitude is metres-comparable at this latitude
    coslat = math.cos(math.radians(float(crashes["lat"].mean())))

    matched_oid = np.full(len(crashes), -1, dtype=int)
    lats = crashes["lat"].to_numpy()
    lons = crashes["lon"].to_numpy()
    cap_deg = config.ACCIDENT_MATCH_RADIUS_M / _M_PER_DEG_LAT  # rough deg cap on lat axis
    for i in range(len(crashes)):
        pt = Point(lons[i], lats[i])
        j = int(tree.nearest(pt))
        line = geoms[j]
        np_pt = line.interpolate(line.project(pt))
        dlat = (np_pt.y - lats[i]) * _M_PER_DEG_LAT
        dlon = (np_pt.x - lons[i]) * _M_PER_DEG_LAT * coslat
        dist_m = math.hypot(dlat, dlon)
        if dist_m <= config.ACCIDENT_MATCH_RADIUS_M:
            matched_oid[i] = oids[j]
        _ = cap_deg  # (kept for readability of the radius logic)
    crashes["OBJECTID"] = matched_oid

    m = crashes[crashes["OBJECTID"] >= 0]
    print(f"[accidents] matched within {config.ACCIDENT_MATCH_RADIUS_M:.0f} m: "
          f"{len(m)} / {len(crashes)} ({100*len(m)/len(crashes):.1f}%)")

    # Each kept crash is exactly one of the two causes, so n_speeding + n_cutting
    # == n_crashes (the counts sum correctly).
    agg = m.groupby("OBJECTID").agg(
        n_crashes=("OBJECTID", "size"),
        n_speeding=("is_speeding", "sum"),
        n_cutting=("is_cutting", "sum"),
        n_fatalities=("n_fatal", "sum"),
    ).reset_index()
    for c in agg.columns:
        if c != "OBJECTID":
            agg[c] = agg[c].astype(int)
    agg["road_length_km"] = agg["OBJECTID"].map(length_km).round(3)

    agg.to_csv(config.SEGMENT_ACCIDENTS_CSV, index=False)
    print(f"[accidents] wrote {config.SEGMENT_ACCIDENTS_CSV}  "
          f"({len(agg)} segments with >=1 crash)")
    print(f"[accidents] kept causes only: crashes={int(agg.n_crashes.sum())} "
          f"(speeding={int(agg.n_speeding.sum())}, cutting-in={int(agg.n_cutting.sum())}); "
          f"check sum: {int(agg.n_speeding.sum()) + int(agg.n_cutting.sum())}")
    return agg


if __name__ == "__main__":
    run()
