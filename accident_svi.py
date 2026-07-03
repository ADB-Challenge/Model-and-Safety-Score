"""Accident-point Street View experiment.

New sampling: keep road segments with >=2 (speeding/cutting) accidents, sample
1000 of them, and for each pick the 2 accident points FARTHEST APART along the
segment. Street View is fetched AT each accident coordinate (not the segment
midpoint), then YOLO + Mask2Former run on the 2000 images.

Outputs to output/_accident_svi/ (separate from the main pipeline).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.strtree import STRtree

from adb_pipeline import config
from adb_pipeline.accidents import _load_crashes, _load_segments
from adb_pipeline.accidents import _M_PER_DEG_LAT as M_PER_DEG_LAT

# --- redirect outputs BEFORE importing the imagery stages ---
EXP = config.OUTPUT_DIR / "_accident_svi"
config.IMAGES_DIR = EXP / "images"
config.LABELLED_DIR = EXP / "labelled"
config.GSV_METADATA_CSV = EXP / "gsv_metadata.csv"
config.DETECTIONS_JSON = EXP / "detections.json"
config.SAMPLES_CSV = EXP / "samples.csv"
for d in (EXP, config.IMAGES_DIR, config.LABELLED_DIR):
    d.mkdir(parents=True, exist_ok=True)

N_SEG = 1000


def _hav(la1, lo1, la2, lo2):
    k = math.cos(math.radians((la1 + la2) / 2))
    return math.hypot((la2 - la1) * M_PER_DEG_LAT, (lo2 - lo1) * M_PER_DEG_LAT * k)


def build_points() -> pd.DataFrame:
    geoms, oids, _ = _load_segments()
    geom_by_oid = dict(zip(oids, geoms))
    crashes = _load_crashes()   # already filtered to speeding + cutting-in
    gj = json.load(open(config.GEOJSON_PATH, encoding="utf-8"))
    props = {int(f["properties"]["OBJECTID"]): f["properties"]
             for f in gj["features"] if f["properties"].get("OBJECTID") is not None}

    # nearest-segment join, keeping each individual accident
    tree = STRtree(geoms)
    coslat = math.cos(math.radians(float(crashes["lat"].mean())))
    lats, lons = crashes["lat"].to_numpy(), crashes["lon"].to_numpy()
    matched = np.full(len(crashes), -1, dtype=int)
    for i in range(len(crashes)):
        pt = Point(lons[i], lats[i])
        j = int(tree.nearest(pt))
        line = geoms[j]
        npt = line.interpolate(line.project(pt))
        d = math.hypot((npt.y - lats[i]) * M_PER_DEG_LAT,
                       (npt.x - lons[i]) * M_PER_DEG_LAT * coslat)
        if d <= config.ACCIDENT_MATCH_RADIUS_M:
            matched[i] = oids[j]
    crashes = crashes.assign(OBJECTID=matched)
    m = crashes[crashes["OBJECTID"] >= 0].reset_index(drop=True)

    counts = m.groupby("OBJECTID").size()
    eligible = counts[counts >= 2].index.to_numpy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    picks = rng.choice(eligible, size=min(N_SEG, len(eligible)), replace=False)

    rows, pid = [], 0
    for oid in picks:
        sub = m[m["OBJECTID"] == oid].reset_index(drop=True)
        la, lo = sub["lat"].to_numpy(), sub["lon"].to_numpy()
        # 2 farthest-apart points
        if len(sub) == 2:
            ia, ib = 0, 1
        else:
            best, ia, ib = -1.0, 0, 1
            for a in range(len(sub)):
                for b in range(a + 1, len(sub)):
                    dd = _hav(la[a], lo[a], la[b], lo[b])
                    if dd > best:
                        best, ia, ib = dd, a, b
        line = geom_by_oid[int(oid)]
        p = props.get(int(oid), {})
        f85, sl = p.get("F85thPercentileSpeed"), p.get("SpeedLimit")
        for idx in (ia, ib):
            r = sub.loc[idx]
            pt = Point(r["lon"], r["lat"])
            dpos, L = line.project(pt), line.length
            q1 = line.interpolate(max(dpos - L * 0.01, 0.0))
            q2 = line.interpolate(min(dpos + L * 0.01, L))
            heading = math.degrees(math.atan2(q2.x - q1.x, q2.y - q1.y)) % 360
            rows.append({
                "sample_id": f"A{pid:04d}", "OBJECTID": int(oid),
                "n_accidents_segment": int(counts[int(oid)]),
                "point_in_segment": 1 if idx == ia else 2,
                "lat": round(float(r["lat"]), 7), "lon": round(float(r["lon"]), 7),
                "heading": round(heading, 1),
                "cause": "speeding" if int(r["is_speeding"]) else "cutting-in",
                "n_fatal": int(r["n_fatal"]),
                "SpeedLimit": sl, "RoadClass": p.get("RoadClass"),
                "F85thPercentileSpeed": f85, "MedianSpeed": p.get("MedianSpeed"),
                "PercentOverLimit": p.get("PercentOverLimit"), "LandUse": p.get("LandUse"),
                "mismatch_direction": "over" if (f85 or 0) - (sl or 0) > 0 else "under",
            })
            pid += 1
    return pd.DataFrame(rows)


def main(fetch: bool = True) -> None:
    df = build_points()
    df.to_csv(config.SAMPLES_CSV, index=False)
    print(f"[acc-svi] segments sampled: {df['OBJECTID'].nunique()} | accident points: {len(df)}")
    print(df["cause"].value_counts().to_string())
    print(f"[acc-svi] wrote {config.SAMPLES_CSV}")
    if not fetch:
        return
    from adb_pipeline import stage2_gsv, stage3_detect, labelled
    labelled.LABELLED_DIR = config.LABELLED_DIR
    fetch_log = stage2_gsv.run(df)
    detections = stage3_detect.run(fetch_log)
    labelled.run(detections, df)
    print(f"\n[acc-svi] DONE. images: {config.IMAGES_DIR} | labelled: {config.LABELLED_DIR}")


if __name__ == "__main__":
    import sys
    main(fetch="--sample-only" not in sys.argv)
