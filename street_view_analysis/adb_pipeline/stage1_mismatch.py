"""Stage 1 — detect speed mismatches and draw 1000 random samples.

Mismatch definition (per the project decision): compare each valid segment's
``MedianSpeed`` against its posted ``SpeedLimit``.

    delta = MedianSpeed - SpeedLimit
        delta > +threshold  -> "over"   (traffic faster than the limit)
        delta < -threshold  -> "under"  (traffic slower than the limit)
        otherwise           -> "match"  (excluded from the sample)

We then draw N samples balanced across the two mismatch directions so both the
"higher" and "lower" cases are well represented downstream.
"""

from __future__ import annotations

import json
import random

import pandas as pd

from . import config
from .geo import heading_at_midpoint, representative_point

# Attributes carried through from each GeoJSON feature.
_KEEP_FIELDS = [
    "OBJECTID", "OvertureID", "english_ro", "RoadClass", "ProvinceID",
    "SpeedLimit", "SpeedLimitFloor", "MedianSpeed", "F85thPercentileSpeed",
    "PercentOverLimit", "NumberOverLimit", "RoadLength", "SampleSizeTotal",
    "LandUse", "AnalysisStatus",
]


def load_valid_features(path=config.GEOJSON_PATH) -> list[dict]:
    """Load features that are usable for the analysis."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    valid = []
    for feat in data["features"]:
        p = feat["properties"]
        if (
            p.get("AnalysisStatus") == "Valid"
            and p.get("SpeedLimit") is not None
            and p.get("F85thPercentileSpeed") is not None
        ):
            valid.append(feat)
    return valid


def _classify(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "over"
    if delta < -threshold:
        return "under"
    return "match"


def build_mismatch_table(features: list[dict]) -> pd.DataFrame:
    """Return a DataFrame of all valid segments with mismatch metrics + geometry."""
    rows = []
    for feat in features:
        p = feat["properties"]
        coords = feat["geometry"]["coordinates"]
        # GeoJSON may nest MultiLineString-style; normalise to a flat coord list.
        if coords and isinstance(coords[0][0], list):
            coords = [pt for part in coords for pt in part]
        # Over/under judged on the 85th-percentile (operating) speed, not median.
        delta = float(p["F85thPercentileSpeed"]) - float(p["SpeedLimit"])
        lon, lat = representative_point(coords)
        row = {k: p.get(k) for k in _KEEP_FIELDS}
        row.update(
            speed_delta=round(delta, 2),
            abs_delta=round(abs(delta), 2),
            mismatch_direction=_classify(delta, config.MISMATCH_THRESHOLD_KMH),
            pct_over_limit=(float(delta) / float(p["SpeedLimit"]) * 100.0)
            if p["SpeedLimit"]
            else None,
            lon=round(lon, 7),
            lat=round(lat, 7),
            heading=round(heading_at_midpoint(coords), 1),
            geometry=json.dumps(feat["geometry"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def sample_mismatches(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balanced random sample of `n` mismatched segments across over/under."""
    rng = random.Random(seed)
    over = df[df.mismatch_direction == "over"]
    under = df[df.mismatch_direction == "under"]

    half = n // 2
    n_over = min(half, len(over))
    n_under = min(n - n_over, len(under))
    # If one bucket is short, backfill from the other.
    if n_over + n_under < n:
        n_over = min(len(over), n - n_under)

    picks = []
    if n_over:
        picks.append(over.sample(n=n_over, random_state=seed))
    if n_under:
        picks.append(under.sample(n=n_under, random_state=seed + 1))
    sampled = pd.concat(picks).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "sample_id", [f"S{ i:04d}" for i in range(len(sampled))])
    return sampled


def _to_geojson(df: pd.DataFrame) -> dict:
    feats = []
    for _, r in df.iterrows():
        props = r.drop(labels=["geometry"]).to_dict()
        feats.append(
            {"type": "Feature", "geometry": json.loads(r["geometry"]), "properties": props}
        )
    return {"type": "FeatureCollection", "features": feats}


def run() -> pd.DataFrame:
    """Execute Stage 1 and write artifacts. Returns the sampled DataFrame."""
    config.ensure_dirs()
    print(f"[stage1] loading {config.GEOJSON_PATH.name} ...")
    features = load_valid_features()
    print(f"[stage1] valid segments: {len(features)}")

    df = build_mismatch_table(features)
    counts = df.mismatch_direction.value_counts().to_dict()
    print(f"[stage1] mismatch breakdown (threshold "
          f"{config.MISMATCH_THRESHOLD_KMH} km/h): {counts}")

    mismatched = df[df.mismatch_direction != "match"]
    n = min(config.N_SAMPLES, len(mismatched))
    if n < config.N_SAMPLES:
        print(f"[stage1] WARNING: only {len(mismatched)} mismatched segments; "
              f"sampling {n} instead of {config.N_SAMPLES}.")
    sampled = sample_mismatches(mismatched, n, config.RANDOM_SEED)
    print(f"[stage1] sampled {len(sampled)} "
          f"({(sampled.mismatch_direction == 'over').sum()} over / "
          f"{(sampled.mismatch_direction == 'under').sum()} under)")

    # Write CSV (drop bulky geometry) and a geometry-preserving GeoJSON.
    sampled.drop(columns=["geometry"]).to_csv(config.SAMPLES_CSV, index=False)
    with open(config.SAMPLES_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(_to_geojson(sampled), f)
    print(f"[stage1] wrote {config.SAMPLES_CSV}")
    print(f"[stage1] wrote {config.SAMPLES_GEOJSON}")
    return sampled


if __name__ == "__main__":
    run()
