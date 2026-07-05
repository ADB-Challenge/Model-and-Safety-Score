"""Stage 1 (safety-framework variant) — sample road segments STRATIFIED BY POSTED
SPEED-LIMIT CATEGORY, observe over/under compliance, and compute a Speed Safety
Score per segment.

Unlike the mismatch sampler, this does NOT pre-balance over/under. It draws
segments *randomly within each speed-limit category* (50/60/70/80/90 kph), so the
observed share running over the limit is the real compliance rate for that
category — which is what the safety scoring needs.

Speed Safety Score (0-100, higher = less safe) blends:
  * prevalence — PercentOverLimit (share of vehicles exceeding the limit), and
  * severity   — (F85thPercentileSpeed - SpeedLimit), how far the fast cohort
                 exceeds, capped at SAFETY_SEV_CAP km/h.
"""

from __future__ import annotations

import json

import pandas as pd

from . import config
from .geo import heading_at_midpoint, representative_point
from .stage1_mismatch import _KEEP_FIELDS, load_valid_features


def speed_category(limit: float) -> int | None:
    """Bin a posted limit to the nearest 10 and keep only the framework categories."""
    b = int(round(float(limit) / 10.0) * 10)
    return b if b in config.SPEED_CATEGORIES else None


def safety_score(percent_over, f85, limit) -> float:
    """0-100, higher = less safe (more & faster over-limit driving)."""
    prev = min(max(float(percent_over or 0.0), 0.0), 1.0)
    sev = max(0.0, float(f85) - float(limit)) / config.SAFETY_SEV_CAP
    sev = min(sev, 1.0)
    return round(100.0 * (config.SAFETY_W_PREV * prev + config.SAFETY_W_SEV * sev), 1)


def _grade(score: float) -> str:
    return ("A (safe)" if score < 20 else "B" if score < 40 else "C" if score < 60
            else "D" if score < 80 else "E (unsafe)")


def build_table() -> pd.DataFrame:
    feats = load_valid_features()
    rows = []
    for feat in feats:
        p = feat["properties"]
        if p.get("F85thPercentileSpeed") is None or p.get("PercentOverLimit") is None:
            continue
        cat = speed_category(p["SpeedLimit"])
        if cat is None:
            continue
        coords = feat["geometry"]["coordinates"]
        if coords and isinstance(coords[0][0], list):
            coords = [pt for part in coords for pt in part]
        lon, lat = representative_point(coords)
        # Over/under is judged on the 85th-percentile (operating) speed vs the
        # posted limit — the standard speed-limit-setting reference — not the median.
        delta = float(p["F85thPercentileSpeed"]) - float(p["SpeedLimit"])
        score = safety_score(p["PercentOverLimit"], p["F85thPercentileSpeed"], p["SpeedLimit"])
        row = {k: p.get(k) for k in _KEEP_FIELDS}
        row.update(
            speed_category=cat,
            speed_delta=round(delta, 2),
            mismatch_direction="over" if delta > 0 else "under",
            is_over=int(delta > 0),
            exceedance_85=round(float(p["F85thPercentileSpeed"]) - float(p["SpeedLimit"]), 1),
            safety_score=score,
            safety_grade=_grade(score),
            lon=round(lon, 7),
            lat=round(lat, 7),
            heading=round(heading_at_midpoint(coords), 1),
            geometry=json.dumps(feat["geometry"]),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _to_geojson(df: pd.DataFrame) -> dict:
    feats = []
    for _, r in df.iterrows():
        props = r.drop(labels=["geometry"]).to_dict()
        feats.append({"type": "Feature", "geometry": json.loads(r["geometry"]),
                      "properties": props})
    return {"type": "FeatureCollection", "features": feats}


def run() -> pd.DataFrame:
    config.ensure_dirs()
    print("[strata] loading valid segments ...")
    df = build_table()
    print(f"[strata] usable segments (with compliance data, limit in categories): {len(df)}")

    picks = []
    for cat in config.SPEED_CATEGORIES:
        sub = df[df.speed_category == cat]
        n = min(config.CATEGORY_CAP, len(sub))
        if n > 0:
            picks.append(sub.sample(n=n, random_state=config.RANDOM_SEED + cat))
        print(f"[strata]   {cat} kph: {len(sub)} available -> sampling {n}")
    sampled = pd.concat(picks).sample(frac=1.0, random_state=config.RANDOM_SEED).reset_index(drop=True)
    sampled.insert(0, "sample_id", [f"S{i:04d}" for i in range(len(sampled))])

    print(f"[strata] total sampled: {len(sampled)}")
    print("[strata] observed over-limit rate by category:")
    print(sampled.groupby("speed_category")["is_over"].mean().round(3).to_string())

    sampled.drop(columns=["geometry"]).to_csv(config.SAMPLES_CSV, index=False)
    with open(config.SAMPLES_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(_to_geojson(sampled), f)
    print(f"[strata] wrote {config.SAMPLES_CSV}")
    return sampled


if __name__ == "__main__":
    run()
