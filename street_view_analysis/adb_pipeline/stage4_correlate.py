"""Stage 4 — correlate detected objects with the speed mismatch.

We join Stage-1 mismatch metrics with Stage-3 detections and ask: which
roadside objects are associated with traffic running *over* vs *under* the
posted limit?

Per the project decision, **every** raw object label produced by the detector is
treated as its own feature (no bucketing). For each object we report, across the
sampled segments:
  * presence counts;
  * mean speed delta (MedianSpeed - SpeedLimit) when present vs absent;
  * the gap between those means (positive => object linked to faster traffic);
  * point-biserial correlation of presence vs speed delta;
  * the share of "over" segments among present vs absent.

Outputs a ranked CSV (all objects), a per-sample joined CSV, and a Markdown
summary of the strongest associations.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from . import config


def _all_labels(detections: dict[str, list[dict]]) -> list[str]:
    """Sorted union of every raw label the detector produced."""
    labels = {d["label"] for dets in detections.values() for d in dets}
    return sorted(labels)


def _presence_matrix(detections: dict[str, list[dict]], classes: list[str]) -> pd.DataFrame:
    """One row per sample, one column per detected label = count of detections.

    Images with zero detections are kept as all-zero rows so they correctly form
    part of the "object absent" group.
    """
    mat = pd.DataFrame(0, index=list(detections.keys()), columns=classes, dtype=int)
    for sid, dets in detections.items():
        c = Counter(d["label"] for d in dets)
        for lbl, n in c.items():
            mat.at[sid, lbl] = n
    mat.index.name = "sample_id"
    return mat.reset_index()


def _point_biserial(present: np.ndarray, value: np.ndarray) -> float:
    if present.std() == 0 or value.std() == 0:
        return float("nan")
    return float(np.corrcoef(present.astype(float), value)[0, 1])


def run(detections: dict | None = None, samples: pd.DataFrame | None = None) -> pd.DataFrame:
    config.ensure_dirs()
    if detections is None:
        with open(config.DETECTIONS_JSON, "r", encoding="utf-8") as f:
            detections = json.load(f)
    if samples is None:
        samples = pd.read_csv(config.SAMPLES_CSV)

    classes = _all_labels(detections)
    if not classes:
        print("[stage4] no detections to correlate.")
        return pd.DataFrame()
    counts = _presence_matrix(detections, classes)

    df = samples.merge(counts, on="sample_id", how="inner")
    if df.empty:
        print("[stage4] no overlap between samples and detections; nothing to do.")
        return pd.DataFrame()

    # Bring in scene-segmentation "stuff" (barriers, grass, buildings...) that
    # object detection cannot see. Each coverage fraction becomes a presence flag.
    feature_kind = {c: "object" for c in classes}
    lanes_csv = config.LANES_CSV
    if lanes_csv.exists():
        lanes = pd.read_csv(lanes_csv)
        frac_cols = [c for c in lanes.columns if c.endswith("_frac")
                     and c not in ("road_frac", "lane_marking_frac")]
        df = df.merge(lanes[["sample_id"] + frac_cols], on="sample_id", how="left")
        for c in frac_cols:
            name = "scene:" + c.replace("_frac", "")
            df[name] = (df[c].fillna(0) > 0.01).astype(int)  # present if >=1% of frame
            classes = classes + [name]
            feature_kind[name] = "scene"

    df["is_over"] = (df["mismatch_direction"] == "over").astype(int)
    delta = df["speed_delta"].to_numpy(dtype=float)

    # Per-sample joined export (object + scene presence alongside mismatch metrics).
    df.to_csv(config.CORRELATION_DIR / "samples_with_objects.csv", index=False)

    rows = []
    for cls in classes:
        present_mask = (df[cls] > 0).to_numpy()
        n_present = int(present_mask.sum())
        if n_present < config.CORR_MIN_SUPPORT:
            continue
        n_absent = int((~present_mask).sum())
        mean_p = float(delta[present_mask].mean())
        mean_a = float(delta[~present_mask].mean()) if n_absent else float("nan")
        over_p = float(df.loc[present_mask, "is_over"].mean())
        over_a = float(df.loc[~present_mask, "is_over"].mean()) if n_absent else float("nan")
        rows.append(
            {
                "object": cls,
                "kind": feature_kind[cls],
                "n_present": n_present,
                "n_absent": n_absent,
                "mean_delta_present": round(mean_p, 2),
                "mean_delta_absent": round(mean_a, 2),
                "delta_gap": round(mean_p - mean_a, 2),
                "over_rate_present": round(over_p, 3),
                "over_rate_absent": round(over_a, 3),
                "over_rate_gap": round(over_p - over_a, 3),
                "point_biserial_r": round(_point_biserial(present_mask, delta), 3),
            }
        )

    report = pd.DataFrame(rows).sort_values("delta_gap", ascending=False, key=abs)
    report.to_csv(config.CORRELATION_DIR / "object_speed_correlation.csv", index=False)

    _write_markdown(df, report)
    print(f"[stage4] correlated {len(report)} object types "
          f"(support >= {config.CORR_MIN_SUPPORT}) -> {config.CORRELATION_DIR}")
    return report


def _table(rows: pd.DataFrame) -> list[str]:
    out = [
        "| object | n present | Δ present | Δ absent | gap (km/h) | over-rate gap | r |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in rows.iterrows():
        out.append(
            f"| {r['object']} | {r['n_present']} | {r['mean_delta_present']} | "
            f"{r['mean_delta_absent']} | {r['delta_gap']} | {r['over_rate_gap']} | "
            f"{r['point_biserial_r']} |"
        )
    return out


def _write_markdown(df: pd.DataFrame, report: pd.DataFrame) -> None:
    n = len(df)
    n_over = int(df["is_over"].sum())
    # Focus the headline tables on objects with enough support to be meaningful.
    solid = report[report["n_absent"] > 0]
    solid = solid[solid["n_present"] >= max(config.CORR_MIN_SUPPORT, 10)]
    lines = [
        "# Speed-Mismatch ↔ Street-View Object Correlation",
        "",
        f"- Samples analysed: **{n}** ({n_over} over-limit, {n - n_over} under-limit)",
        f"- Mean speed delta (Median − Limit): **{df['speed_delta'].mean():.2f} km/h**",
        f"- Distinct object types detected: **{len(report)}** "
        f"(showing those present in ≥10 segments below)",
        "",
        "## Objects most associated with *faster* traffic (largest positive delta gap)",
        "",
        *_table(solid.sort_values("delta_gap", ascending=False).head(15)),
        "",
        "## Objects most associated with *slower* traffic (largest negative delta gap)",
        "",
        *_table(solid.sort_values("delta_gap", ascending=True).head(15)),
        "",
        "_Interpretation: `delta gap` = mean(MedianSpeed−SpeedLimit | object present) "
        "− mean(… | object absent). Positive ⇒ the object co-occurs with traffic "
        "exceeding the limit; negative ⇒ co-occurs with slower-than-limit traffic. "
        "`r` is the point-biserial correlation of object presence with the speed "
        "delta. Full results for every detected object are in "
        "`object_speed_correlation.csv`. These are observational associations, not "
        "causal claims._",
    ]
    (config.CORRELATION_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
