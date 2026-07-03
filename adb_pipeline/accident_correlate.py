"""Correlate detected objects/scene with REAL 2024 crashes.

Same idea as Stage 4 (object ↔ speed mismatch), but the outcome is now the
accident record instead of the speed delta. For each sampled segment we attach
the crashes joined to it (by OBJECTID, from ``accidents.py``), then for every
detected object and scene class we report, across the sampled segments:

  * presence counts;
  * mean crashes per segment when the class is present vs absent (and the gap);
  * the same split by cause (speeding / cutting-in) and for fatalities;
  * the share of segments that had ANY crash, present vs absent;
  * point-biserial correlation of presence with the crash count.

Outputs a ranked CSV (all classes) and a Markdown summary. These are
observational associations, not causal claims.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config
from .stage4_correlate import _all_labels, _point_biserial, _presence_matrix

# accident metrics merged from segment_accidents.csv (0 where a segment had none)
_ACC_COLS = ["n_crashes", "n_speeding", "n_cutting", "n_fatalities"]


def _attach_accidents(samples: pd.DataFrame) -> pd.DataFrame:
    """Merge per-segment crash counts (by OBJECTID) onto the sampled segments."""
    if not config.SEGMENT_ACCIDENTS_CSV.exists():
        raise FileNotFoundError(
            f"{config.SEGMENT_ACCIDENTS_CSV} missing — run `adb-pipeline accidents` first.")
    acc = pd.read_csv(config.SEGMENT_ACCIDENTS_CSV)
    df = samples.merge(acc[["OBJECTID"] + _ACC_COLS], on="OBJECTID", how="left")
    for c in _ACC_COLS:
        df[c] = df[c].fillna(0).astype(int)
    df["had_crash"] = (df["n_crashes"] > 0).astype(int)
    return df


def run(detections: dict | None = None, samples: pd.DataFrame | None = None) -> pd.DataFrame:
    config.ensure_dirs()
    if detections is None:
        with open(config.DETECTIONS_JSON, "r", encoding="utf-8") as f:
            detections = json.load(f)
    if samples is None:
        samples = pd.read_csv(config.SAMPLES_CSV)
    if "OBJECTID" not in samples.columns:
        print("[accident-corr] samples have no OBJECTID; cannot join crashes.")
        return pd.DataFrame()

    classes = _all_labels(detections)
    if not classes:
        print("[accident-corr] no detections to correlate.")
        return pd.DataFrame()
    counts = _presence_matrix(detections, classes)

    df = _attach_accidents(samples).merge(counts, on="sample_id", how="inner")
    if df.empty:
        print("[accident-corr] no overlap between samples and detections.")
        return pd.DataFrame()

    # Scene "stuff" (barriers, buildings...) the object detector cannot see.
    feature_kind = {c: "object" for c in classes}
    if config.LANES_CSV.exists():
        lanes = pd.read_csv(config.LANES_CSV)
        frac_cols = [c for c in lanes.columns if c.endswith("_frac")
                     and c not in ("road_frac", "lane_marking_frac")]
        df = df.merge(lanes[["sample_id"] + frac_cols], on="sample_id", how="left")
        for c in frac_cols:
            name = "scene:" + c.replace("_frac", "")
            df[name] = (df[c].fillna(0) > 0.01).astype(int)
            classes = classes + [name]
            feature_kind[name] = "scene"

    crashes = df["n_crashes"].to_numpy(dtype=float)
    df.to_csv(config.CORRELATION_DIR / "samples_with_accidents.csv", index=False)

    rows = []
    for cls in classes:
        present = (df[cls] > 0).to_numpy()
        n_present = int(present.sum())
        if n_present < config.CORR_MIN_SUPPORT:
            continue
        n_absent = int((~present).sum())

        def gap(col):
            p = df.loc[present, col].mean()
            a = df.loc[~present, col].mean() if n_absent else float("nan")
            return round(float(p), 3), round(float(a), 3), round(float(p - a), 3)

        cr_p, cr_a, cr_g = gap("n_crashes")
        fa_p, fa_a, fa_g = gap("n_fatalities")
        sp_p, sp_a, sp_g = gap("n_speeding")
        cu_p, cu_a, cu_g = gap("n_cutting")
        any_p = round(float(df.loc[present, "had_crash"].mean()), 3)
        any_a = round(float(df.loc[~present, "had_crash"].mean()), 3) if n_absent else float("nan")
        rows.append({
            "object": cls, "kind": feature_kind[cls],
            "n_present": n_present, "n_absent": n_absent,
            "mean_crashes_present": cr_p, "mean_crashes_absent": cr_a, "crash_gap": cr_g,
            "mean_speeding_present": sp_p, "speeding_gap": sp_g,
            "mean_cutting_present": cu_p, "cutting_gap": cu_g,
            "mean_fatalities_present": fa_p, "fatality_gap": fa_g,
            "any_crash_rate_present": any_p, "any_crash_rate_absent": any_a,
            "any_crash_rate_gap": round(any_p - any_a, 3) if n_absent else float("nan"),
            "point_biserial_r": round(_point_biserial(present, crashes), 3),
        })

    report = pd.DataFrame(rows).sort_values("crash_gap", ascending=False, key=abs)
    out = config.CORRELATION_DIR / "object_accident_correlation.csv"
    report.to_csv(out, index=False)
    _write_markdown(df, report)
    print(f"[accident-corr] {len(df)} samples, "
          f"{int(df['had_crash'].sum())} with >=1 crash; "
          f"correlated {len(report)} classes -> {out}")
    return report


def _table(rows: pd.DataFrame) -> list[str]:
    out = ["| class | n present | crashes present | crashes absent | gap | speeding gap | cutting gap | r |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in rows.iterrows():
        out.append(f"| {r['object']} | {r['n_present']} | {r['mean_crashes_present']} | "
                   f"{r['mean_crashes_absent']} | {r['crash_gap']} | {r['speeding_gap']} | "
                   f"{r['cutting_gap']} | {r['point_biserial_r']} |")
    return out


def _write_markdown(df: pd.DataFrame, report: pd.DataFrame) -> None:
    solid = report[(report["n_absent"] > 0) & (report["n_present"] >= max(config.CORR_MIN_SUPPORT, 10))]
    lines = [
        "# Street-View Object ↔ 2024 Crash Correlation", "",
        f"- Sampled segments analysed: **{len(df)}** "
        f"({int(df['had_crash'].sum())} had ≥1 crash within 120 m)",
        f"- Total crashes on these segments: **{int(df['n_crashes'].sum())}** "
        f"(speeding {int(df['n_speeding'].sum())}, cutting-in {int(df['n_cutting'].sum())}, "
        f"fatalities {int(df['n_fatalities'].sum())})",
        f"- Distinct classes detected: **{len(report)}** (showing those present in ≥10 segments)",
        "",
        "## Classes most associated with *more* crashes (largest positive crash gap)", "",
        *_table(solid.sort_values("crash_gap", ascending=False).head(15)), "",
        "## Classes most associated with *fewer* crashes (largest negative crash gap)", "",
        *_table(solid.sort_values("crash_gap", ascending=True).head(15)), "",
        "_`crash gap` = mean crashes/segment when the class is present − when absent. "
        "Positive ⇒ the class co-occurs with more crashes. `r` is the point-biserial "
        "correlation of class presence with the crash count. Full results for every "
        "class are in `object_accident_correlation.csv`. Observational associations, "
        "not causal claims._",
    ]
    (config.CORRELATION_DIR / "accident_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
