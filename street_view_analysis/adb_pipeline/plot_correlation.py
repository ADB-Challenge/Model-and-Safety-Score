"""The speed / objects / accidents correlation bubble chart.

Each detected class (YOLO-World object, Mapillary scene class) plus lane-width
groups is placed by its relationship to ACCIDENTS (x = crash_gap) and to SPEED
(y = delta_gap). Upper-right = goes with both more crashes AND faster traffic
(the dangerous combination). Saved to output/charts/speed_objects_accidents.png.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config

OBJ_COLOR = "#3b75af"    # YOLO-World objects
SCENE_COLOR = "#e1812c"  # Mapillary scene classes
LANE_COLOR = SCENE_COLOR  # orange, matching the Mapillary scene bubbles


def _short(name: str) -> str:
    if name.startswith("scene:"):
        return name.replace("scene:", "").replace("_", " ")
    return name.split("/")[0]


def _lane_bubbles() -> pd.DataFrame:
    """Lane width as two bubbles (narrow <=2 vs wide 3+), positioned the same way
    as the object/scene bubbles: speed gap and crash gap vs the other lane group."""
    try:
        s = pd.read_csv(config.SAMPLES_CSV)
        ln = pd.read_csv(config.LANES_CSV)[["sample_id", "est_lane_count"]]
        df = s[["sample_id", "OBJECTID", "speed_delta"]].merge(ln, on="sample_id", how="inner")
        df = df.dropna(subset=["est_lane_count"])
        if config.SEGMENT_ACCIDENTS_CSV.exists():
            ac = pd.read_csv(config.SEGMENT_ACCIDENTS_CSV)[["OBJECTID", "n_crashes"]]
            df = df.merge(ac, on="OBJECTID", how="left")
        df["n_crashes"] = df.get("n_crashes", pd.Series(0, index=df.index)).fillna(0)
        narrow = df["est_lane_count"] <= 2
        wide = ~narrow
        if int(narrow.sum()) < 10 or int(wide.sum()) < 10:
            return pd.DataFrame()
        d_gap = df.loc[narrow, "speed_delta"].mean() - df.loc[wide, "speed_delta"].mean()
        c_gap = df.loc[narrow, "n_crashes"].mean() - df.loc[wide, "n_crashes"].mean()
        return pd.DataFrame([
            {"object": "narrow lanes (<=2)", "kind": "lane", "n_present": int(narrow.sum()),
             "delta_gap": round(float(d_gap), 2), "crash_gap": round(float(c_gap), 3)},
            {"object": "wide lanes (3+)", "kind": "lane", "n_present": int(wide.sum()),
             "delta_gap": round(float(-d_gap), 2), "crash_gap": round(float(-c_gap), 3)},
        ])
    except Exception as e:
        print(f"[plot] lane bubbles skipped: {e}")
        return pd.DataFrame()


def plot_speed_objects_accidents(min_support: int = 15, label_n: int = 28) -> None:
    """Combined view: each detected class positioned by its relationship to SPEED
    (y = delta_gap) and to ACCIDENTS (x = crash_gap). Upper-right = goes with both
    faster traffic AND more crashes (the dangerous combination). Lane width (narrow
    vs wide) is added as two extra bubbles. Every bubble is labelled.
    """
    sp = pd.read_csv(config.CORRELATION_DIR / "object_speed_correlation.csv")
    ac = pd.read_csv(config.CORRELATION_DIR / "object_accident_correlation.csv")
    m = sp[["object", "kind", "n_present", "delta_gap"]].merge(
        ac[["object", "kind", "crash_gap"]], on=["object", "kind"], how="inner")
    m = m[m["n_present"] >= min_support].copy()
    m = pd.concat([m, _lane_bubbles()], ignore_index=True)
    if m.empty:
        return

    colors = m["kind"].map({"object": OBJ_COLOR, "scene": SCENE_COLOR,
                            "lane": LANE_COLOR}).fillna(OBJ_COLOR)
    sizes = 30 + 4.0 * m["n_present"]

    # x = ACCIDENT association (crash_gap), y = SPEED association (delta_gap)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.scatter(m["crash_gap"], m["delta_gap"], s=sizes, c=colors,
               alpha=0.7, edgecolor="black", linewidth=0.4)

    # label EVERY bubble; adjustText then pushes the labels apart so none overlap.
    texts = [ax.text(r["crash_gap"], r["delta_gap"], _short(r["object"]),
                     fontsize=7.5, color="black",
                     fontweight=("bold" if r["kind"] == "lane" else "normal"))
             for _, r in m.iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, expand=(1.2, 1.5), force_text=(0.4, 0.6),
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.5))
    except Exception as e:
        print(f"[plot] adjustText unavailable ({e}); labels may overlap")

    xmax = m["crash_gap"].abs().max() * 1.15
    ymax = m["delta_gap"].abs().max() * 1.15
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    q = dict(fontsize=9, color="#555", ha="center", va="center", style="italic")
    ax.text(xmax * 0.6, ymax * 0.92, "MORE crashes +\nfaster traffic", **q)
    ax.text(-xmax * 0.6, ymax * 0.92, "fewer crashes +\nfaster traffic", **q)
    ax.text(xmax * 0.6, -ymax * 0.92, "MORE crashes +\nslower traffic", **q)
    ax.text(-xmax * 0.6, -ymax * 0.92, "fewer crashes +\nslower traffic", **q)

    ax.set_xlabel("Accident association  (crashes per segment)", fontsize=10)
    ax.set_ylabel("Speed association  (km/h vs limit;  under-limit / over-limit)", fontsize=10)
    ax.set_title("Speed, Objects and Accident Correlation", fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=OBJ_COLOR, label="YOLO-World object"),
                       Patch(color=SCENE_COLOR, label="Mapillary scene class"),
                       Patch(color=LANE_COLOR, label="Lane width (narrow/wide)")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9,
              borderaxespad=0, frameon=True)
    ax.grid(ls=":", alpha=0.35)
    fig.tight_layout()
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.CHARTS_DIR / "speed_objects_accidents.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out}")


def run() -> None:
    plot_speed_objects_accidents()


if __name__ == "__main__":
    run()
