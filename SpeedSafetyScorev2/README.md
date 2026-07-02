# Road Safety Scoring Pipeline

This document explains what `ai for safer roads.ipynb` does, why each modelling
choice was made, and what the pipeline actually produced on the sample dataset
(2,990 training road segments + 11,544 candidate "unmerged" segments).

## Data Requirement

This code requires the following data:
1. ADB Dataset.
2. Public Thailand Accident Dataset (provided)

---

## 1. What problem this solves

Each row is a road segment with speed, traffic-volume, road-class, land-use and
geometry attributes, plus two observed outcome counts:

- `total_accidents` — historical accident count on that segment
- `death` — historical fatality count on that segment

The goal is a **single risk score per road segment** that can be used to
prioritise which roads get attention first — for both segments already in the
training network (`df_clean`) and brand-new segments not yet merged into it
(`df_clean_unmerged`).

Both target counts are **over-dispersed** (variance ≫ mean) and **spatially
correlated** — a dangerous corridor tends to have several dangerous segments
in a row, not one isolated one. That combination is why the pipeline uses
Negative Binomial regression (not plain Poisson) and explicitly tests for,
and corrects for, spatial autocorrelation before finalising a score.

---

## 2. Model selection (why B, B_INT, D, D+W and not others)

Before settling on the four production models, five candidates were compared
on the training data (`df_clean`, n=2,990) via cross-validation (Leave-One-Out
for small folds / stratified 10-fold otherwise). Lower MAE / deviance is
better; higher Spearman ρ and Top-5%-precision are better.

**Target: `total_accidents`** (0% zero-count rows, variance/mean = 30.2 — heavily over-dispersed)

| Model                        | CV MAE | CV Poisson Deviance | Spearman ρ | Top-5% precision |
|-------------------------------|-------:|---------------------:|-----------:|------------------:|
| A — NB, no exposure            | 2.723  | 4.185                | 0.184      | 0.247              |
| B — NB + log(volume)           | 2.293  | 2.866                | 0.426      | 0.393              |
| B_INT — NB + interactions      | 2.234  | 2.673                | 0.426      | 0.407              |
| D — GBT (Poisson loss)         | 2.181  | 2.817                | 0.408      | 0.387              |
| **D+W — GBT + spatial lag**    | **2.124** | **2.710**          | **0.425**  | **0.413**          |

**Target: `death`** (74.8% zero-count rows, variance/mean = 2.09 — moderately over-dispersed)

| Model                        | CV MAE | CV Poisson Deviance | Spearman ρ | Top-5% precision |
|-------------------------------|-------:|---------------------:|-----------:|------------------:|
| A — NB, no exposure            | 0.581  | 1.211                | 0.067      | 0.073              |
| B — NB + log(volume)           | 0.556  | 1.146                | 0.195      | 0.120              |
| B_INT — NB + interactions      | 0.556  | 1.149                | 0.185      | 0.140              |
| D — GBT (Poisson loss)         | 0.554  | 1.211                | 0.143      | 0.120              |
| **D+W — GBT + spatial lag**    | **0.552** | **1.207**          | 0.154      | 0.113              |

**Takeaways used to design the production pipeline:**

- **Model A (no traffic-volume term) is dominated on every metric** by every
  model that includes `log_volume` — traffic exposure matters a lot, so it's
  kept as a covariate in every model going forward, never as a fixed offset.
- **B_INT's interaction terms help accidents CV metrics modestly** (MAE
  2.293→2.234, Top-5% 0.393→0.407) but **don't help death** (MAE/deviance/ρ
  roughly flat, only Top-5% improves). It's kept anyway because it's the most
  *interpretable* model (named coefficients, p-values) and the merge step
  wants an interpretable score alongside the machine-learned one.
- **The tree model (D) alone slightly underperforms B/B_INT on deviance** but
  captures non-linear structure the NB models can't. It's kept as a
  complementary signal, not the primary score.
- **Adding the spatial-lag feature always helps GBT** (D→D+W improves MAE and
  deviance for both targets, and improves Spearman ρ for both targets) — this
  is the direct payoff of the spatial autocorrelation testing described below,
  and it's why **Model D+W is the primary score** the pipeline ranks roads by.

### Spatial autocorrelation check (why the "+W" step exists at all)

A topological adjacency matrix was built from road-segment endpoints
(segments that physically touch are "neighbours"). Moran's I on the residuals
of the non-spatial models confirmed the segments are *not* independent
observations:

| Target | Model B residuals | Model D residuals |
|---|---|---|
| `total_accidents` | I = 0.126, p = 0.0002 *** | I = 0.180, p < 0.0001 *** |
| `death`            | I = 0.070, p = 0.037 *  | I = 0.070, p = 0.038 *   |

Both are significant positive spatial autocorrelation → risk clusters
along corridors, not just on individual segments. This is the empirical
justification for feeding a spatially-lagged neighbour signal
(`W_log_target`, the row-standardised mean of a segment's neighbours'
`log1p(target)`) into the GBT as **Model D+W**, and for later reusing that
same spatial-lag mechanism when scoring brand-new (unmerged) segments in
Part 3 — so a new road can borrow risk signal from the known roads it
physically connects to, even though it has no accident/death history itself.

---

## 3. Pipeline stages

### Part 1 — Train scoring (`df_clean`)

For each target (`total_accidents`, `death`) independently:

1. Feature engineering: speed spread, winsorised `Speed_Diff_Pct`, rural flag,
   road-class dummies, `pct_over_limit`, `log_volume` (base features), plus
   speed × road-class interaction terms and continuous geometry extras
   (`abs_skew`, `log_n_segs`, `Percentile`) for the interpretable model.
2. Build a train-only network adjacency matrix `W_net_train` from segment
   endpoints.
3. Fit four models on the full training data and predict on the same data:
   - **Model B** — Negative Binomial, `log_volume` as a covariate
   - **Model B_INT** — Model B + interaction/continuous terms
   - **Model D** — Gradient-boosted trees, Poisson loss
   - **Model D+W** — Model D + spatially-lagged `log1p(target)` feature
4. Combine `pred_DW`, `pred_B_INT`, `pred_D` into a z-score **ensemble score**
   (Model B is excluded from the ensemble — it's kept only as a reference
   column, since B_INT strictly extends it).
5. Rank segments by `pred_DW` (Model D+W) into a `risk_percentile` /
   `risk_decile`.
6. Export `road_safety_scores_accident.csv` and `road_safety_scores_death.csv`.

**Illustrative output — highest-risk segments (`total_accidents`, Model D+W):**
top 10% skews toward `trunk` (37.5%) and `primary` (28.4%) roads, split almost
evenly between rural (50.8%) and urban (49.2%).

**Illustrative output — highest-risk segments (`death`):** top 10% skews even
more rural (74.2% vs. 25.8% urban) and toward `trunk`/`secondary`/`primary`
roads — `motorway` drops from 18.1% of the top decile (accidents) to just
6.7% (death), consistent with motorways having more, but less severe, crashes.

### Part 2 — Merge + risk classification

`export_df` (accidents) and `export_df_death` are merged on `OBJECTID`.
`final_risk_percentile` is the average of the two targets' `risk_percentile`,
and is bucketed into five bands:

| `final_risk_percentile` | Classification |
|---|---|
| < 20  | Stable |
| 20–39 | Aware |
| 40–59 | Monitor |
| 60–79 | Attention |
| ≥ 80  | Critical |

Geometry is attached from `gdf_merged` and the merged, classified table is
written to **`final_safety_risk.geojson`**.

### Part 3 — Score unmerged roads (`df_clean_unmerged`)

New roads have no accident/death history, so they can't be scored the same
way. Instead:

1. A **combined** adjacency matrix is built across **both** `df_clean`
   (training roads) and `df_clean_unmerged` (new roads), so a new road that
   physically connects to a known road can inherit spatial signal from it.
2. Model D+W is refit on the training data only, using the same feature set
   as Part 1, but with the spatial-lag feature now computed over the combined
   network.
3. The fitted model scores the new roads directly (`pred_new`); roads with no
   train-network neighbour fall back to the mean spatial-lag value.
4. Accident and death predictions are each converted to a risk percentile,
   averaged into `risk_percentile_combined`, and bucketed into deciles.
5. Exported to **`road_safety_scores_unmerged.csv`** and
   **`road_safety_scores_unmerged.geojson`**.

**Actual run output (11,544 unmerged candidate segments):**

```
Building combined network adjacency (train + new roads) ...
  [W] Parsed 11544 segments (0 failed to parse geometry)
  [W] Segments with >=1 network neighbour: 8463 / 11544
  [W] Mean neighbours: 1.8

Fitting Model D+W — ACCIDENTS (unmerged scoring)
  [accidents] train lag mean=0.533  new lag mean=0.472  (fallback used where no train neighbour)

Fitting Model D+W — DEATHS (unmerged scoring)
  [death] train lag mean=0.110  new lag mean=0.093  (fallback used where no train neighbour)
```

**Top 3 highest-risk unmerged segments** (ranked against `unmerged_only`):

| rank | OBJECTID | RoadClass | LandUse | pred_accidents_DW | pred_death_DW | risk_pct_accidents | risk_pct_death | risk_pct_combined | decile |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | 35551 | trunk   | URBAN | 10.398 | 1.269 | 99.977 | 99.895 | 99.936 | 10 |
| 2 | 17533 | primary | URBAN | 10.184 | 1.195 | 99.965 | 99.836 | 99.901 | 10 |
| 3 | 26558 | primary | RURAL |  7.839 | 1.239 | 99.918 | 99.871 | 99.895 | 10 |

Of the 8,463 new segments with at least one known-network neighbour, ~73%
could borrow real spatial signal from a training-set neighbour; the remaining
~27% (islands with no shared endpoint to any known road) fall back to the
population mean lag, which is a conservative default rather than a
zero-risk assumption.

---

## 4. Outputs

| File | Produced in | Contents |
|---|---|---|
| `road_safety_scores_accident.csv` | Part 1 | Per-segment accident predictions (B, B_INT, D, D+W) + ensemble + risk rank, training roads only |
| `road_safety_scores_death.csv` | Part 1 | Same, for death |
| `final_safety_risk.geojson` | Part 2 | Merged accident+death risk with 5-band classification and geometry, training roads only |
| `road_safety_scores_unmerged.csv` | Part 3 | Accident/death Model D+W predictions + combined risk percentile/decile, new roads only |
| `road_safety_scores_unmerged.geojson` | Part 3 | Same, with geometry |

## 5. Required notebook scope

`df_clean`, `df_clean_unmerged`, `gdf_merged` must already be defined before
running (each needs a `save_geometry_linestring` WKT column and an
`OBJECTID` column; `df_clean` and `df_clean_unmerged` must have completely
disjoint `OBJECTID`s). Run with `%run road_safety_pipeline_v3.py`.