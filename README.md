# Model and Safety Score

Street View object detection and the Speed Safety Score for the ADB AI for Safer
Roads challenge.

## Pipeline

1. Load road segments from `ADB_Innovation_Thailand.geojson`.
2. Match 2024 accident points (speeding / cutting-in) to their nearest segment.
3. Sample 1000 segments with at least two accidents. For each, take the two
   farthest-apart accident points and fetch Google Street View at those exact
   coordinates.
4. Run YOLO-World object detection and Mask2Former scene segmentation on each of
   the 2000 images and write labelled imagery.
5. Compute the per-segment Speed Safety Score and the speed / objects / accidents
   correlation bubble chart.

## Setup

    uv sync
    cp .env.example .env      # add GSV_API_KEY for live Street View

torch and torchvision are pinned to the CUDA 12.8 wheel index (RTX 50-series).

## Run

    uv run python accident_svi.py       # imagery pipeline: points, SVI, YOLO + Mask2Former
    uv run adb-pipeline strata          # per-segment Speed Safety Score
    uv run adb-pipeline accidents       # join 2024 crashes to segments
    uv run adb-pipeline correlate       # object vs speed stats
    uv run adb-pipeline accident-correlate
    uv run adb-pipeline plot            # speed / objects / accidents bubble chart
    uv run adb-pipeline all             # every analysis stage in order

## Layout

- `accident_svi.py`: the Street View imagery pipeline
- `adb_pipeline/`: segment loading, accident join, detection, segmentation,
  safety score, correlation, bubble chart
- `ADB_Innovation_Thailand.geojson`, `accident_data_2024_english.csv`: inputs

Model weights (YOLO-World, Mask2Former) download at runtime and are not committed.
Over/under is judged on the 85th-percentile operating speed vs the posted limit.
Correlations are observational, not causal.
