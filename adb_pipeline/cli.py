"""Command-line orchestrator for the analysis stages.

The Street View imagery pipeline is the standalone script:

    uv run python accident_svi.py   # sample accident points -> fetch Street View
                                    #   -> YOLO-World + Mask2Former -> labelled images

It writes to output/_accident_svi/. The analysis stages below read those outputs:

    uv run adb-pipeline strata              # per-segment Speed Safety Score
    uv run adb-pipeline accidents           # join 2024 crashes to segments
    uv run adb-pipeline correlate           # object <-> speed presence stats
    uv run adb-pipeline accident-correlate  # object <-> crash presence stats
    uv run adb-pipeline plot                # speed / objects / accidents bubble chart
    uv run adb-pipeline all                 # every analysis stage, in order
"""

from __future__ import annotations

import argparse

from . import config

# The analysis stages read the Street View pipeline's outputs.
EXP = config.OUTPUT_DIR / "_accident_svi"
config.SAMPLES_CSV = EXP / "samples.csv"
config.DETECTIONS_JSON = EXP / "detections.json"
config.LANES_CSV = EXP / "lanes.csv"

STAGES = ["all", "strata", "accidents", "correlate", "accident-correlate", "plot"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="adb-pipeline", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=STAGES, help="which stage to run")
    args = parser.parse_args(argv)

    if args.stage in ("all", "strata"):
        from . import stage1_strata
        stage1_strata.run()

    if args.stage in ("all", "accidents"):
        from . import accidents
        accidents.run()

    if args.stage in ("all", "correlate"):
        from . import stage4_correlate
        stage4_correlate.run()

    if args.stage in ("all", "accident-correlate"):
        from . import accident_correlate
        accident_correlate.run()

    if args.stage in ("all", "plot"):
        from . import plot_correlation
        plot_correlation.run()

    print("[done]")


if __name__ == "__main__":
    main()
