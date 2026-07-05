"""Stage 2 — fetch a Google Street View Static image per sampled segment.

For each sample we already computed a representative (lat, lon) and a road
heading in Stage 1. We:
  1. Query the GSV *metadata* endpoint (free) to confirm imagery exists and get
     the actual pano location/date.
  2. If imagery exists, fetch the Street View Static image aimed along the road.

When ``GSV_API_KEY`` is absent (config.GSV_MOCK), we generate a deterministic
synthetic placeholder image instead, so Stages 3-4 remain fully runnable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

from . import config

_META_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
_IMG_URL = "https://maps.googleapis.com/maps/api/streetview"


def _mock_image(path: Path, lat: float, lon: float, heading: float) -> dict:
    """Deterministic placeholder so the pipeline runs without a real API key."""
    seed = int(hashlib.md5(f"{lat},{lon}".encode()).hexdigest(), 16)
    w, h = (int(x) for x in config.GSV_IMAGE_SIZE.split("x"))
    img = Image.new("RGB", (w, h), (seed % 90 + 100, (seed >> 8) % 90 + 100, 150))
    d = ImageDraw.Draw(img)
    # crude "road" so detectors have some structure to chew on
    d.polygon([(w * 0.45, h), (w * 0.55, h), (w * 0.52, h * 0.55), (w * 0.48, h * 0.55)],
              fill=(60, 60, 60))
    d.text((10, 10), f"MOCK GSV\n{lat:.5f},{lon:.5f}\nhead {heading:.0f}", fill=(255, 255, 255))
    img.save(path)
    return {"status": "MOCK", "pano_date": None, "pano_lat": lat, "pano_lon": lon}


def _fetch_metadata(lat: float, lon: float) -> dict:
    r = requests.get(
        _META_URL,
        params={"location": f"{lat},{lon}", "key": config.GSV_API_KEY},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def _fetch_image(path: Path, lat: float, lon: float, heading: float) -> None:
    r = requests.get(
        _IMG_URL,
        params={
            "location": f"{lat},{lon}",
            "size": config.GSV_IMAGE_SIZE,
            "heading": f"{heading:.0f}",
            "fov": config.GSV_FOV,
            "pitch": config.GSV_PITCH,
            "key": config.GSV_API_KEY,
        },
        timeout=30,
    )
    r.raise_for_status()
    path.write_bytes(r.content)


def run(samples: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fetch images for all samples. Returns a fetch-log DataFrame."""
    config.ensure_dirs()
    if samples is None:
        samples = pd.read_csv(config.SAMPLES_CSV)

    mode = "MOCK (no API key)" if config.GSV_MOCK else "LIVE Google Street View"
    print(f"[stage2] fetching {len(samples)} images — mode: {mode}")

    records = []
    for _, r in tqdm(samples.iterrows(), total=len(samples), desc="GSV"):
        sid = r["sample_id"]
        lat, lon, heading = float(r["lat"]), float(r["lon"]), float(r["heading"])
        img_path = config.IMAGES_DIR / f"{sid}.jpg"
        rec = {"sample_id": sid, "lat": lat, "lon": lon, "heading": heading,
               "image_path": str(img_path)}

        try:
            if config.GSV_MOCK:
                rec.update(_mock_image(img_path, lat, lon, heading))
            else:
                meta = _fetch_metadata(lat, lon)
                rec["status"] = meta.get("status")
                if meta.get("status") == "OK":
                    loc = meta.get("location", {})
                    rec["pano_date"] = meta.get("date")
                    rec["pano_lat"] = loc.get("lat")
                    rec["pano_lon"] = loc.get("lng")
                    _fetch_image(img_path, lat, lon, heading)
                else:
                    rec["image_path"] = None  # no imagery (e.g. ZERO_RESULTS)
        except Exception as e:  # network/HTTP errors shouldn't kill the batch
            rec["status"] = f"ERROR: {e}"
            rec["image_path"] = None
        records.append(rec)

    log = pd.DataFrame(records)
    log.to_csv(config.GSV_METADATA_CSV, index=False)
    ok = (log["image_path"].notna()).sum()
    print(f"[stage2] images saved: {ok}/{len(log)} -> {config.IMAGES_DIR}")
    print(f"[stage2] wrote {config.GSV_METADATA_CSV}")
    return log


if __name__ == "__main__":
    run()
