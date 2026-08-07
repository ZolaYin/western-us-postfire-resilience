#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT_TABLE = (
    ROOT
    / "westernus_tcc_build_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc.parquet"
)
ROAD_RASTER = (
    ROOT.parent.parent
    / "WesternUS_drivers_raw"
    / "westernus_road_trail_build_legacy_nlcd_20260411"
    / "road_density_WesternUS_filtered_r5km_1km_5070_corrected.tif"
)
TRAIL_RASTER = (
    ROOT.parent.parent
    / "WesternUS_drivers_raw"
    / "westernus_road_trail_build_legacy_nlcd_20260411"
    / "trail_density_WesternUS_r10km_1km_5070_corrected.tif"
)
OUT_DIR = ROOT / "westernus_roadtrail_append_legacy_nlcd_20260411"
OUT_TABLE = OUT_DIR / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
OUT_SAMPLE = OUT_DIR / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail_sample.csv"
OUT_SUMMARY = OUT_DIR / "westernus_roadtrail_append_summary.json"
OUT_REPORT = OUT_DIR / "westernus_roadtrail_append_report.md"


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def sample_raster(raster_path: Path, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, dict]:
    with rasterio.open(raster_path) as src:
        vals = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=np.float32)
        nodata = src.nodata
        if nodata is not None:
            vals[np.isclose(vals, nodata)] = np.nan
        vals[~np.isfinite(vals)] = np.nan
        meta = {
            "path": str(raster_path),
            "crs": str(src.crs),
            "shape": [src.height, src.width],
            "res": list(src.res),
            "bounds": list(src.bounds),
            "nodata": None if src.nodata is None else float(src.nodata),
        }
    return vals, meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_TABLE).copy()
    xs = df["x"].to_numpy(dtype=float)
    ys = df["y"].to_numpy(dtype=float)

    road_vals, road_meta = sample_raster(ROAD_RASTER, xs, ys)
    trail_vals, trail_meta = sample_raster(TRAIL_RASTER, xs, ys)

    df["HUM_roaddens_r5km"] = road_vals
    df["HUM_traildens_r10km"] = trail_vals
    df["HUM_roaddens_r5km_z"] = zscore(df["HUM_roaddens_r5km"])
    df["HUM_traildens_r10km_z"] = zscore(df["HUM_traildens_r10km"])

    df.to_parquet(OUT_TABLE, index=False)
    df.head(1000).to_csv(OUT_SAMPLE, index=False)

    summary = {
        "input_table": str(INPUT_TABLE),
        "output_table": str(OUT_TABLE),
        "road_raster": road_meta,
        "trail_raster": trail_meta,
        "rows": int(len(df)),
        "road_valid": int(df["HUM_roaddens_r5km"].notna().sum()),
        "trail_valid": int(df["HUM_traildens_r10km"].notna().sum()),
        "road_min": float(np.nanmin(df["HUM_roaddens_r5km"])),
        "road_max": float(np.nanmax(df["HUM_roaddens_r5km"])),
        "trail_min": float(np.nanmin(df["HUM_traildens_r10km"])),
        "trail_max": float(np.nanmax(df["HUM_traildens_r10km"])),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Corrected Road/Trail Append",
        "",
        f"- Input table: `{INPUT_TABLE}`",
        f"- Output table: `{OUT_TABLE}`",
        f"- Road raster: `{ROAD_RASTER}`",
        f"- Trail raster: `{TRAIL_RASTER}`",
        f"- Rows: `{len(df)}`",
        f"- Road valid: `{summary['road_valid']}`",
        f"- Trail valid: `{summary['trail_valid']}`",
        f"- Road min/max: `{summary['road_min']}` / `{summary['road_max']}`",
        f"- Trail min/max: `{summary['trail_min']}` / `{summary['trail_max']}`",
        "",
        "- Added columns: `HUM_roaddens_r5km`, `HUM_roaddens_r5km_z`, `HUM_traildens_r10km`, `HUM_traildens_r10km_z`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
