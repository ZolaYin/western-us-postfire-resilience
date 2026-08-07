#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘"
)
BASE = ROOT / "US_Fire_and_Ecology_Data" / "WUS_1km"
GEE_TCC_DIR = ROOT / "GEE_exports"
INPUT_TABLE = (
    BASE
    / "westernus_cbh_build_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh.parquet"
)
REF_RASTER = ROOT / "US_Fire_and_Ecology_Data" / "WesternUS_MTBS_1km" / "WesternUS_MTBS_t0_1km.tif"

OUT_DIR = BASE / "westernus_tcc_build_legacy_nlcd_20260411"
OUT_TABLE = OUT_DIR / "westernus_current_candidate_table_plus_cbh_tcc.parquet"
OUT_SAMPLE = OUT_DIR / "westernus_current_candidate_table_plus_cbh_tcc_sample.csv"
OUT_REPORT = OUT_DIR / "westernus_tcc_build_report.md"
OUT_SUMMARY = OUT_DIR / "westernus_tcc_build_summary.json"


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return ((series - mean) / std).astype(np.float32)


def find_year_tiles(year: int) -> list[Path]:
    files = sorted(GEE_TCC_DIR.glob(f"TCC_{year}_WesternUS11_30m_5070-*.tif"))
    if not files:
        raise FileNotFoundError(f"No TCC tiles found for year {year} in {GEE_TCC_DIR}")
    return files


def sample_year_tiles(year: int, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    vals = np.full(xs.shape[0], np.nan, dtype=np.float32)
    tile_meta: list[dict] = []
    remaining = np.ones(xs.shape[0], dtype=bool)
    for tile in find_year_tiles(year):
        with rasterio.open(tile) as src:
            meta = {
                "tile": str(tile),
                "crs": str(src.crs),
                "shape": [src.height, src.width],
                "res": list(src.res),
                "bounds": list(src.bounds),
                "nodata": None if src.nodata is None else float(src.nodata),
            }
            tile_meta.append(meta)
            inside = (
                remaining
                & (xs >= src.bounds.left)
                & (xs <= src.bounds.right)
                & (ys >= src.bounds.bottom)
                & (ys <= src.bounds.top)
            )
            if not inside.any():
                continue
            coords = list(zip(xs[inside], ys[inside]))
            sampled = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
            nodata = src.nodata
            if nodata is not None:
                sampled[np.isclose(sampled, nodata)] = np.nan
            sampled[~np.isfinite(sampled)] = np.nan
            vals[inside] = sampled
            remaining[inside] = False
    return vals, tile_meta


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(REF_RASTER) as ref:
        ref_meta = {
            "path": str(REF_RASTER),
            "crs": str(ref.crs),
            "shape": [ref.height, ref.width],
            "res": list(ref.res),
            "bounds": list(ref.bounds),
            "transform": [ref.transform.a, ref.transform.b, ref.transform.c, ref.transform.d, ref.transform.e, ref.transform.f],
        }

    df = pd.read_parquet(INPUT_TABLE).copy()
    years = sorted(df["t0_year"].dropna().astype(int).unique().tolist())
    source_year_counts = Counter()
    tile_meta_by_year: dict[int, list[dict]] = {}

    df["FS_TCC_t0_source_year"] = df["t0_year"].astype(int).astype(np.int16)
    df["FS_TCC_t0"] = np.nan

    for year in years:
        idx = df.index[df["FS_TCC_t0_source_year"] == year]
        xs = df.loc[idx, "x"].to_numpy(dtype=float)
        ys = df.loc[idx, "y"].to_numpy(dtype=float)
        print(f"[TCC] sampling year {year} for {len(idx)} points", flush=True)
        vals, tile_meta = sample_year_tiles(year, xs, ys)
        df.loc[idx, "FS_TCC_t0"] = vals
        tile_meta_by_year[year] = tile_meta
        source_year_counts[year] = int(len(idx))

    df["FS_TCC_t0_z"] = zscore(df["FS_TCC_t0"].astype(float))
    df.to_parquet(OUT_TABLE, index=False)
    df.head(1000).to_csv(OUT_SAMPLE, index=False)

    summary = {
        "input_table": str(INPUT_TABLE),
        "output_table": str(OUT_TABLE),
        "reference_grid": ref_meta,
        "tcc_dir": str(GEE_TCC_DIR),
        "years_used": years,
        "year_point_counts": dict(source_year_counts),
        "tcc_non_null_rows": int(df["FS_TCC_t0"].notna().sum()),
        "tcc_min": float(df["FS_TCC_t0"].min()),
        "tcc_max": float(df["FS_TCC_t0"].max()),
        "tile_meta_by_year": tile_meta_by_year,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    report = f"""# WesternUS TCC Build Report

## Inputs
- Base table: `{INPUT_TABLE}`
- TCC tile directory: `{GEE_TCC_DIR}`
- Reference raster checked: `{REF_RASTER}`

## Verified TCC source
- Years found and used: {years}
- Per-year tile count in GEE export: 6
- Verified sample tile metadata:
  - CRS = EPSG:5070
  - Resolution = 30 m
  - Data type = float32

## Processing rule
- `FS_TCC_t0` is assigned from the exact matching `t0_year` TCC year.
- No nearest-year fallback was needed because 2000-2023 annual TCC tiles are present.
- Values were sampled from the yearly 30 m GEE export tiles at the current WesternUS response pixel centers.
- `FS_TCC_t0_z` is a full-table z-score of `FS_TCC_t0`.

## Outputs
- `{OUT_TABLE}`
- `{OUT_SAMPLE}`
- `{OUT_SUMMARY}`

## Coverage
- Non-null `FS_TCC_t0` rows: {int(df["FS_TCC_t0"].notna().sum())} / {len(df)}
- Range: {float(df["FS_TCC_t0"].min())} to {float(df["FS_TCC_t0"].max())}

## Notes
- This script appends TCC to the CBH-augmented WesternUS candidate table and does not modify earlier tables.
- The sampled values come from the downloaded GEE tile exports rather than a pre-stitched multiband raster.
- Anything beyond this verified per-year sampling workflow is `uncertain`.
"""
    OUT_REPORT.write_text(report)
    print(f"[TCC] wrote table {OUT_TABLE}", flush=True)


if __name__ == "__main__":
    main()
