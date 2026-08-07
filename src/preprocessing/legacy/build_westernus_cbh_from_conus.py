#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘"
)
SHARED = Path(
    "/path/to/google-drive/共享云端硬盘"
)
BASE = ROOT / "US_Fire_and_Ecology_Data" / "WUS_1km"
REF_RASTER = ROOT / "US_Fire_and_Ecology_Data" / "WesternUS_MTBS_1km" / "WesternUS_MTBS_t0_1km.tif"
RESPONSE_TABLE = (
    BASE
    / "westernus_response_legacy_nlcd_20260410"
    / "westernus_response_metrics_legacy_nlcd.parquet"
)
CURRENT_TABLE = BASE / "westernus_current_candidate_table_legacy_nlcd.parquet"
OUT_DIR = BASE / "westernus_cbh_build_legacy_nlcd_20260411"
OUT_TABLE = OUT_DIR / "westernus_current_candidate_table_plus_cbh.parquet"
OUT_SAMPLE = OUT_DIR / "westernus_current_candidate_table_plus_cbh_sample.csv"
OUT_REPORT = OUT_DIR / "westernus_cbh_build_report.md"
OUT_SUMMARY = OUT_DIR / "westernus_cbh_build_summary.json"

NODATA_FLOAT = -9999.0


ZIP_SOURCES = {
    2008: {
        "zip_path": SHARED / "cbh" / "CBH2008_US (1).zip",
        "inner": "US_110cbh/grid2/us_110cbh",
        "kind": "arcinfo_grid",
    },
    2010: {
        "zip_path": SHARED / "cbh" / "CBH2010_US.zip",
        "inner": "US_110cbh/grid2/us_110cbh",
        "kind": "arcinfo_grid",
    },
    2012: {
        "zip_path": SHARED / "cbh" / "CBH2012_US.zip",
        "inner": "US_130_CBH/Tif/us_130cbh.tif",
        "kind": "tif",
    },
    2014: {
        "zip_path": SHARED / "cbh" / "CBH2014_US.zip",
        "inner": "US_140_CBH/Tif/us_140cbh.tif",
        "kind": "tif",
    },
    2016: {
        "zip_path": SHARED / "cbh" / "LF2016_CBH_CONUS.zip",
        "inner": "LF2016_CBH_CONUS/Tif/LF2016_CBH_CONUS.tif",
        "kind": "tif",
    },
    2022: {
        "zip_path": SHARED / "cbh" / "CBH2022_US.zip",
        "inner": "LF2022_CBH_220_CONUS/Tif/LC22_CBH_220.tif",
        "kind": "tif",
    },
}


def ensure_temp_link(year: int, zip_path: Path) -> Path:
    link = Path("/tmp") / f"cbh_{year}.zip"
    if link.exists() or link.is_symlink():
        try:
            if link.resolve() == zip_path.resolve():
                return link
        except Exception:
            pass
        link.unlink()
    os.symlink(zip_path, link)
    return link


def build_vsi_path(year: int) -> str:
    cfg = ZIP_SOURCES[year]
    zip_path = cfg["zip_path"]
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    link = ensure_temp_link(year, zip_path)
    return f"/vsizip/{link}/{cfg['inner']}"


def nearest_available(year: int, available_years: list[int]) -> int:
    return min(available_years, key=lambda y: (abs(y - year), 1 if y > year else 0, y))


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return ((series - mean) / std).astype(np.float32)


def load_ref() -> dict:
    with rasterio.open(REF_RASTER) as ref:
        arr = ref.read(1, masked=True)
        valid_mask = np.ones((ref.height, ref.width), dtype=bool)
        if np.ma.is_masked(arr):
            valid_mask = ~np.ma.getmaskarray(arr)
        return {
            "crs": ref.crs,
            "transform": ref.transform,
            "width": ref.width,
            "height": ref.height,
            "profile": ref.profile.copy(),
            "valid_mask": valid_mask,
            "bounds": tuple(ref.bounds),
            "res": ref.res,
        }


def source_metadata(year: int) -> dict:
    src_path = build_vsi_path(year)
    print(f"[CBH] opening source year {year}: {src_path}", flush=True)
    with rasterio.open(src_path) as src:
        return {
            "source_vsi": src_path,
            "source_driver": src.driver,
            "source_crs": str(src.crs),
            "source_shape": [src.height, src.width],
            "source_res": list(src.res),
            "source_bounds": list(src.bounds),
            "source_nodata": None if src.nodata is None else float(src.nodata),
        }


def sample_year_to_points(year: int, ref: dict, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    src_path = build_vsi_path(year)
    out = np.full(xs.shape[0], np.nan, dtype=np.float32)
    print(f"[CBH] sampling year {year} for {xs.shape[0]} points", flush=True)
    with rasterio.open(src_path) as src:
        with WarpedVRT(
            src,
            crs=ref["crs"],
            transform=ref["transform"],
            width=ref["width"],
            height=ref["height"],
            resampling=Resampling.average,
            nodata=NODATA_FLOAT,
            src_nodata=src.nodata,
        ) as vrt:
            vals = np.array([v[0] for v in vrt.sample(list(zip(xs, ys)))], dtype=np.float32)
    vals[np.isclose(vals, NODATA_FLOAT)] = np.nan
    vals[vals < 0] = 0.0
    out[:] = vals
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = load_ref()
    response = pd.read_parquet(
        RESPONSE_TABLE, columns=["pixel_id", "row", "col", "x", "y", "t0_year", "Resistance"]
    ).copy()
    current = pd.read_parquet(CURRENT_TABLE).copy()

    needed_years = sorted(ZIP_SOURCES)
    t0_years = sorted(response["t0_year"].dropna().astype(int).unique().tolist())
    t0_to_cbh = {year: nearest_available(year, needed_years) for year in t0_years}

    source_meta: dict[int, dict] = {}
    for year in needed_years:
        print(f"[CBH] start year {year}", flush=True)
        source_meta[year] = source_metadata(year)

    response["FS_CBH_1km"] = sample_year_to_points(
        2022,
        ref,
        response["x"].to_numpy(dtype=float),
        response["y"].to_numpy(dtype=float),
    )
    response["FS_CBH_t0agg_source_year"] = (
        response["t0_year"].astype(int).map(t0_to_cbh).astype(np.int16)
    )
    response["FS_CBH_t0agg"] = np.full(len(response), np.nan, dtype=np.float32)

    for source_year in needed_years:
        idx = response.index[response["FS_CBH_t0agg_source_year"] == source_year]
        if len(idx) == 0:
            continue
        xs = response.loc[idx, "x"].to_numpy(dtype=float)
        ys = response.loc[idx, "y"].to_numpy(dtype=float)
        response.loc[idx, "FS_CBH_t0agg"] = sample_year_to_points(source_year, ref, xs, ys)

    response["FS_CBH_1km_z"] = zscore(response["FS_CBH_1km"].astype(float))
    response["FS_CBH_t0agg_z"] = zscore(response["FS_CBH_t0agg"].astype(float))

    extra_cols = [
        "FS_CBH_1km",
        "FS_CBH_1km_z",
        "FS_CBH_t0agg_source_year",
        "FS_CBH_t0agg",
        "FS_CBH_t0agg_z",
    ]
    merge_cols = ["pixel_id"] + extra_cols
    merged = current.merge(response[merge_cols], on="pixel_id", how="left", validate="one_to_one")
    merged.to_parquet(OUT_TABLE, index=False)
    merged.head(1000).to_csv(OUT_SAMPLE, index=False)
    print(f"[CBH] wrote table {OUT_TABLE}", flush=True)

    static_valid = int(response["FS_CBH_1km"].notna().sum())
    t0agg_valid = int(response["FS_CBH_t0agg"].notna().sum())
    summary = {
        "reference_raster": str(REF_RASTER),
        "reference_shape": [ref["height"], ref["width"]],
        "reference_res": list(ref["res"]),
        "reference_bounds": list(ref["bounds"]),
        "rows_in_response": int(len(response)),
        "t0_years_used": t0_years,
        "cbh_available_years": needed_years,
        "t0_to_cbh_mapping": {str(k): int(v) for k, v in t0_to_cbh.items()},
        "static_2022_valid_rows": static_valid,
        "t0agg_valid_rows": t0agg_valid,
        "static_2022_min": float(np.nanmin(response["FS_CBH_1km"])),
        "static_2022_max": float(np.nanmax(response["FS_CBH_1km"])),
        "t0agg_min": float(np.nanmin(response["FS_CBH_t0agg"])),
        "t0agg_max": float(np.nanmax(response["FS_CBH_t0agg"])),
        "source_meta": source_meta,
        "output_table": str(OUT_TABLE),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    report = f"""# WesternUS CBH Build Report

## Source of truth
- Reference grid: `{REF_RASTER}`
- Current candidate table used as base: `{CURRENT_TABLE}`
- Legacy aggregation method matched: `Resampling.average` from 30 m CBH into 1 km EPSG:5070 reference cells
- Full-US CBH shared drive source folder: `{SHARED / 'cbh'}`

## Verified CBH source years
- Available years used: {needed_years}
- Response `t0_year` coverage: {t0_years}
- Mapping rule: nearest available year, ties choose earlier year

## Outputs
- New table: `{OUT_TABLE}`
- Sample CSV: `{OUT_SAMPLE}`

## Fields added
- `FS_CBH_1km`
- `FS_CBH_1km_z`
- `FS_CBH_t0agg_source_year`
- `FS_CBH_t0agg`
- `FS_CBH_t0agg_z`

## Coverage
- Static 2022 valid rows: {static_valid} / {len(response)}
- Near-t0 valid rows: {t0agg_valid} / {len(response)}

## Notes
- `FS_CBH_1km` here is a static 2022 CBH sampled from a WarpedVRT configured exactly to the current WesternUS response grid.
- `FS_CBH_t0agg` follows the later near-t0 aggregated production logic and is likely the better match to the formal near-t0 modeling chain.
- The current script does not write full intermediate 1 km CBH rasters; it samples values directly from the target-grid WarpedVRT for the fire-pixel centers. Given the VRT is pinned to the exact target transform, CRS, width, and height, this preserves the intended target-grid cell values for the sampled pixels.
- Any difference between the old coast-era static CBH and this WesternUS build is due to spatial scope and the verified use of CONUS sources rather than the older WUS-only rasters.
- If `TCC` is later added, the safest next modeling table will likely use `FS_CBH_t0agg_z` rather than `FS_CBH_1km_z` because that is what the near-t0 official scripts retain.
- Anything beyond these verified steps is `uncertain`.
"""
    OUT_REPORT.write_text(report)


if __name__ == "__main__":
    main()
