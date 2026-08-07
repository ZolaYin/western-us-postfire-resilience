#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.transform import array_bounds
from scipy.signal import fftconvolve
from shapely.geometry import box


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘"
)
ROAD_SOURCE = (
    ROOT
    / "WesternUS_drivers_raw"
    / "westernus_road_trail_build_20260406"
    / "Roads_main_WesternUS.gpkg"
)
TRAIL_SOURCE = (
    ROOT
    / "WesternUS_drivers_raw"
    / "westernus_road_trail_build_20260406"
    / "Trails_like_WesternUS.gpkg"
)
REF_RASTER = (
    ROOT
    / "US_Fire_and_Ecology_Data"
    / "WesternUS_MTBS_1km"
    / "WesternUS_MTBS_t0_1km.tif"
)
OUT_DIR = ROOT / "WesternUS_drivers_raw" / "westernus_road_trail_build_legacy_nlcd_20260411"
ROAD_OUT = OUT_DIR / "road_density_WesternUS_filtered_r5km_1km_5070_corrected.tif"
TRAIL_OUT = OUT_DIR / "trail_density_WesternUS_r10km_1km_5070_corrected.tif"
SUMMARY_JSON = OUT_DIR / "corrected_road_trail_density_summary.json"
REPORT_MD = (
    ROOT / "US_Fire_and_Ecology_Data" / "WUS_1km" / "corrected_road_trail_density_report.md"
)

TARGET_EPSG = 5070
BLOCK_SIZE = 128
NODATA_VALUE = -9999.0


def load_reference_grid(reference_path: Path) -> dict:
    with rasterio.open(reference_path) as src:
        arr = src.read(1, masked=True)
        profile = src.profile.copy()
        valid_mask = np.ones((src.height, src.width), dtype=bool)
        if np.ma.is_masked(arr):
            valid_mask = ~np.ma.getmaskarray(arr)
        return {
            "path": reference_path,
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "bounds": src.bounds,
            "res": src.res,
            "profile": profile,
            "valid_mask": valid_mask,
        }


def load_lines(source_path: Path) -> tuple[gpd.GeoDataFrame, dict]:
    info = pyogrio.read_info(source_path)
    gdf = pyogrio.read_dataframe(source_path)
    if gdf.crs is None:
        raise ValueError(f"CRS missing for {source_path}")
    gdf = gdf.loc[~gdf.geometry.isna()].copy()
    gdf = gdf.loc[~gdf.geometry.is_empty].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf.loc[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if str(gdf.crs).upper() != f"EPSG:{TARGET_EPSG}":
        gdf = gdf.to_crs(f"EPSG:{TARGET_EPSG}")
    meta = {
        "path": str(source_path),
        "crs": str(info["crs"]),
        "geometry_type": str(info["geometry_type"]),
        "features": int(info["features"]),
        "bounds": [float(x) for x in info["total_bounds"]],
        "fields": [str(x) for x in info["fields"]],
    }
    return gdf, meta


def clip_lines_to_reference(lines_5070: gpd.GeoDataFrame, ref_bounds) -> gpd.GeoDataFrame:
    ref_box = box(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
    clipped = lines_5070.loc[lines_5070.intersects(ref_box)].copy()
    if clipped.empty:
        raise ValueError("No features intersect the corrected WesternUS reference extent.")
    clipped["geometry"] = clipped.geometry.intersection(ref_box)
    clipped = clipped.loc[~clipped.geometry.is_empty].copy()
    clipped = clipped.explode(index_parts=False).reset_index(drop=True)
    clipped = clipped.loc[clipped.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    return clipped


def cell_boxes_for_window(
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    transform,
    valid_mask_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list]:
    rows, cols = np.where(valid_mask_block)
    if rows.size == 0:
        return rows, cols, []
    abs_rows = row0 + rows
    abs_cols = col0 + cols
    x_left = transform.c + abs_cols * transform.a
    x_right = x_left + transform.a
    y_top = transform.f + abs_rows * transform.e
    y_bottom = y_top + transform.e
    geoms = [
        box(min(xl, xr), min(yb, yt), max(xl, xr), max(yb, yt))
        for xl, xr, yt, yb in zip(x_left, x_right, y_top, y_bottom)
    ]
    return abs_rows, abs_cols, geoms


def build_exact_length_raster(lines_5070: gpd.GeoDataFrame, ref: dict, block_size: int) -> np.ndarray:
    out = np.zeros((ref["height"], ref["width"]), dtype=np.float32)
    if lines_5070.empty:
        return out

    sindex = lines_5070.sindex
    if sindex is None:
        raise RuntimeError("Unable to build spatial index for lines.")

    for row0 in range(0, ref["height"], block_size):
        row1 = min(row0 + block_size, ref["height"])
        for col0 in range(0, ref["width"], block_size):
            col1 = min(col0 + block_size, ref["width"])
            block_mask = ref["valid_mask"][row0:row1, col0:col1]
            if not block_mask.any():
                continue

            block_transform = ref["transform"] * rasterio.Affine.translation(col0, row0)
            minx, miny, maxx, maxy = array_bounds(row1 - row0, col1 - col0, block_transform)
            candidate_idx = list(sindex.intersection((minx, miny, maxx, maxy)))
            if not candidate_idx:
                continue

            subset = lines_5070.iloc[candidate_idx][["geometry"]].copy()
            subset = subset.loc[subset.intersects(box(minx, miny, maxx, maxy))]
            if subset.empty:
                continue

            abs_rows, abs_cols, geoms = cell_boxes_for_window(
                row0=row0,
                row1=row1,
                col0=col0,
                col1=col1,
                transform=ref["transform"],
                valid_mask_block=block_mask,
            )
            if not geoms:
                continue

            cells = gpd.GeoDataFrame(
                {
                    "cell_id": np.arange(len(geoms), dtype=np.int64),
                    "row": abs_rows,
                    "col": abs_cols,
                },
                geometry=geoms,
                crs=lines_5070.crs,
            )
            intersections = gpd.overlay(cells, subset, how="intersection", keep_geom_type=False)
            if intersections.empty:
                continue

            intersections["seg_len_m"] = intersections.geometry.length
            length_by_cell = intersections.groupby("cell_id", observed=True)["seg_len_m"].sum()
            rows_hit = cells.loc[length_by_cell.index, "row"].to_numpy(dtype=int)
            cols_hit = cells.loc[length_by_cell.index, "col"].to_numpy(dtype=int)
            out[rows_hit, cols_hit] = length_by_cell.to_numpy(dtype=np.float32) / 1000.0

    return out


def circular_kernel(radius_m: int, cell_size_m: float) -> np.ndarray:
    radius_cells = int(math.ceil(radius_m / cell_size_m))
    offsets = np.arange(-radius_cells, radius_cells + 1, dtype=float)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    dist = np.sqrt((yy * cell_size_m) ** 2 + (xx * cell_size_m) ** 2)
    return (dist <= radius_m).astype(np.float32)


def density_from_length_raster(
    length_km_raster: np.ndarray,
    valid_mask: np.ndarray,
    radius_m: int,
    cell_size_m: float,
) -> np.ndarray:
    kernel = circular_kernel(radius_m=radius_m, cell_size_m=cell_size_m)
    summed_length_km = fftconvolve(length_km_raster, kernel, mode="same")
    circle_area_km2 = math.pi * (radius_m / 1000.0) ** 2
    density = (summed_length_km / circle_area_km2).astype(np.float32)
    density = np.where(density < 0, 0.0, density).astype(np.float32)
    density[~valid_mask] = NODATA_VALUE
    return density


def save_raster(out_path: Path, density: np.ndarray, ref: dict) -> None:
    profile = ref["profile"].copy()
    profile.update(driver="GTiff", dtype="float32", count=1, compress="lzw", nodata=NODATA_VALUE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(density, 1)


def summarize_density(density: np.ndarray) -> dict:
    valid = density != NODATA_VALUE
    vals = density[valid]
    return {
        "min": float(vals.min()) if vals.size else math.nan,
        "max": float(vals.max()) if vals.size else math.nan,
        "mean": float(vals.mean()) if vals.size else math.nan,
        "nodata_cells": int((~valid).sum()),
        "zero_cells": int((vals == 0).sum()) if vals.size else 0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

    ref = load_reference_grid(REF_RASTER)
    if str(ref["crs"]).upper() != "EPSG:5070":
        raise ValueError(f"Reference raster CRS is {ref['crs']}, expected EPSG:5070.")
    if tuple(ref["res"]) != (1000.0, 1000.0):
        raise ValueError(f"Reference raster resolution is {ref['res']}, expected 1000 m.")

    road_gdf, road_meta = load_lines(ROAD_SOURCE)
    trail_gdf, trail_meta = load_lines(TRAIL_SOURCE)

    print("Clipping filtered roads to corrected reference extent...", flush=True)
    road_clip = clip_lines_to_reference(road_gdf, ref["bounds"])
    print("Clipping trail-like lines to corrected reference extent...", flush=True)
    trail_clip = clip_lines_to_reference(trail_gdf, ref["bounds"])

    print("Building exact per-cell road length raster...", flush=True)
    road_length = build_exact_length_raster(road_clip, ref, BLOCK_SIZE)
    print("Building exact per-cell trail length raster...", flush=True)
    trail_length = build_exact_length_raster(trail_clip, ref, BLOCK_SIZE)

    print("Applying 5 km circular road density...", flush=True)
    road_density_r5 = density_from_length_raster(
        length_km_raster=road_length,
        valid_mask=ref["valid_mask"],
        radius_m=5000,
        cell_size_m=float(ref["res"][0]),
    )
    print("Applying 10 km circular trail density...", flush=True)
    trail_density_r10 = density_from_length_raster(
        length_km_raster=trail_length,
        valid_mask=ref["valid_mask"],
        radius_m=10000,
        cell_size_m=float(ref["res"][0]),
    )

    save_raster(ROAD_OUT, road_density_r5, ref)
    save_raster(TRAIL_OUT, trail_density_r10, ref)

    summary = {
        "reference_raster": str(REF_RASTER),
        "road_source": road_meta,
        "trail_source": trail_meta,
        "road_clip_features": int(len(road_clip)),
        "trail_clip_features": int(len(trail_clip)),
        "road_density_r5km_output": str(ROAD_OUT),
        "trail_density_r10km_output": str(TRAIL_OUT),
        "road_density_r5km_stats": summarize_density(road_density_r5),
        "trail_density_r10km_stats": summarize_density(trail_density_r10),
        "method": (
            "Exact line length per 1 km corrected WesternUS cell via vector intersections, "
            "followed by circular moving-window density with fftconvolve. "
            "Road density uses the verified filtered Roads_main_WesternUS source; "
            "trail density uses Trails_like_WesternUS."
        ),
        "uncertainty": (
            "The old project contains both filtered and broader road definitions. "
            "This rebuild uses the filtered main-road source because that is the verified explicit road-class subset."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Corrected WesternUS Road and Trail Density Rebuild",
        "",
        f"- Reference raster: `{REF_RASTER}`",
        f"- Road source: `{ROAD_SOURCE}`",
        f"- Trail source: `{TRAIL_SOURCE}`",
        f"- Road clipped features: `{len(road_clip)}`",
        f"- Trail clipped features: `{len(trail_clip)}`",
        f"- Road output: `{ROAD_OUT}`",
        f"- Trail output: `{TRAIL_OUT}`",
        "",
        "Method:",
        "- exact line length inside each corrected 1 km cell via vector intersections",
        "- road density radius = `5 km`",
        "- trail density radius = `10 km`",
        "- density convolution via `fftconvolve`",
        "- output units = `km/km^2`",
        "",
        "Uncertainty:",
        "- The broader versus filtered road production definition remains `uncertain` in the legacy project.",
        "- This rebuild uses `Roads_main_WesternUS.gpkg` because it is the verified explicit filtered road subset and is the safest match for the human-driver road variable.",
        "",
        "Road r5km stats:",
        f"- min: `{summary['road_density_r5km_stats']['min']}`",
        f"- max: `{summary['road_density_r5km_stats']['max']}`",
        f"- mean: `{summary['road_density_r5km_stats']['mean']}`",
        "",
        "Trail r10km stats:",
        f"- min: `{summary['trail_density_r10km_stats']['min']}`",
        f"- max: `{summary['trail_density_r10km_stats']['max']}`",
        f"- mean: `{summary['trail_density_r10km_stats']['mean']}`",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
