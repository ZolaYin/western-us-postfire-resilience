#!/usr/bin/env python3
"""Spatialized MGWR-based resilience management zoning.

This upgrades the sample-point pilot by:

1. Interpolating existing stage5b Resistance-MGWR coefficients from the 12k
   fitted MGWR sample to all available post-fire candidate pixels using IDW.
2. Computing the same forest/climate/human mechanism scores for the full
   candidate set.
3. Aggregating scores to a regular 50 km EPSG:5070 management grid before
   assigning zones, reducing speckle and making the output closer to a usable
   management zoning map.

The coefficient interpolation is a pilot approximation. A final product should
prefer true MGWR prediction/coefficient output on the full grid if available.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import geopandas as gpd

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import box
from sklearn.neighbors import NearestNeighbors

from compute_resilience_management_zoning_mgwr_pilot import (
    BASE,
    COEF_FILE,
    PREDICTOR_FILE,
    CANDIDATE_FILE,
    DIMENSIONS,
    ZONE_ORDER,
    ZONE_COLORS,
    add_integer_keys,
    compute_scores,
    assign_zones,
    quartile_level,
)


OUT_DIR = BASE / "resilience_management_zoning_mgwr_spatialized_2026-05-28"
CELL_KM_LIST = [50, 100]
MIN_PIXELS_PER_CELL = 10
IDW_K = 8
IDW_POWER = 2.0


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def load_full_candidate_predictor_table() -> pd.DataFrame:
    predictors = add_integer_keys(pd.read_parquet(PREDICTOR_FILE))
    candidate = add_integer_keys(pd.read_parquet(CANDIDATE_FILE))

    status_cols = [
        "sev",
        "Forest_at_t0",
        "T80",
        "T80_reached",
        "IRI_good_pow2",
        "STAB_good_pow2",
        "FS_EVT2022_code",
        "FS_EVT_group_class",
        "region",
        "lon_wgs84",
        "lat_wgs84",
    ]
    status_cols = [c for c in status_cols if c in candidate.columns]
    full = predictors.merge(
        candidate[["_xi", "_yi", *status_cols]],
        on=["_xi", "_yi"],
        how="left",
    )
    return full


def interpolate_coefficients(full: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    all_drivers = [c for cols in DIMENSIONS.values() for c in cols]
    coef = add_integer_keys(pd.read_parquet(COEF_FILE))
    coef = coef.rename(columns={c: f"beta_{c}" for c in all_drivers})

    sample_coords = coef[["x", "y"]].to_numpy(dtype=float)
    target_coords = full[["x", "y"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=IDW_K, algorithm="ball_tree")
    nn.fit(sample_coords)
    dists, inds = nn.kneighbors(target_coords, return_distance=True)

    safe_dists = np.maximum(dists, 1.0)
    weights = 1.0 / np.power(safe_dists, IDW_POWER)
    weights = weights / weights.sum(axis=1, keepdims=True)

    out = full.copy()
    for driver in all_drivers:
        beta = coef[f"beta_{driver}"].to_numpy(dtype=float)
        out[f"beta_{driver}"] = np.sum(beta[inds] * weights, axis=1)

    meta = {
        "idw_k": IDW_K,
        "idw_power": IDW_POWER,
        "nearest_distance_median_m": float(np.median(dists[:, 0])),
        "nearest_distance_p90_m": float(np.percentile(dists[:, 0], 90)),
        "nearest_distance_p95_m": float(np.percentile(dists[:, 0], 95)),
        "nearest_distance_max_m": float(np.max(dists[:, 0])),
    }
    return out, meta


def prepare_spatialized_points() -> tuple[pd.DataFrame, dict]:
    full = load_full_candidate_predictor_table()
    full, interp_meta = interpolate_coefficients(full)

    all_drivers = [c for cols in DIMENSIONS.values() for c in cols]
    required = [
        "Resistance",
        "T80",
        "IRI_good_pow2",
        "STAB_good_pow2",
        *all_drivers,
        *[f"beta_{c}" for c in all_drivers],
    ]
    full = full.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    scored = compute_scores(full)
    point_zones = assign_zones(scored)
    return point_zones, interp_meta


def aggregate_to_grid(points: pd.DataFrame, cell_km: int) -> gpd.GeoDataFrame:
    cell_m = float(cell_km * 1000)
    df = points.copy()
    df["grid_x"] = np.floor(df["x"] / cell_m).astype(int)
    df["grid_y"] = np.floor(df["y"] / cell_m).astype(int)

    mode_cols = ["region", "FS_EVT_group_class", "dominant_constraint_dimension", "dominant_effect_dimension"]
    for col in mode_cols:
        if col not in df.columns:
            df[col] = ""

    grouped = (
        df.groupby(["grid_x", "grid_y"], observed=True)
        .agg(
            n_pixels=("x", "size"),
            x_mean=("x", "mean"),
            y_mean=("y", "mean"),
            Resistance=("Resistance", "mean"),
            T80=("T80", "mean"),
            IRI_good_pow2=("IRI_good_pow2", "mean"),
            STAB_good_pow2=("STAB_good_pow2", "mean"),
            R_comp=("R_comp", "mean"),
            forest_structure_support=("forest_structure_support", "median"),
            forest_structure_constraint=("forest_structure_constraint", "median"),
            climate_constraint_score=("climate_constraint_score", "median"),
            human_pressure_score=("human_pressure_score", "median"),
            forest_structure_net=("forest_structure_net", "median"),
            climate_net=("climate_net", "median"),
            human_net=("human_net", "median"),
            region_mode=("region", lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else ""),
            evt_group_mode=(
                "FS_EVT_group_class",
                lambda s: s.mode(dropna=True).iloc[0] if not s.mode(dropna=True).empty else "",
            ),
            point_zone_mode=(
                "management_zone",
                lambda s: s.astype(str).mode(dropna=True).iloc[0] if not s.astype(str).mode(dropna=True).empty else "",
            ),
        )
        .reset_index()
    )
    grouped = grouped[grouped["n_pixels"] >= MIN_PIXELS_PER_CELL].reset_index(drop=True)

    grouped["forest_level"] = quartile_level(grouped["forest_structure_support"])
    grouped["climate_level"] = quartile_level(grouped["climate_constraint_score"])
    grouped["human_level"] = quartile_level(grouped["human_pressure_score"])
    grouped["cube_code"] = (
        grouped["forest_level"].astype(str)
        + "-"
        + grouped["climate_level"].astype(str)
        + "-"
        + grouped["human_level"].astype(str)
    )
    grouped["max_constraint_score"] = grouped[
        ["forest_structure_constraint", "climate_constraint_score", "human_pressure_score"]
    ].max(axis=1)
    grouped["dominant_constraint_dimension"] = grouped[
        ["forest_structure_constraint", "climate_constraint_score", "human_pressure_score"]
    ].idxmax(axis=1).map(
        {
            "forest_structure_constraint": "forest_structure",
            "climate_constraint_score": "climate_constraint",
            "human_pressure_score": "human_pressure",
        }
    )
    grouped["dominant_effect_dimension"] = pd.DataFrame(
        {
            "forest_structure": grouped["forest_structure_net"].abs(),
            "climate_constraint": grouped["climate_net"].abs(),
            "human_pressure": grouped["human_net"].abs(),
        }
    ).idxmax(axis=1)

    zoned = assign_zones(grouped)
    geometries = [
        box(row.grid_x * cell_m, row.grid_y * cell_m, (row.grid_x + 1) * cell_m, (row.grid_y + 1) * cell_m)
        for row in zoned.itertuples(index=False)
    ]
    gdf = gpd.GeoDataFrame(zoned, geometry=geometries, crs="EPSG:5070")
    return gdf


def summarize_grid_zones(grid: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = []
    for zone_name in ZONE_ORDER:
        sub = grid[grid["management_zone"].astype(str) == zone_name]
        if sub.empty:
            continue
        w = sub["n_pixels"]
        rows.append(
            {
                "management_zone": zone_name,
                "n_cells": int(len(sub)),
                "n_pixels": int(w.sum()),
                "pct_cells": 100 * len(sub) / len(grid),
                "pct_pixels": 100 * w.sum() / grid["n_pixels"].sum(),
                "weighted_R_comp": weighted_mean(sub["R_comp"], w),
                "weighted_Resistance": weighted_mean(sub["Resistance"], w),
                "weighted_T80": weighted_mean(sub["T80"], w),
                "weighted_IRI_good_pow2": weighted_mean(sub["IRI_good_pow2"], w),
                "weighted_STAB_good_pow2": weighted_mean(sub["STAB_good_pow2"], w),
                "median_forest_support": sub["forest_structure_support"].median(),
                "median_climate_constraint": sub["climate_constraint_score"].median(),
                "median_human_pressure": sub["human_pressure_score"].median(),
                "evt_group_mode": sub["evt_group_mode"].mode(dropna=True).iloc[0]
                if not sub["evt_group_mode"].mode(dropna=True).empty
                else "",
                "region_mode": sub["region_mode"].mode(dropna=True).iloc[0]
                if not sub["region_mode"].mode(dropna=True).empty
                else "",
            }
        )
    return pd.DataFrame(rows)


def save_point_geopackage(points: pd.DataFrame, out_dir: Path) -> None:
    keep = [
        "pixel_id",
        "row",
        "col",
        "x",
        "y",
        "lon_wgs84",
        "lat_wgs84",
        "region",
        "FS_EVT_group_class",
        "Resistance",
        "T80",
        "IRI_good_pow2",
        "STAB_good_pow2",
        "R_comp",
        "forest_structure_support",
        "forest_structure_constraint",
        "climate_constraint_score",
        "human_pressure_score",
        "forest_level",
        "climate_level",
        "human_level",
        "cube_code",
        "management_zone",
        "low_priority_subtype",
    ]
    keep = [c for c in keep if c in points.columns]
    gdf = gpd.GeoDataFrame(
        points[keep].copy(),
        geometry=gpd.points_from_xy(points["x"], points["y"]),
        crs="EPSG:5070",
    )
    gdf.to_file(out_dir / "spatialized_point_zones_full_candidate.gpkg", driver="GPKG")


def plot_grid(grid: gpd.GeoDataFrame, summary: pd.DataFrame, out_dir: Path, cell_km: int) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    panels = [
        ("R_comp", "Composite resilience status", "RdYlGn"),
        ("forest_structure_support", "Forest structure support", "Greens"),
        ("climate_constraint_score", "Climate constraint", "Blues"),
        ("human_pressure_score", "Human pressure", "Oranges"),
    ]
    for ax, (col, title, cmap) in zip(axes.ravel()[:4], panels):
        grid.plot(column=col, ax=ax, cmap=cmap, linewidth=0, legend=True)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    ax = axes.ravel()[4]
    for zone_name in ZONE_ORDER:
        sub = grid[grid["management_zone"].astype(str) == zone_name]
        if sub.empty:
            continue
        sub.plot(ax=ax, color=ZONE_COLORS[zone_name], linewidth=0, alpha=0.9)
    ax.set_title(f"{cell_km} km aggregated management zones", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        mpatches.Patch(color=ZONE_COLORS[z], label=z.replace(" zone", ""))
        for z in ZONE_ORDER
        if not grid[grid["management_zone"].astype(str) == z].empty
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=True)

    ax = axes.ravel()[5]
    plot_summary = summary.set_index("management_zone").reindex(ZONE_ORDER).dropna(subset=["n_cells"])
    y_pos = np.arange(len(plot_summary))
    bars = ax.barh(
        y_pos,
        plot_summary["pct_pixels"],
        color=[ZONE_COLORS[z] for z in plot_summary.index],
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([z.replace(" zone", "") for z in plot_summary.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of candidate pixels in retained grid cells")
    ax.set_title("Zone share and weighted mean resilience", fontsize=11, fontweight="bold")
    for bar, (_, row) in zip(bars, plot_summary.iterrows()):
        ax.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"cells={int(row['n_cells'])}, R={row['weighted_R_comp']:.2f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(40, plot_summary["pct_pixels"].max() + 10))
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Spatialized MGWR resilience management zoning",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(out_dir / f"resilience_management_zoning_spatialized_{cell_km}km.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_note(out_dir: Path, metadata: dict, cell_km: int) -> None:
    text = f"""# Spatialized MGWR Resilience Management Zoning

This output addresses the speckled sample-point behavior in the first pilot.

## What changed

1. Existing stage5b Resistance-MGWR coefficients were interpolated from 12,000 fitted MGWR sample points to all candidate post-fire pixels with inverse-distance weighting.
2. Forest, climate, and human mechanism scores were computed at the full candidate-pixel level.
3. Scores were aggregated to {cell_km} km EPSG:5070 grid cells before classifying management zones.
4. Cells with fewer than {MIN_PIXELS_PER_CELL} candidate pixels were omitted from the grid map.

## Why this is more appropriate

Management zones should be spatial units, not individual random MGWR sample points. The first pilot was useful for testing the mechanism-score logic, but it was not cartographically or managerially coherent. This version uses the point scores as evidence and classifies larger management units, which should be closer to a publishable zoning figure.

## Remaining caveat

The coefficient surface is still approximated by IDW interpolation. Final analysis should use true full-grid MGWR coefficient prediction/output if available, or explicitly describe this as a spatialized pilot/sensitivity analysis.

## Interpolation diagnostics

- IDW k: {metadata["idw_k"]}
- IDW power: {metadata["idw_power"]}
- Median nearest MGWR sample distance: {metadata["nearest_distance_median_m"]:.1f} m
- 90th percentile nearest distance: {metadata["nearest_distance_p90_m"]:.1f} m
- 95th percentile nearest distance: {metadata["nearest_distance_p95_m"]:.1f} m
- Maximum nearest distance: {metadata["nearest_distance_max_m"]:.1f} m
"""
    (out_dir / f"spatialized_zoning_notes_{cell_km}km.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points, interp_meta = prepare_spatialized_points()

    points.to_parquet(OUT_DIR / "spatialized_point_zones_full_candidate.parquet", index=False)
    save_point_geopackage(points, OUT_DIR)

    metadata = {
        **interp_meta,
        "n_full_candidate_points": int(len(points)),
        "min_pixels_per_cell": MIN_PIXELS_PER_CELL,
        "grid_outputs": {},
        "output_dir": str(OUT_DIR),
    }
    for cell_km in CELL_KM_LIST:
        grid = aggregate_to_grid(points, cell_km=cell_km)
        summary = summarize_grid_zones(grid)
        grid.to_file(OUT_DIR / f"spatialized_management_zones_{cell_km}km.gpkg", driver="GPKG")
        summary.to_csv(OUT_DIR / f"spatialized_zone_summary_{cell_km}km.csv", index=False)
        plot_grid(grid, summary, OUT_DIR, cell_km=cell_km)
        metadata["grid_outputs"][str(cell_km)] = {
            "cell_km": cell_km,
            "n_grid_cells": int(len(grid)),
            "zone_counts_cells": grid["management_zone"].astype(str).value_counts().to_dict(),
            "zone_counts_pixels": grid.groupby(grid["management_zone"].astype(str))["n_pixels"].sum().astype(int).to_dict(),
        }
        write_note(OUT_DIR, {**metadata, **interp_meta}, cell_km)

    (OUT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
