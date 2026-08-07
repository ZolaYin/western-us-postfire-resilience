#!/usr/bin/env python3
"""Build manuscript assets from the complete MGWR outputs.

The script regenerates coefficient maps, MGWR diagnostics, EPA L3 zoning tables,
and zoning figures for the three retained resilience dimensions.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")
os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


BASE = Path(
    "/path/to/google-drive"
    "/共享云端硬盘/Zola Research Storage3/WesternUS_Fire_Ecology_Project_2026-06-15"
    "/analysis_and_docs/US_Fire_and_Ecology_Data/WUS_1km"
)
MGWR_ROOT = BASE / "mgwr_complete_sample_2026-06-24"
POINTS = BASE / "resilience_management_zoning_mgwr_spatialized_2026-05-28/spatialized_point_zones_full_candidate.parquet"
ECO_L3 = BASE.parent / "EPA_Ecoregions/us_eco_l3/us_eco_l3.shp"
OUT_DIR = BASE / "manuscript_complete_mgwr_assets_2026-06-30"
ZONING_DIR = BASE / "resilience_management_zoning_epa_l3_complete_mgwr_2026-06-30"

RESPONSE_DIRS = {
    "Resistance": MGWR_ROOT / "complete_sample_mgwr_20260624_113719",
    "IRI_good_pow2": MGWR_ROOT / "complete_sample_mgwr_IRI_good_pow2_20260625_1216",
    "STAB_good_pow2": MGWR_ROOT / "complete_sample_mgwr_STAB_good_pow2_20260625_1216",
}
RESPONSE_LABELS = {
    "Resistance": "Resistance",
    "IRI_good_pow2": "IRI",
    "STAB_good_pow2": "STAB",
}
RESPONSE_FILE_LABELS = {
    "Resistance": "Resistance",
    "IRI_good_pow2": "IRI",
    "STAB_good_pow2": "STAB",
}

PREDICTOR_LABELS = {
    "TS_elev_m_z": "Elevation",
    "TS_slope_deg_z": "Slope",
    "TS_SOC_0_30cm_z": "Soil organic carbon",
    "FS_TCC_t0_z": "Tree canopy cover",
    "FS_CBH_t0agg_z": "Canopy base height",
    "HUM_roaddens_r5km_z": "Road density",
    "HUM_traildens_r10km_z": "Trail density",
    "HUM_viirs_near_t0_log_z": "Nighttime light",
    "HUM_imperv_near_t0_z": "Imperviousness",
    "CLIM_pr_sum_pre_z": "Pre-fire precipitation",
    "CLIM_tmmn_mean_pre_z": "Pre-fire minimum temp.",
}
DRIVER_POINT_COLS = {
    "TS_SOC_0_30cm_z": "TS_SOC_0_30cm_clean_z",
}
DIMENSIONS = {
    "forest_structure": ["FS_TCC_t0_z", "FS_CBH_t0agg_z"],
    "climate_constraint": ["CLIM_pr_sum_pre_z", "CLIM_tmmn_mean_pre_z"],
    "human_pressure": [
        "HUM_roaddens_r5km_z",
        "HUM_traildens_r10km_z",
        "HUM_viirs_near_t0_log_z",
        "HUM_imperv_near_t0_z",
    ],
}
ALL_DRIVERS = [driver for drivers in DIMENSIONS.values() for driver in drivers]

MIN_POINTS_PER_ECOREGION = 80
IDW_K = 8
IDW_POWER = 2.0

ZONE_ORDER = [
    "High-resilience conservation zone",
    "Low-resilience restoration priority zone",
    "Climate-dominated recovery constraint zone",
    "Structure-dominated resilience zone",
    "Human-pressure affected zone",
    "Mixed-control transition zone",
]
MECHANISM_ORDER = [
    "Climate-dominated recovery constraint zone",
    "Structure-dominated resilience zone",
    "Human-pressure affected zone",
    "Mixed-control transition zone",
]
ZONE_COLORS = {
    "High-resilience conservation zone": "#00722e",
    "Low-resilience restoration priority zone": "#8e3b8f",
    "Climate-dominated recovery constraint zone": "#3393bd",
    "Structure-dominated resilience zone": "#36a557",
    "Human-pressure affected zone": "#cf6618",
    "Mixed-control transition zone": "#9c9c9c",
}
SHORT_LABELS = {
    "High-resilience conservation zone": "High-resilience conservation",
    "Low-resilience restoration priority zone": "Low-resilience restoration priority",
    "Climate-dominated recovery constraint zone": "Climate-dominated recovery constraint",
    "Structure-dominated resilience zone": "Structure-dominated resilience",
    "Human-pressure affected zone": "Human-pressure affected",
    "Mixed-control transition zone": "Mixed-control transition",
}
SCORE_COLUMNS = [
    "forest_structure_support",
    "climate_constraint_score",
    "human_pressure_score",
]


def rank01(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True).clip(0, 1)


def quartile_level(values: pd.Series) -> pd.Series:
    ranks = pd.to_numeric(values, errors="coerce").rank(method="average", pct=True)
    return pd.cut(
        ranks,
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=[1, 2, 3, 4],
        include_lowest=True,
    ).astype(int)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def mode_or_blank(values: pd.Series) -> str:
    mode = values.mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else ""


def score_class(value: float, low: float, high: float) -> str:
    if value <= low:
        return "low"
    if value >= high:
        return "high"
    return "medium"


def response_std_name(response: str) -> str:
    return RESPONSE_LABELS[response]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ZONING_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bandwidths(response: str) -> pd.Series:
    bw = pd.read_csv(RESPONSE_DIRS[response] / "mgwr_complete_bandwidths.csv")
    return bw.set_index("term")["bandwidth"]


def compute_knn_moran(coords: np.ndarray, values: np.ndarray, k: int = 8) -> float:
    coords = np.asarray(coords, dtype=float)
    z = np.asarray(values, dtype=float)
    z = z - np.nanmean(z)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(coords)
    _, inds = nn.kneighbors(coords, return_distance=True)
    nbrs = inds[:, 1:]
    lag = z[nbrs].mean(axis=1)
    denom = np.nansum(z * z)
    if denom == 0:
        return float("nan")
    return float(np.nansum(z * lag) / denom)


def build_mgwr_diagnostics() -> pd.DataFrame:
    rows = []
    for response, label in RESPONSE_LABELS.items():
        metrics = load_json(RESPONSE_DIRS[response] / "mgwr_complete_metrics.json")
        residuals = pd.read_parquet(RESPONSE_DIRS[response] / "mgwr_complete_residuals.parquet")
        moran_i = compute_knn_moran(
            residuals[["x", "y"]].to_numpy(),
            residuals["residual"].to_numpy(),
            k=8,
        )
        rows.append(
            {
                "response": label,
                "rows_used": int(metrics["rows_used"]),
                "r2": float(metrics["r2"]),
                "rmse": float(metrics["rmse"]),
                "mae": float(metrics["mean_abs_residual"]),
                "bias_observed_minus_predicted": float(metrics["bias_observed_minus_predicted"]),
                "residual_moran_i_8nn": moran_i,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "complete_mgwr_model_performance.csv", index=False)
    return out


def plot_coefficient_maps() -> None:
    panel_letters = list("abcdefghijkl")
    for response, label in RESPONSE_LABELS.items():
        coef = pd.read_parquet(RESPONSE_DIRS[response] / "mgwr_complete_coefficients.parquet")
        bandwidths = load_bandwidths(response)
        predictors = list(PREDICTOR_LABELS)
        fig, axes = plt.subplots(3, 4, figsize=(14.6, 10.4), constrained_layout=True)
        axes_flat = axes.ravel()
        for idx, predictor in enumerate(predictors):
            ax = axes_flat[idx]
            vals = pd.to_numeric(coef[predictor], errors="coerce").to_numpy()
            finite = np.isfinite(vals)
            vmax = float(np.nanpercentile(np.abs(vals[finite]), 98)) if finite.any() else 1.0
            vmax = max(vmax, 1e-6)
            sc = ax.scatter(
                coef["x"],
                coef["y"],
                c=vals,
                s=0.32,
                marker="s",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                linewidths=0,
                rasterized=True,
            )
            ax.set_aspect("equal")
            ax.axis("off")
            bw = int(round(float(bandwidths.loc[predictor])))
            ax.set_title(
                f"({panel_letters[idx]}) {PREDICTOR_LABELS[predictor]}\nBW={bw:,}",
                loc="left",
                fontsize=8.6,
                fontweight="bold",
                pad=2,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.01)
            cbar.ax.tick_params(labelsize=6, length=2)
        for ax in axes_flat[len(predictors) :]:
            ax.axis("off")
        png = OUT_DIR / f"fig_complete_mgwr_coefficients_{label.lower()}.png"
        pdf = OUT_DIR / f"fig_complete_mgwr_coefficients_{label.lower()}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)


def load_points() -> pd.DataFrame:
    df = pd.read_parquet(POINTS).copy()
    required = [
        "x",
        "y",
        *RESPONSE_DIRS.keys(),
        *[DRIVER_POINT_COLS.get(driver, driver) for driver in ALL_DRIVERS],
    ]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    df["R_comp_no_t80"] = df[list(RESPONSE_DIRS.keys())].apply(rank01).mean(axis=1)
    return df


def merge_or_interpolate_coefficients(
    points: pd.DataFrame,
    response: str,
    coef_path: Path,
) -> tuple[pd.DataFrame, dict]:
    coef = pd.read_parquet(coef_path)
    missing = [driver for driver in ALL_DRIVERS if driver not in coef.columns]
    if missing:
        raise ValueError(f"{response} coefficient table is missing drivers: {missing}")

    keep = ["x", "y", *ALL_DRIVERS]
    merged = points[["x", "y"]].merge(coef[keep], on=["x", "y"], how="left", sort=False)
    out = pd.DataFrame(index=points.index)
    missing_mask = merged[ALL_DRIVERS].isna().any(axis=1)

    if missing_mask.any():
        sample_coords = coef[["x", "y"]].to_numpy(dtype=float)
        target_coords = points.loc[missing_mask, ["x", "y"]].to_numpy(dtype=float)
        nn = NearestNeighbors(n_neighbors=IDW_K, algorithm="ball_tree")
        nn.fit(sample_coords)
        dists, inds = nn.kneighbors(target_coords, return_distance=True)
        safe_dists = np.maximum(dists, 1.0)
        weights = 1.0 / np.power(safe_dists, IDW_POWER)
        weights = weights / weights.sum(axis=1, keepdims=True)
        for driver in ALL_DRIVERS:
            vals = merged[driver].to_numpy(dtype=float)
            beta = coef[driver].to_numpy(dtype=float)
            vals[missing_mask.to_numpy()] = np.sum(beta[inds] * weights, axis=1)
            out[f"beta_{response}_{driver}"] = vals
        nearest_stats = {
            "nearest_distance_median_m": float(np.median(dists[:, 0])),
            "nearest_distance_p95_m": float(np.percentile(dists[:, 0], 95)),
            "nearest_distance_max_m": float(np.max(dists[:, 0])),
        }
    else:
        for driver in ALL_DRIVERS:
            out[f"beta_{response}_{driver}"] = merged[driver].to_numpy(dtype=float)
        nearest_stats = {
            "nearest_distance_median_m": 0.0,
            "nearest_distance_p95_m": 0.0,
            "nearest_distance_max_m": 0.0,
        }

    meta = {
        "coef_path": str(coef_path),
        "n_coef_points": int(len(coef)),
        "n_target_points": int(len(points)),
        "n_exact_coordinate_matches": int((~missing_mask).sum()),
        **nearest_stats,
    }
    return out, meta


def compute_response_scores(points: pd.DataFrame, responses: dict[str, Path]) -> tuple[pd.DataFrame, dict]:
    out = points.copy()
    response_stds = {}
    merge_meta = {}

    for response, coef_path in responses.items():
        beta_df, meta = merge_or_interpolate_coefficients(out, response, coef_path)
        out = pd.concat([out, beta_df], axis=1)
        merge_meta[response] = meta
        response_stds[response] = float(out[response].std(ddof=1))

        for driver in ALL_DRIVERS:
            point_col = DRIVER_POINT_COLS.get(driver, driver)
            effect = out[f"beta_{response}_{driver}"] * out[point_col] / response_stds[response]
            out[f"effect_{response}_{driver}"] = effect
            out[f"support_{response}_{driver}"] = np.clip(effect, 0, None)
            out[f"constraint_{response}_{driver}"] = np.clip(-effect, 0, None)

        forest = DIMENSIONS["forest_structure"]
        climate = DIMENSIONS["climate_constraint"]
        human = DIMENSIONS["human_pressure"]
        out[f"{response}_forest_structure_support"] = out[[f"support_{response}_{c}" for c in forest]].sum(axis=1)
        out[f"{response}_forest_structure_constraint"] = out[[f"constraint_{response}_{c}" for c in forest]].sum(axis=1)
        out[f"{response}_forest_structure_net"] = out[[f"effect_{response}_{c}" for c in forest]].sum(axis=1)
        out[f"{response}_climate_constraint_score"] = out[[f"constraint_{response}_{c}" for c in climate]].sum(axis=1)
        out[f"{response}_climate_net"] = out[[f"effect_{response}_{c}" for c in climate]].sum(axis=1)
        out[f"{response}_human_pressure_score"] = out[[f"constraint_{response}_{c}" for c in human]].sum(axis=1)
        out[f"{response}_human_net"] = out[[f"effect_{response}_{c}" for c in human]].sum(axis=1)

    for name in [
        "forest_structure_support",
        "forest_structure_constraint",
        "forest_structure_net",
        "climate_constraint_score",
        "climate_net",
        "human_pressure_score",
        "human_net",
    ]:
        cols = [f"{response}_{name}" for response in responses]
        out[name] = out[cols].mean(axis=1)

    out["max_constraint_score"] = out[
        ["forest_structure_constraint", "climate_constraint_score", "human_pressure_score"]
    ].max(axis=1)
    out["dominant_constraint_dimension"] = out[
        ["forest_structure_constraint", "climate_constraint_score", "human_pressure_score"]
    ].idxmax(axis=1).map(
        {
            "forest_structure_constraint": "forest_structure",
            "climate_constraint_score": "climate_constraint",
            "human_pressure_score": "human_pressure",
        }
    )
    out["dominant_effect_dimension"] = pd.DataFrame(
        {
            "forest_structure": out["forest_structure_net"].abs(),
            "climate_constraint": out["climate_net"].abs(),
            "human_pressure": out["human_net"].abs(),
        }
    ).idxmax(axis=1)

    meta = {
        "response_stds_used_for_effect_standardization": response_stds,
        "coefficient_join": merge_meta,
    }
    return out, meta


def load_l3() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(ECO_L3).to_crs("EPSG:5070")
    keep = [
        "US_L3CODE",
        "US_L3NAME",
        "NA_L3CODE",
        "NA_L3NAME",
        "NA_L2CODE",
        "NA_L2NAME",
        "NA_L1CODE",
        "NA_L1NAME",
        "geometry",
    ]
    dissolved = gdf[keep].dissolve(
        by="US_L3CODE",
        aggfunc="first",
        as_index=False,
    )
    dissolved["eco_id"] = dissolved["US_L3CODE"].astype(str) + " " + dissolved["US_L3NAME"].astype(str)
    return dissolved


def spatial_join(points: pd.DataFrame, ecoregions: gpd.GeoDataFrame) -> pd.DataFrame:
    pts = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy(points["x"], points["y"]), crs="EPSG:5070")
    join_cols = [
        "eco_id",
        "US_L3CODE",
        "US_L3NAME",
        "NA_L3CODE",
        "NA_L3NAME",
        "NA_L2CODE",
        "NA_L2NAME",
        "NA_L1CODE",
        "NA_L1NAME",
        "geometry",
    ]
    joined = gpd.sjoin(pts, ecoregions[join_cols], how="left", predicate="within")
    unmatched = joined["eco_id"].isna()
    if unmatched.any():
        nearest = gpd.sjoin_nearest(
            pts.loc[unmatched, pts.columns],
            ecoregions[join_cols],
            how="left",
            distance_col="nearest_l3_distance_m",
        )
        for col in join_cols:
            if col != "geometry":
                joined.loc[unmatched, col] = nearest[col].to_numpy()
        joined.loc[unmatched, "nearest_l3_distance_m"] = nearest["nearest_l3_distance_m"].to_numpy()
    if "nearest_l3_distance_m" not in joined.columns:
        joined["nearest_l3_distance_m"] = 0.0
    joined["nearest_l3_distance_m"] = joined["nearest_l3_distance_m"].fillna(0.0)
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))


def assign_zones(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mechanism = np.full(len(out), "Mixed-control transition zone", dtype=object)
    climate_dom = (
        (out["climate_level"] >= 3)
        & (out["climate_constraint_score"] >= out["forest_structure_constraint"])
        & (out["climate_constraint_score"] >= out["human_pressure_score"])
    )
    human_dom = (
        (out["human_level"] >= 3)
        & (out["human_pressure_score"] >= out["forest_structure_constraint"])
        & (out["human_pressure_score"] >= out["climate_constraint_score"])
    )
    structure_constraint_q75 = out["forest_structure_constraint"].quantile(0.75)
    structure_competitor = out[["climate_constraint_score", "human_pressure_score"]].max(axis=1)
    structure_constraint_dom = (
        (out["forest_level"] >= 2)
        & (out["forest_structure_constraint"] >= structure_constraint_q75)
        & (out["forest_structure_constraint"] >= 3.0 * structure_competitor)
    )
    structure_dom = (
        (
            (out["forest_level"] >= 3)
            & (out["forest_structure_support"] >= out["climate_constraint_score"])
            & (out["forest_structure_support"] >= out["human_pressure_score"])
        )
        | structure_constraint_dom
    )
    mechanism[climate_dom] = "Climate-dominated recovery constraint zone"
    mechanism[human_dom] = "Human-pressure affected zone"
    mechanism[structure_dom] = "Structure-dominated resilience zone"
    out["mechanism_zone"] = pd.Categorical(mechanism, categories=ZONE_ORDER, ordered=True)

    r25, r75 = out["R_comp_no_t80"].quantile([0.25, 0.75])
    final = mechanism.copy()
    final[out["R_comp_no_t80"] <= r25] = "Low-resilience restoration priority zone"
    final[out["R_comp_no_t80"] >= r75] = "High-resilience conservation zone"
    out["management_zone"] = pd.Categorical(final, categories=ZONE_ORDER, ordered=True)
    out["low_priority_subtype"] = np.where(
        out["management_zone"].astype(str).eq("Low-resilience restoration priority zone"),
        out["dominant_constraint_dimension"],
        "",
    )
    out["high_resilience_mechanism_subtype"] = np.where(
        out["management_zone"].astype(str).eq("High-resilience conservation zone"),
        out["mechanism_zone"].astype(str),
        "",
    )
    return out


def aggregate_l3(joined: pd.DataFrame, ecoregions: gpd.GeoDataFrame, responses: dict[str, Path]) -> gpd.GeoDataFrame:
    agg_spec = {
        "n_pixels": ("x", "size"),
        "x_mean": ("x", "mean"),
        "y_mean": ("y", "mean"),
        "Resistance": ("Resistance", "mean"),
        "IRI_good_pow2": ("IRI_good_pow2", "mean"),
        "STAB_good_pow2": ("STAB_good_pow2", "mean"),
        "R_comp_no_t80": ("R_comp_no_t80", "mean"),
        "forest_structure_support": ("forest_structure_support", "median"),
        "forest_structure_constraint": ("forest_structure_constraint", "median"),
        "climate_constraint_score": ("climate_constraint_score", "median"),
        "human_pressure_score": ("human_pressure_score", "median"),
        "forest_structure_net": ("forest_structure_net", "median"),
        "climate_net": ("climate_net", "median"),
        "human_net": ("human_net", "median"),
        "evt_group_mode": ("FS_EVT_group_class", mode_or_blank),
        "project_region_mode": ("region", mode_or_blank),
        "nearest_l3_distance_max_m": ("nearest_l3_distance_m", "max"),
    }
    for response in responses:
        for score in [
            "forest_structure_support",
            "forest_structure_constraint",
            "climate_constraint_score",
            "human_pressure_score",
        ]:
            agg_spec[f"{response}_{score}"] = (f"{response}_{score}", "median")

    grouped = (
        joined.groupby(
            [
                "eco_id",
                "US_L3CODE",
                "US_L3NAME",
                "NA_L3CODE",
                "NA_L3NAME",
                "NA_L2CODE",
                "NA_L2NAME",
                "NA_L1CODE",
                "NA_L1NAME",
            ],
            dropna=False,
        )
        .agg(**agg_spec)
        .reset_index()
    )
    grouped = grouped[grouped["n_pixels"] >= MIN_POINTS_PER_ECOREGION].copy()
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
    out = ecoregions.merge(
        zoned,
        on=[
            "eco_id",
            "US_L3CODE",
            "US_L3NAME",
            "NA_L3CODE",
            "NA_L3NAME",
            "NA_L2CODE",
            "NA_L2NAME",
            "NA_L1CODE",
            "NA_L1NAME",
        ],
    )
    return gpd.GeoDataFrame(out, geometry="geometry", crs=ecoregions.crs)


def summarize_zone_column(zoned: gpd.GeoDataFrame, zone_col: str, output_col: str) -> pd.DataFrame:
    thresholds = {
        col: zoned[col].quantile([1 / 3, 2 / 3]).to_numpy(dtype=float)
        for col in SCORE_COLUMNS
    }
    rows = []
    for zone_name in ZONE_ORDER:
        sub = zoned[zoned[zone_col].astype(str) == zone_name]
        if sub.empty:
            continue
        w = sub["n_pixels"]
        forest_mean = weighted_mean(sub["forest_structure_support"], w)
        climate_mean = weighted_mean(sub["climate_constraint_score"], w)
        human_mean = weighted_mean(sub["human_pressure_score"], w)
        row = {
            output_col: zone_name,
            "n_l3_ecoregions": int(len(sub)),
            "n_pixels": int(w.sum()),
            "pct_l3_ecoregions": 100 * len(sub) / len(zoned),
            "pct_pixels": 100 * w.sum() / zoned["n_pixels"].sum(),
            "weighted_R_comp_no_t80": weighted_mean(sub["R_comp_no_t80"], w),
            "weighted_Resistance": weighted_mean(sub["Resistance"], w),
            "weighted_IRI_good_pow2": weighted_mean(sub["IRI_good_pow2"], w),
            "weighted_STAB_good_pow2": weighted_mean(sub["STAB_good_pow2"], w),
            "weighted_forest_structure_support": forest_mean,
            "weighted_forest_structure_constraint": weighted_mean(sub["forest_structure_constraint"], w),
            "weighted_climate_constraint_score": climate_mean,
            "weighted_human_pressure_score": human_mean,
            "forest_score_class": score_class(forest_mean, *thresholds["forest_structure_support"]),
            "climate_score_class": score_class(climate_mean, *thresholds["climate_constraint_score"]),
            "human_score_class": score_class(human_mean, *thresholds["human_pressure_score"]),
            "dominant_l3_names": "; ".join(
                sub.sort_values("n_pixels", ascending=False)["US_L3NAME"].head(5).tolist()
            ),
        }
        if zone_col != "mechanism_zone" and "mechanism_zone" in sub:
            row["dominant_mechanism_zone"] = mode_or_blank(sub["mechanism_zone"].astype(str))
        rows.append(row)
    return pd.DataFrame(rows)


def draw_zone_map(ax, zones: gpd.GeoDataFrame, zone_col: str, order: list[str] | None = None) -> None:
    order = order or ZONE_ORDER
    zones.boundary.plot(ax=ax, color="#ffffff", linewidth=0.25)
    for zone_name in order:
        sub = zones[zones[zone_col].astype(str) == zone_name]
        if sub.empty:
            continue
        sub.plot(
            ax=ax,
            color=ZONE_COLORS[zone_name],
            edgecolor="#ffffff",
            linewidth=0.28,
            alpha=0.95,
        )
    ax.set_aspect("equal")
    ax.axis("off")


def add_zone_legend(ax, zones: gpd.GeoDataFrame, zone_col: str, order: list[str]) -> None:
    handles = [
        mpatches.Patch(color=ZONE_COLORS[z], label=SHORT_LABELS[z])
        for z in order
        if (zones[zone_col].astype(str) == z).any()
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=6.4,
        frameon=True,
        framealpha=0.9,
        borderpad=0.35,
        labelspacing=0.25,
    )


def annotate_barh(ax, bars, rows, xmax: float, fontsize: float, value_col: str = "weighted_R_comp_no_t80") -> None:
    for bar, (_, row) in zip(bars, rows):
        label = f"L3={int(row['n_l3_ecoregions'])}, R={row[value_col]:.2f}"
        width = float(bar.get_width())
        y = bar.get_y() + bar.get_height() / 2
        if width > xmax * 0.58:
            ax.text(
                max(width - xmax * 0.015, xmax * 0.02),
                y,
                label,
                va="center",
                ha="right",
                fontsize=fontsize,
                color="white",
                clip_on=True,
            )
        else:
            ax.text(
                width + xmax * 0.015,
                y,
                label,
                va="center",
                ha="left",
                fontsize=fontsize,
                color="black",
                clip_on=True,
            )


def save_figure(fig: plt.Figure, out_dir: Path, base_stem: str, descriptive_stem: str) -> None:
    fig.savefig(out_dir / f"{base_stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{base_stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{descriptive_stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out_dir / f"{descriptive_stem}.pdf", bbox_inches="tight")


def plot_management_panel(zones: gpd.GeoDataFrame, summary: pd.DataFrame, out_dir: Path, method_label: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    continuous_panels = [
        ("R_comp_no_t80", "(a)", "RdYlGn"),
        ("forest_structure_support", "(b)", "Greens"),
        ("climate_constraint_score", "(c)", "Blues"),
        ("human_pressure_score", "(d)", "Oranges"),
    ]
    for ax, (col, title, cmap) in zip(axes.ravel()[:4], continuous_panels):
        zones.plot(
            column=col,
            ax=ax,
            cmap=cmap,
            linewidth=0.25,
            edgecolor="#ffffff",
            legend=True,
            legend_kwds={"shrink": 0.74, "pad": 0.02},
        )
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        ax.set_aspect("equal")
        ax.axis("off")

    ax = axes.ravel()[4]
    draw_zone_map(ax, zones, "management_zone", ZONE_ORDER)
    ax.set_title("(e)", fontsize=11, fontweight="bold", loc="left")
    add_zone_legend(ax, zones, "management_zone", ZONE_ORDER)

    ax = axes.ravel()[5]
    plot_summary = summary.set_index("management_zone").reindex(ZONE_ORDER).dropna(subset=["n_l3_ecoregions"])
    y_pos = np.arange(len(plot_summary))
    bars = ax.barh(
        y_pos,
        plot_summary["pct_pixels"],
        color=[ZONE_COLORS[z] for z in plot_summary.index],
        height=0.72,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS[z] for z in plot_summary.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of candidate pixels in retained Level III ecoregions", fontsize=9)
    ax.set_title("(f)", fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.22)
    xmax = max(55, float(plot_summary["pct_pixels"].max()) + 16)
    ax.set_xlim(0, xmax)
    annotate_barh(ax, bars, plot_summary.iterrows(), xmax, fontsize=7.6)

    stem = f"fig_{method_label.lower().replace(' ', '_').replace('-', '_')}_management_zoning_panel"
    save_figure(fig, out_dir, stem, f"{method_label} Management Zoning")
    plt.close(fig)


def plot_mechanism_panel(zones: gpd.GeoDataFrame, summary: pd.DataFrame, out_dir: Path, method_label: str) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 7.4),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )
    ax = axes[0]
    draw_zone_map(ax, zones, "mechanism_zone", MECHANISM_ORDER)
    ax.set_title("(a)", fontsize=13, fontweight="bold", loc="left")
    handles = [
        mpatches.Patch(color=ZONE_COLORS[z], label=SHORT_LABELS[z])
        for z in MECHANISM_ORDER
        if (zones["mechanism_zone"].astype(str) == z).any()
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

    ax = axes[1]
    plot_summary = summary.set_index("mechanism_zone").reindex(MECHANISM_ORDER).dropna(subset=["n_l3_ecoregions"])
    y_pos = np.arange(len(plot_summary))
    bars = ax.barh(
        y_pos,
        plot_summary["pct_pixels"],
        color=[ZONE_COLORS[z] for z in plot_summary.index],
        height=0.72,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([SHORT_LABELS[z] for z in plot_summary.index], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of candidate pixels", fontsize=10)
    ax.set_title("(b)", fontsize=13, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.22)
    xmax = max(65, float(plot_summary["pct_pixels"].max()) + 16)
    ax.set_xlim(0, xmax)
    annotate_barh(ax, bars, plot_summary.iterrows(), xmax, fontsize=8.4)

    stem = f"fig_{method_label.lower().replace(' ', '_').replace('-', '_')}_mechanism_only_zones"
    save_figure(fig, out_dir, stem, f"{method_label} Mechanism Zones")
    plt.close(fig)


def write_zoning_outputs(method_label: str, out_dir: Path, zones: gpd.GeoDataFrame, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    management = summarize_zone_column(zones, "management_zone", "management_zone")
    mechanism = summarize_zone_column(zones, "mechanism_zone", "mechanism_zone")
    csv_cols = [c for c in zones.columns if c != "geometry"]
    zones[csv_cols].to_csv(out_dir / "epa_l3_multiresponse_management_zones.csv", index=False)
    zones.to_file(out_dir / "epa_l3_multiresponse_management_zones.gpkg", driver="GPKG")
    management.to_csv(out_dir / "epa_l3_multiresponse_zone_summary.csv", index=False)
    mechanism.to_csv(out_dir / "epa_l3_multiresponse_mechanism_zone_summary.csv", index=False)
    plot_management_panel(zones, management, out_dir, method_label)
    plot_mechanism_panel(zones, mechanism, out_dir, method_label)
    (out_dir / "zoning_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def compute_zoning_outputs() -> None:
    points = load_points()
    ecoregions = load_l3()
    response_paths = {
        response: RESPONSE_DIRS[response] / "mgwr_complete_coefficients.parquet"
        for response in RESPONSE_DIRS
    }

    scored, meta = compute_response_scores(points, response_paths)
    joined = spatial_join(scored, ecoregions)
    zones = aggregate_l3(joined, ecoregions, response_paths)
    write_zoning_outputs("Multi-Response", ZONING_DIR, zones, meta)

    for response, coef_path in response_paths.items():
        label = RESPONSE_FILE_LABELS[response]
        response_out = ZONING_DIR / f"{label.lower()}_only"
        scored_one, meta_one = compute_response_scores(points, {response: coef_path})
        joined_one = spatial_join(scored_one, ecoregions)
        zones_one = aggregate_l3(joined_one, ecoregions, {response: coef_path})
        write_zoning_outputs(f"{label}-Only", response_out, zones_one, meta_one)


def main() -> None:
    ensure_dirs()
    diagnostics = build_mgwr_diagnostics()
    print(diagnostics.to_string(index=False))
    plot_coefficient_maps()
    compute_zoning_outputs()
    print(f"Saved manuscript assets to: {OUT_DIR}")
    print(f"Saved zoning outputs to: {ZONING_DIR}")


if __name__ == "__main__":
    main()
