#!/usr/bin/env python3
"""Multi-response MGWR mechanism zoning by EPA Level III ecoregion.

This script upgrades the existing Resistance-centered zoning by using local
coefficient surfaces for Resistance, IRI_good_pow2, and STAB_good_pow2. For each
response, local driver effects are computed as beta * standardized predictor,
then divided by the response standard deviation so that the three response
dimensions have comparable weight. Positive and negative effects are separated
before averaging across responses.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")

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
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
POINTS = BASE / "resilience_management_zoning_mgwr_spatialized_2026-05-28/spatialized_point_zones_full_candidate.parquet"
ECO_L3 = BASE.parent / "EPA_Ecoregions/us_eco_l3/us_eco_l3.shp"
COEF_DIR = BASE / "mgwr_retained_response_coefficients_2026-05-29"
OUT_DIR = BASE / "resilience_management_zoning_epa_l3_multiresponse_2026-06-03"

RESPONSES = {
    "Resistance": COEF_DIR / "Resistance_mgwr_coefficients.parquet",
    "IRI_good_pow2": COEF_DIR / "IRI_good_pow2/mgwr_coefficients.parquet",
    "STAB_good_pow2": COEF_DIR / "STAB_good_pow2/mgwr_coefficients.parquet",
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
ZONE_COLORS = {
    "High-resilience conservation zone": "#00722e",
    "Low-resilience restoration priority zone": "#8e3b8f",
    "Climate-dominated recovery constraint zone": "#3393bd",
    "Structure-dominated resilience zone": "#36a557",
    "Human-pressure affected zone": "#cf6618",
    "Mixed-control transition zone": "#9c9c9c",
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


def load_points() -> pd.DataFrame:
    df = pd.read_parquet(POINTS).copy()
    required = ["x", "y", *RESPONSES.keys(), *ALL_DRIVERS]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).reset_index(drop=True)
    df["R_comp_no_t80"] = df[list(RESPONSES.keys())].apply(rank01).mean(axis=1)
    return df


def interpolate_response_coefficients(
    points: pd.DataFrame,
    response: str,
    coef_path: Path,
) -> tuple[pd.DataFrame, dict]:
    coef = pd.read_parquet(coef_path)
    missing = [driver for driver in ALL_DRIVERS if driver not in coef.columns]
    if missing:
        raise ValueError(f"{response} coefficient table is missing drivers: {missing}")

    sample_coords = coef[["x", "y"]].to_numpy(dtype=float)
    target_coords = points[["x", "y"]].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=IDW_K, algorithm="ball_tree")
    nn.fit(sample_coords)
    dists, inds = nn.kneighbors(target_coords, return_distance=True)

    safe_dists = np.maximum(dists, 1.0)
    weights = 1.0 / np.power(safe_dists, IDW_POWER)
    weights = weights / weights.sum(axis=1, keepdims=True)

    beta_out = pd.DataFrame(index=points.index)
    for driver in ALL_DRIVERS:
        beta = coef[driver].to_numpy(dtype=float)
        beta_out[f"beta_{response}_{driver}"] = np.sum(beta[inds] * weights, axis=1)

    meta = {
        "coef_path": str(coef_path),
        "n_coef_points": int(len(coef)),
        "idw_k": IDW_K,
        "idw_power": IDW_POWER,
        "nearest_distance_median_m": float(np.median(dists[:, 0])),
        "nearest_distance_p90_m": float(np.percentile(dists[:, 0], 90)),
        "nearest_distance_p95_m": float(np.percentile(dists[:, 0], 95)),
        "nearest_distance_max_m": float(np.max(dists[:, 0])),
    }
    return beta_out, meta


def compute_multiresponse_scores(points: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = points.copy()
    interpolation_meta = {}
    response_stds = {}

    for response, coef_path in RESPONSES.items():
        beta_df, meta = interpolate_response_coefficients(out, response, coef_path)
        out = pd.concat([out, beta_df], axis=1)
        interpolation_meta[response] = meta
        response_stds[response] = float(out[response].std(ddof=1))

        for driver in ALL_DRIVERS:
            effect = out[f"beta_{response}_{driver}"] * out[driver] / response_stds[response]
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
        cols = [f"{response}_{name}" for response in RESPONSES]
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
        "interpolation": interpolation_meta,
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


def aggregate_l3(joined: pd.DataFrame, ecoregions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
    for response in RESPONSES:
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
            "dominant_l3_names": "; ".join(sub.sort_values("n_pixels", ascending=False)["US_L3NAME"].head(5).tolist()),
        }
        if zone_col != "mechanism_zone" and "mechanism_zone" in sub:
            row["dominant_mechanism_zone"] = mode_or_blank(sub["mechanism_zone"].astype(str))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to_resistance_only(zoned: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_path = BASE / "resilience_management_zoning_epa_l3_no_t80_2026-05-29/epa_l3_resilience_management_zones_no_t80.csv"
    old = pd.read_csv(old_path)
    key_cols = [
        "eco_id",
        "US_L3CODE",
        "US_L3NAME",
        "NA_L3CODE",
        "NA_L3NAME",
        "NA_L2CODE",
        "NA_L2NAME",
        "NA_L1CODE",
        "NA_L1NAME",
    ]
    keep = [*key_cols, "management_zone", "mechanism_zone"]
    left = zoned.drop(columns="geometry").copy()
    right = old[keep].copy()
    for col in key_cols:
        left[col] = left[col].astype(str)
        right[col] = right[col].astype(str)
    compare = left.merge(
        right.rename(
            columns={
                "management_zone": "management_zone_resistance_only",
                "mechanism_zone": "mechanism_zone_resistance_only",
            }
        ),
        on=key_cols,
        how="left",
    )
    compare["management_zone_changed"] = (
        compare["management_zone"].astype(str) != compare["management_zone_resistance_only"].astype(str)
    )
    compare["mechanism_zone_changed"] = (
        compare["mechanism_zone"].astype(str) != compare["mechanism_zone_resistance_only"].astype(str)
    )
    management_transition = pd.crosstab(
        compare["management_zone_resistance_only"],
        compare["management_zone"],
        dropna=False,
    )
    mechanism_transition = pd.crosstab(
        compare["mechanism_zone_resistance_only"],
        compare["mechanism_zone"],
        dropna=False,
    )
    return compare, management_transition, mechanism_transition


def plot_l3(zoned: gpd.GeoDataFrame, summary: pd.DataFrame, mechanism_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    panels = [
        ("forest_structure_support", "Multi-response forest support", "Greens"),
        ("climate_constraint_score", "Multi-response climate constraint", "Blues"),
        ("human_pressure_score", "Multi-response human pressure", "Oranges"),
    ]
    for ax, (col, title, cmap) in zip(axes.ravel()[:3], panels):
        zoned.plot(column=col, ax=ax, cmap=cmap, linewidth=0.25, edgecolor="#f7f7f7", legend=True)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    for ax, zone_col, title in [
        (axes.ravel()[3], "mechanism_zone", "Mechanism zones"),
        (axes.ravel()[4], "management_zone", "Management zones"),
    ]:
        for zone_name in ZONE_ORDER:
            sub = zoned[zoned[zone_col].astype(str) == zone_name]
            if sub.empty:
                continue
            sub.plot(ax=ax, color=ZONE_COLORS[zone_name], linewidth=0.35, edgecolor="white", alpha=0.92)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    ax = axes.ravel()[5]
    plot_summary = summary.set_index("management_zone").reindex(ZONE_ORDER).dropna(subset=["n_l3_ecoregions"])
    y_pos = np.arange(len(plot_summary))
    ax.barh(y_pos, plot_summary["pct_pixels"], color=[ZONE_COLORS[z] for z in plot_summary.index])
    ax.set_yticks(y_pos)
    ax.set_yticklabels([z.replace(" zone", "") for z in plot_summary.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of retained candidate pixels")
    ax.set_title("Management-zone extent", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    for i, (_, row) in enumerate(plot_summary.iterrows()):
        ax.text(row["pct_pixels"] + 0.4, i, f"L3={int(row['n_l3_ecoregions'])}", va="center", fontsize=8)

    handles = [
        mpatches.Patch(color=ZONE_COLORS[z], label=z.replace(" zone", ""))
        for z in ZONE_ORDER
        if (zoned["management_zone"].astype(str) == z).any() or (zoned["mechanism_zone"].astype(str) == z).any()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Multi-response MGWR-informed zoning by EPA Level III ecoregion", fontsize=15, fontweight="bold")
    fig.savefig(OUT_DIR / "multiresponse_mgwr_zoning_epa_l3.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_notes(meta: dict) -> None:
    responses = list(meta["response_stds_used_for_effect_standardization"])
    response_text = ", ".join(responses)
    response_scope = (
        f"This output uses {response_text} MGWR coefficient surfaces to compute mechanism scores."
        if len(responses) > 1
        else f"This output uses the {response_text} MGWR coefficient surface to compute response-specific mechanism scores."
    )
    text = f"""# Multi-response MGWR-informed EPA L3 zoning

{response_scope}

For response r, driver j, and pixel i:

`e_ijr = beta_ijr * z_ij / sd(y_r)`

The division by `sd(y_r)` standardizes local effects because the retained MGWR
runs were fit on the original response scales rather than z-scored responses.

Positive and negative evidence are separated before averaging across the
selected response set. For one-response sensitivity runs, this mean is simply
that response's positive or negative evidence:

`support_ij = mean_r(max(e_ijr, 0))`

`constraint_ij = mean_r(max(-e_ijr, 0))`

Dimension scores are then summed by driver group:

- Forest structure support: TCC and CBH positive evidence.
- Forest structure constraint: TCC and CBH negative evidence.
- Climate constraint: precipitation and minimum temperature negative evidence.
- Human pressure: roads, trails, VIIRS nighttime light, and imperviousness negative evidence.

The final EPA L3 workflow matches the no-T80 zoning logic: point scores are
joined to EPA Level III ecoregions, ecoregions with fewer than
{MIN_POINTS_PER_ECOREGION} candidate pixels are omitted, mechanism scores are
aggregated by median, and top/bottom quartiles of `R_comp_no_t80` overlay the
mechanism zones as conservation/restoration-priority classes.

## Metadata

```json
{json.dumps(meta, indent=2)}
```
"""
    (OUT_DIR / "multiresponse_zoning_notes.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points = load_points()
    scored, score_meta = compute_multiresponse_scores(points)
    ecoregions = load_l3()
    joined = spatial_join(scored, ecoregions)
    raw_counts = joined.groupby("eco_id", dropna=False).size()
    zoned = aggregate_l3(joined, ecoregions)
    summary = summarize_zone_column(zoned, "management_zone", "management_zone")
    mechanism_summary = summarize_zone_column(zoned, "mechanism_zone", "mechanism_zone")
    compare, management_transition, mechanism_transition = compare_to_resistance_only(zoned)

    point_keep = [
        "pixel_id",
        "row",
        "col",
        "x",
        "y",
        "Resistance",
        "IRI_good_pow2",
        "STAB_good_pow2",
        "R_comp_no_t80",
        "forest_structure_support",
        "forest_structure_constraint",
        "climate_constraint_score",
        "human_pressure_score",
        "dominant_constraint_dimension",
        "dominant_effect_dimension",
        "region",
        "FS_EVT_group_class",
    ]
    point_keep = [c for c in point_keep if c in scored.columns]
    scored[point_keep].to_parquet(OUT_DIR / "multiresponse_point_mechanism_scores.parquet", index=False)
    joined.drop(columns=[c for c in joined.columns if c.startswith("beta_")], errors="ignore").to_parquet(
        OUT_DIR / "points_with_epa_l3_assignment_multiresponse.parquet",
        index=False,
    )
    zoned.to_file(OUT_DIR / "epa_l3_multiresponse_management_zones.gpkg", driver="GPKG")
    zoned.drop(columns="geometry").to_csv(OUT_DIR / "epa_l3_multiresponse_management_zones.csv", index=False)
    summary.to_csv(OUT_DIR / "epa_l3_multiresponse_zone_summary.csv", index=False)
    mechanism_summary.to_csv(OUT_DIR / "epa_l3_multiresponse_mechanism_zone_summary.csv", index=False)
    compare.to_csv(OUT_DIR / "comparison_vs_resistance_only_no_t80.csv", index=False)
    management_transition.to_csv(OUT_DIR / "management_zone_transition_vs_resistance_only.csv")
    mechanism_transition.to_csv(OUT_DIR / "mechanism_zone_transition_vs_resistance_only.csv")
    plot_l3(zoned, summary, mechanism_summary)

    meta = {
        "n_input_points": int(len(points)),
        "n_joined_points": int(joined["eco_id"].notna().sum()),
        "n_total_l3_in_file": int(len(ecoregions)),
        "n_l3_with_points_before_filter": int((raw_counts > 0).sum()),
        "n_retained_l3": int(len(zoned)),
        "n_dropped_l3_below_minimum_support": int((raw_counts < MIN_POINTS_PER_ECOREGION).sum()),
        "min_points_per_ecoregion": MIN_POINTS_PER_ECOREGION,
        "management_zone_counts_l3": zoned["management_zone"].astype(str).value_counts().to_dict(),
        "mechanism_zone_counts_l3": zoned["mechanism_zone"].astype(str).value_counts().to_dict(),
        "management_zone_counts_pixels": zoned.groupby(zoned["management_zone"].astype(str))["n_pixels"]
        .sum()
        .astype(int)
        .to_dict(),
        "mechanism_zone_counts_pixels": zoned.groupby(zoned["mechanism_zone"].astype(str))["n_pixels"]
        .sum()
        .astype(int)
        .to_dict(),
        "management_zone_changed_l3_vs_resistance_only": int(compare["management_zone_changed"].sum()),
        "mechanism_zone_changed_l3_vs_resistance_only": int(compare["mechanism_zone_changed"].sum()),
        "output_dir": str(OUT_DIR),
        **score_meta,
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_notes(meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
