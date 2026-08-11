#!/usr/bin/env python3
"""Build MGWR constraint-based management zoning, framework v3.

This script reclassifies the complete MGWR EPA Level III ecoregion outputs into
the constraint-focused paper framework:

1. Use realized local effects, beta * standardized X / sd(Y), already stored in
   the complete MGWR zoning table as response-comparable mechanism scores.
2. Use negative realized effects only to define dominant recovery constraints.
3. Use the equal-weight composite of Resistance, IRI_good_pow2, and
   STAB_good_pow2 to define restoration/conservation priority.
4. Run dominance-threshold sensitivity at the 70th, 75th, and 80th percentiles.
5. Compare equal-weight R_comp with a PCA-based resilience composite.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "westernus_postfire_mplconfig")
)

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


MAIN_Q = 0.75
SENSITIVITY_QS = [0.70, 0.75, 0.80]
FOREST_DIRECTION_RATIO = 1.20

RESPONSE_COLS = ["Resistance", "IRI_good_pow2", "STAB_good_pow2"]

MECHANISM_COLORS = {
    "Climate-dominated constraint": "#2b8cbe",
    "Human-pressure dominated": "#d95f0e",
    "Structure-dominated constraint": "#b35806",
    "Mixed-control transition": "#969696",
}

MANAGEMENT_COLORS = {
    "High-resilience conservation priority": "#006d2c",
    "Low-resilience restoration priority": "#8e3b8f",
    "Climate-constrained management": "#2b8cbe",
    "Human-pressure management": "#d95f0e",
    "Structure-constraint management": "#b35806",
    "Mixed-control transition": "#969696",
}

CLUSTER_COLORS = {
    "Climate-constrained cluster": "#2b8cbe",
    "High structure-constraint cluster": "#b35806",
    "Moderate structure-constraint cluster": "#dfc27d",
    "Low-intensity mixed cluster": "#969696",
    "Human-pressure cluster": "#d95f0e",
    "Structure-support cluster": "#238b45",
    "Structure-bidirectional cluster": "#756bb1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-gpkg",
        required=True,
        help="EPA Level III multiresponse zoning GeoPackage from compute_multiresponse_zoning.py.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def checked_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def read_input(input_gpkg: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(input_gpkg)
    required = [
        "US_L3NAME",
        "n_pixels",
        *RESPONSE_COLS,
        "R_comp_no_t80",
        "forest_structure_support",
        "forest_structure_constraint",
        "climate_constraint_score",
        "human_pressure_score",
    ]
    missing = [col for col in required if col not in gdf.columns]
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")
    gdf = gdf.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    return gdf


def rank01(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True).clip(0, 1)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def weighted_pct(values: pd.Series, weights: pd.Series, label: str) -> float:
    mask = values.astype(str).eq(label)
    total = float(pd.to_numeric(weights, errors="coerce").fillna(0).sum())
    if total <= 0:
        return float("nan")
    return 100.0 * float(pd.to_numeric(weights[mask], errors="coerce").fillna(0).sum()) / total


def classify_forest_direction(row: pd.Series) -> str:
    support = float(row["forest_structure_support"])
    constraint = float(row["forest_structure_constraint"])
    if support >= constraint * FOREST_DIRECTION_RATIO:
        return "support"
    if constraint >= support * FOREST_DIRECTION_RATIO:
        return "constraint"
    return "bidirectional"


def prepare_scores(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["R_comp_equal"] = out[RESPONSE_COLS].apply(rank01).mean(axis=1)
    out["forest_structure_influence"] = out["forest_structure_constraint"]
    out["forest_structure_abs_net"] = (
        out["forest_structure_support"] - out["forest_structure_constraint"]
    ).abs()
    out["forest_structure_direction"] = out.apply(classify_forest_direction, axis=1)

    score_table = pd.DataFrame(
        {
            "structure": out["forest_structure_constraint"],
            "climate": out["climate_constraint_score"],
            "human": out["human_pressure_score"],
        },
        index=out.index,
    )
    out["dominant_mechanism_raw"] = score_table.idxmax(axis=1)
    out["dominant_score"] = score_table.max(axis=1)
    sorted_scores = np.sort(score_table.to_numpy(dtype=float), axis=1)
    out["dominance_margin"] = sorted_scores[:, -1] - sorted_scores[:, -2]
    out["dominance_ratio"] = np.divide(
        sorted_scores[:, -1],
        sorted_scores[:, -2],
        out=np.full(len(out), np.nan),
        where=sorted_scores[:, -2] > 0,
    )
    return out


def mechanism_label(row: pd.Series) -> str:
    raw = str(row["dominant_mechanism_raw"])
    if raw == "climate":
        return "Climate-dominated constraint"
    if raw == "human":
        return "Human-pressure dominated"
    return "Structure-dominated constraint"


def assign_for_threshold(gdf: gpd.GeoDataFrame, q: float) -> tuple[gpd.GeoDataFrame, float, float, float]:
    out = gdf.copy()
    threshold = float(out["dominant_score"].quantile(q))
    r_low = float(out["R_comp_equal"].quantile(0.25))
    r_high = float(out["R_comp_equal"].quantile(0.75))

    labels = []
    for _, row in out.iterrows():
        if float(row["dominant_score"]) < threshold:
            labels.append("Mixed-control transition")
        else:
            labels.append(mechanism_label(row))
    out[f"mechanism_zone_q{int(q * 100)}"] = labels

    priority = np.full(len(out), "Moderate", dtype=object)
    priority[out["R_comp_equal"] <= r_low] = "Low"
    priority[out["R_comp_equal"] >= r_high] = "High"
    out["resilience_priority_class"] = priority

    management = []
    mech_col = f"mechanism_zone_q{int(q * 100)}"
    for _, row in out.iterrows():
        if row["resilience_priority_class"] == "High":
            management.append("High-resilience conservation priority")
        elif row["resilience_priority_class"] == "Low":
            management.append("Low-resilience restoration priority")
        else:
            mech = str(row[mech_col])
            if mech == "Climate-dominated constraint":
                management.append("Climate-constrained management")
            elif mech == "Human-pressure dominated":
                management.append("Human-pressure management")
            elif mech == "Structure-dominated constraint":
                management.append("Structure-constraint management")
            else:
                management.append("Mixed-control transition")
    out[f"management_zone_q{int(q * 100)}"] = management
    return out, threshold, r_low, r_high


def summarize_zones(gdf: gpd.GeoDataFrame, zone_col: str) -> pd.DataFrame:
    rows = []
    total_pixels = float(gdf["n_pixels"].sum())
    for zone, sub in gdf.groupby(zone_col, dropna=False):
        w = sub["n_pixels"]
        rows.append(
            {
                "zone": str(zone),
                "n_l3_ecoregions": int(len(sub)),
                "n_pixels": int(w.sum()),
                "pct_l3_ecoregions": 100.0 * len(sub) / len(gdf),
                "pct_pixels": 100.0 * float(w.sum()) / total_pixels,
                "weighted_R_comp_equal": weighted_mean(sub["R_comp_equal"], w),
                "weighted_Resistance": weighted_mean(sub["Resistance"], w),
                "weighted_IRI_good_pow2": weighted_mean(sub["IRI_good_pow2"], w),
                "weighted_STAB_good_pow2": weighted_mean(sub["STAB_good_pow2"], w),
                "weighted_forest_structure_influence": weighted_mean(
                    sub["forest_structure_influence"], w
                ),
                "weighted_forest_structure_support": weighted_mean(
                    sub["forest_structure_support"], w
                ),
                "weighted_forest_structure_constraint": weighted_mean(
                    sub["forest_structure_constraint"], w
                ),
                "weighted_climate_constraint_score": weighted_mean(
                    sub["climate_constraint_score"], w
                ),
                "weighted_human_pressure_score": weighted_mean(
                    sub["human_pressure_score"], w
                ),
                "largest_l3_ecoregions": "; ".join(
                    sub.sort_values("n_pixels", ascending=False)["US_L3NAME"].head(5).astype(str)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_pixels", ascending=False)


def make_sensitivity_summary(
    base: gpd.GeoDataFrame, assigned: dict[float, gpd.GeoDataFrame], thresholds: dict[float, float]
) -> pd.DataFrame:
    rows = []
    for q, gdf in assigned.items():
        mech_col = f"mechanism_zone_q{int(q * 100)}"
        mgmt_col = f"management_zone_q{int(q * 100)}"
        rows.append(
            {
                "dominance_quantile": q,
                "dominant_score_threshold": thresholds[q],
                "mixed_pct_l3_mechanism": 100.0
                * float(gdf[mech_col].astype(str).eq("Mixed-control transition").mean()),
                "mixed_pct_pixels_mechanism": weighted_pct(
                    gdf[mech_col], gdf["n_pixels"], "Mixed-control transition"
                ),
                "mixed_pct_l3_management": 100.0
                * float(gdf[mgmt_col].astype(str).eq("Mixed-control transition").mean()),
                "mixed_pct_pixels_management": weighted_pct(
                    gdf[mgmt_col], gdf["n_pixels"], "Mixed-control transition"
                ),
                "climate_pct_pixels_mechanism": weighted_pct(
                    gdf[mech_col], gdf["n_pixels"], "Climate-dominated constraint"
                ),
                "human_pct_pixels_mechanism": weighted_pct(
                    gdf[mech_col], gdf["n_pixels"], "Human-pressure dominated"
                ),
                "structure_support_pct_pixels_mechanism": weighted_pct(
                    gdf[mech_col], gdf["n_pixels"], "Structure-dominated support"
                ),
                "structure_constraint_pct_pixels_mechanism": weighted_pct(
                    gdf[mech_col], gdf["n_pixels"], "Structure-dominated constraint"
                ),
                "structure_bidirectional_pct_pixels_mechanism": weighted_pct(
                    gdf[mech_col], gdf["n_pixels"], "Structure-dominated bidirectional"
                ),
            }
        )
    return pd.DataFrame(rows)


def make_pca_check(gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = gdf[RESPONSE_COLS].to_numpy(dtype=float)
    z = StandardScaler().fit_transform(values)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(z).ravel()
    equal = gdf["R_comp_equal"].to_numpy(dtype=float)
    if np.corrcoef(pc1, equal)[0, 1] < 0:
        pc1 = -pc1
    pc1_rank = pd.Series(pc1).rank(method="average", pct=True).to_numpy()

    q_equal_low, q_equal_high = np.quantile(equal, [0.25, 0.75])
    q_pc_low, q_pc_high = np.quantile(pc1_rank, [0.25, 0.75])
    equal_class = np.where(equal <= q_equal_low, "Low", np.where(equal >= q_equal_high, "High", "Moderate"))
    pc_class = np.where(pc1_rank <= q_pc_low, "Low", np.where(pc1_rank >= q_pc_high, "High", "Moderate"))

    detail = gdf[["US_L3NAME", "n_pixels", *RESPONSE_COLS, "R_comp_equal"]].copy()
    detail["R_comp_pca_rank"] = pc1_rank
    detail["equal_priority_class"] = equal_class
    detail["pca_priority_class"] = pc_class

    summary = pd.DataFrame(
        [
            {
                "pc1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
                "equal_vs_pca_rank_correlation": float(np.corrcoef(equal, pc1_rank)[0, 1]),
                "priority_class_agreement_pct_l3": 100.0 * float(np.mean(equal_class == pc_class)),
                "priority_class_agreement_pct_pixels": 100.0
                * float(
                    np.average(
                        equal_class == pc_class,
                        weights=pd.to_numeric(gdf["n_pixels"], errors="coerce").fillna(0),
                    )
                ),
                "pc1_loading_Resistance": float(pca.components_[0, 0]),
                "pc1_loading_IRI_good_pow2": float(pca.components_[0, 1]),
                "pc1_loading_STAB_good_pow2": float(pca.components_[0, 2]),
            }
        ]
    )
    return summary, detail


def assign_kmeans_clusters(gdf: gpd.GeoDataFrame, k: int = 4) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    out = gdf.copy()
    features = [
        "forest_structure_influence",
        "climate_constraint_score",
        "human_pressure_score",
    ]
    x = StandardScaler().fit_transform(out[features].to_numpy(dtype=float))
    km = KMeans(n_clusters=k, random_state=42, n_init=50)
    out["kmeans4_cluster_id"] = km.fit_predict(x)

    rows = []
    raw_centers = out.groupby("kmeans4_cluster_id")[features + [
        "forest_structure_support",
        "forest_structure_constraint",
        "dominant_score",
        "n_pixels",
    ]].agg(
        {
            "forest_structure_influence": "mean",
            "climate_constraint_score": "mean",
            "human_pressure_score": "mean",
            "forest_structure_support": "mean",
            "forest_structure_constraint": "mean",
            "dominant_score": "mean",
            "n_pixels": "sum",
        }
    )
    low_cluster = int(raw_centers["dominant_score"].idxmin())
    forest_clusters = raw_centers[
        (raw_centers["forest_structure_influence"] >= raw_centers["climate_constraint_score"])
        & (raw_centers["forest_structure_influence"] >= raw_centers["human_pressure_score"])
    ].index.tolist()
    high_forest_cluster = None
    if forest_clusters:
        high_forest_cluster = int(
            raw_centers.loc[forest_clusters, "forest_structure_influence"].idxmax()
        )

    label_map = {}
    for cid, row in raw_centers.iterrows():
        cid_int = int(cid)
        if cid_int == low_cluster:
            label = "Low-intensity mixed cluster"
        elif (
            row["climate_constraint_score"] >= row["forest_structure_influence"]
            and row["climate_constraint_score"] >= row["human_pressure_score"]
        ):
            label = "Climate-constrained cluster"
        elif (
            row["human_pressure_score"] >= row["forest_structure_influence"]
            and row["human_pressure_score"] >= row["climate_constraint_score"]
        ):
            label = "Human-pressure cluster"
        else:
            if high_forest_cluster is not None and cid_int == high_forest_cluster:
                label = "High structure-constraint cluster"
            else:
                label = "Moderate structure-constraint cluster"
        label_map[cid_int] = label

    out["kmeans4_mechanism_cluster"] = out["kmeans4_cluster_id"].map(label_map)

    total_pixels = float(out["n_pixels"].sum())
    for cid, sub in out.groupby("kmeans4_cluster_id"):
        w = sub["n_pixels"]
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_label": label_map[int(cid)],
                "n_l3_ecoregions": int(len(sub)),
                "n_pixels": int(w.sum()),
                "pct_pixels": 100.0 * float(w.sum()) / total_pixels,
                "mean_forest_structure_influence": float(
                    sub["forest_structure_influence"].mean()
                ),
                "mean_forest_structure_support": float(
                    sub["forest_structure_support"].mean()
                ),
                "mean_forest_structure_constraint": float(
                    sub["forest_structure_constraint"].mean()
                ),
                "mean_climate_constraint_score": float(
                    sub["climate_constraint_score"].mean()
                ),
                "mean_human_pressure_score": float(sub["human_pressure_score"].mean()),
                "weighted_R_comp_equal": weighted_mean(sub["R_comp_equal"], w),
                "largest_l3_ecoregions": "; ".join(
                    sub.sort_values("n_pixels", ascending=False)["US_L3NAME"].head(5).astype(str)
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values("pct_pixels", ascending=False)
    return out, summary


def save_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    colors: dict[str, str],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.boundary.plot(ax=ax, color="#ffffff", linewidth=0.25)
    for label, color in colors.items():
        sub = gdf[gdf[column].astype(str).eq(label)]
        if len(sub):
            sub.plot(ax=ax, color=color, edgecolor="#ffffff", linewidth=0.25)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, weight="bold")
    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color in colors.items()
        if gdf[column].astype(str).eq(label).any()
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_diagnostic_panel(
    gdf: gpd.GeoDataFrame,
    assigned: dict[float, gpd.GeoDataFrame],
    thresholds: dict[float, float],
    out_path: Path,
) -> None:
    main = assigned[MAIN_Q]
    mech_col = f"mechanism_zone_q{int(MAIN_Q * 100)}"
    mgmt_col = f"management_zone_q{int(MAIN_Q * 100)}"

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.ravel()

    ax = axes[0]
    for label, color in MECHANISM_COLORS.items():
        sub = main[main[mech_col].astype(str).eq(label)]
        if len(sub):
            sub.plot(ax=ax, color=color, edgecolor="#ffffff", linewidth=0.2)
    ax.set_axis_off()
    ax.set_title("A. Constraint zones (q75)", fontsize=12, weight="bold")

    ax = axes[1]
    for label, color in MANAGEMENT_COLORS.items():
        sub = main[main[mgmt_col].astype(str).eq(label)]
        if len(sub):
            sub.plot(ax=ax, color=color, edgecolor="#ffffff", linewidth=0.2)
    ax.set_axis_off()
    ax.set_title("B. Management zones with resilience priority", fontsize=12, weight="bold")

    ax = axes[2]
    ax.hist(gdf["dominant_score"], bins=10, color="#bdbdbd", edgecolor="#525252")
    for q, color in zip(SENSITIVITY_QS, ["#2b8cbe", "#756bb1", "#d95f0e"]):
        ax.axvline(thresholds[q], color=color, linewidth=2, label=f"q{int(q * 100)}")
    ax.set_xlabel("Dominant realized-effect score")
    ax.set_ylabel("EPA L3 ecoregions")
    ax.set_title("C. Dominance-score distribution", fontsize=12, weight="bold")
    ax.legend(frameon=False)

    ax = axes[3]
    sens = make_sensitivity_summary(gdf, assigned, thresholds)
    x = np.arange(len(sens))
    ax.bar(x - 0.18, sens["mixed_pct_pixels_mechanism"], width=0.36, color="#969696", label="Constraint map")
    ax.bar(x + 0.18, sens["mixed_pct_pixels_management"], width=0.36, color="#525252", label="Management map")
    ax.set_xticks(x, [f"q{int(q * 100)}" for q in sens["dominance_quantile"]])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mixed zone (% candidate pixels)")
    ax.set_title("D. Threshold sensitivity", fontsize=12, weight="bold")
    ax.legend(frameon=False)

    handles1 = [
        mpatches.Patch(color=color, label=label)
        for label, color in MECHANISM_COLORS.items()
        if main[mech_col].astype(str).eq(label).any()
    ]
    handles2 = [
        mpatches.Patch(color=color, label=label)
        for label, color in MANAGEMENT_COLORS.items()
        if main[mgmt_col].astype(str).eq(label).any()
    ]
    fig.legend(handles=handles1, loc="lower left", bbox_to_anchor=(0.02, -0.015), ncol=2, fontsize=8)
    fig.legend(handles=handles2, loc="lower right", bbox_to_anchor=(0.98, -0.015), ncol=2, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_crosstab(gdf: gpd.GeoDataFrame, q: float, out_path: Path) -> None:
    mech_col = f"mechanism_zone_q{int(q * 100)}"
    ct_l3 = pd.crosstab(gdf[mech_col], gdf["resilience_priority_class"], dropna=False)
    ct_pix = pd.pivot_table(
        gdf,
        index=mech_col,
        columns="resilience_priority_class",
        values="n_pixels",
        aggfunc="sum",
        fill_value=0,
    )
    ct_l3.to_csv(out_path.with_name(out_path.stem + "_l3_counts.csv"))
    ct_pix.to_csv(out_path.with_name(out_path.stem + "_pixel_counts.csv"))


def write_notes(
    gdf: gpd.GeoDataFrame,
    thresholds: dict[float, float],
    sens: pd.DataFrame,
    pca_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    main_mixed_mech = float(
        sens.loc[sens["dominance_quantile"].eq(MAIN_Q), "mixed_pct_pixels_mechanism"].iloc[0]
    )
    main_mixed_mgmt = float(
        sens.loc[sens["dominance_quantile"].eq(MAIN_Q), "mixed_pct_pixels_management"].iloc[0]
    )
    lines = [
        "# MGWR constraint zoning framework v3",
        "",
        "## Main choices",
        "",
        "- Effect definition: realized local effect, `beta_j,r * X_j / sd(Y_r)`. This means the scores represent realized local contribution/constraint intensity, not coefficient strength alone.",
        "- Main zoning is constraint-focused: dominance is based on negative realized effects for all mechanism groups.",
        "- Forest structure handling: support and constraint are retained separately for diagnostics, but dominance uses `forest_structure_constraint`, consistent with climate and human-pressure constraint scores.",
        "- Resilience priority: `R_comp_equal` is the equal-weight mean of ranked Resistance, IRI_good_pow2, and STAB_good_pow2.",
        "- Main dominance threshold: q75 of `dominant_score`; q70 and q80 are reported as sensitivity tests.",
        "- RF or MGWR-informed RF predictions are not used in this zoning run. The map is mechanism-based.",
        "",
        "## Key diagnostics",
        "",
        f"- EPA L3 ecoregions retained: {len(gdf)}.",
        f"- Candidate pixels represented: {int(gdf['n_pixels'].sum())}.",
        f"- q75 dominant-score threshold: {thresholds[MAIN_Q]:.4f}.",
        f"- q75 mixed-control share in mechanism map: {main_mixed_mech:.1f}% of candidate pixels.",
        f"- q75 mixed-control share after resilience-priority overlay: {main_mixed_mgmt:.1f}% of candidate pixels.",
        f"- PCA PC1 explained variance: {float(pca_summary['pc1_explained_variance_ratio'].iloc[0]):.3f}.",
        f"- Equal composite vs PCA rank correlation: {float(pca_summary['equal_vs_pca_rank_correlation'].iloc[0]):.3f}.",
        f"- Equal vs PCA priority-class agreement: {float(pca_summary['priority_class_agreement_pct_l3'].iloc[0]):.1f}% of L3 ecoregions.",
        f"- K-means four-cluster largest class: {cluster_summary.iloc[0]['cluster_label']} ({float(cluster_summary.iloc[0]['pct_pixels']):.1f}% of candidate pixels).",
        "",
        "## Interpretation guardrails",
        "",
        "- MGWR-derived zones should be described as statistical dominant-factor or realized-effect zones, not as causal proof.",
        "- If q75 leaves too much area in the mixed zone, use the sensitivity table to justify q70 or use the K-means cluster map as a data-driven constraint-space diagnostic.",
        "- The equal-weight resilience composite is transparent and interpretable; the PCA check is a sensitivity diagnostic rather than the main map.",
    ]
    (output_dir / "zoning_framework_v3_notes.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    input_gpkg = checked_file(args.input_gpkg, "Multiresponse zoning GeoPackage")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[constraint zoning 1/4] Loading: {input_gpkg}", flush=True)
    base = prepare_scores(read_input(input_gpkg))

    assigned = {}
    thresholds = {}
    r_thresholds = {}
    for q in SENSITIVITY_QS:
        gdf_q, threshold, r_low, r_high = assign_for_threshold(base, q)
        assigned[q] = gdf_q
        thresholds[q] = threshold
        r_thresholds[q] = {"r_low": r_low, "r_high": r_high}

        qtag = f"q{int(q * 100)}"
        csv_out = output_dir / f"mgwr_constraint_management_zones_{qtag}.csv"
        gpkg_out = output_dir / f"mgwr_constraint_management_zones_{qtag}.gpkg"
        gdf_save = gdf_q.copy()
        for col in gdf_save.select_dtypes(include=["category"]).columns:
            gdf_save[col] = gdf_save[col].astype(str)
        gdf_save.drop(columns="geometry").to_csv(csv_out, index=False)
        gdf_save.to_file(gpkg_out, driver="GPKG")

        mech_col = f"mechanism_zone_q{int(q * 100)}"
        mgmt_col = f"management_zone_q{int(q * 100)}"
        summarize_zones(gdf_q, mech_col).to_csv(
            output_dir / f"constraint_zone_summary_{qtag}.csv", index=False
        )
        summarize_zones(gdf_q, mgmt_col).to_csv(
            output_dir / f"management_zone_summary_{qtag}.csv", index=False
        )
        save_crosstab(gdf_q, q, output_dir / f"constraint_by_priority_crosstab_{qtag}.csv")

    print("[constraint zoning 2/4] Built threshold sensitivity outputs", flush=True)
    main_gdf = assigned[MAIN_Q]
    main_mech_col = f"mechanism_zone_q{int(MAIN_Q * 100)}"
    main_mgmt_col = f"management_zone_q{int(MAIN_Q * 100)}"
    save_map(
        main_gdf,
        main_mech_col,
        MECHANISM_COLORS,
        "MGWR constraint zones (q75 dominance threshold)",
        output_dir / "fig_mgwr_constraint_zones_q75",
    )
    save_map(
        main_gdf,
        main_mgmt_col,
        MANAGEMENT_COLORS,
        "MGWR constraint-based management zones with resilience priority",
        output_dir / "fig_mgwr_constraint_management_zones_q75",
    )
    save_diagnostic_panel(
        base,
        assigned,
        thresholds,
        output_dir / "fig_mgwr_constraint_zoning_framework_v3_diagnostics",
    )
    print("[constraint zoning 3/4] Built q75 maps and diagnostics", flush=True)

    sens = make_sensitivity_summary(base, assigned, thresholds)
    sens.to_csv(output_dir / "dominance_threshold_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {
                "dominance_quantile": q,
                "dominant_score_threshold": thresholds[q],
                **r_thresholds[q],
            }
            for q in SENSITIVITY_QS
        ]
    ).to_csv(output_dir / "threshold_values.csv", index=False)

    pca_summary, pca_detail = make_pca_check(base)
    pca_summary.to_csv(output_dir / "rcomp_equal_vs_pca_summary.csv", index=False)
    pca_detail.to_csv(output_dir / "rcomp_equal_vs_pca_by_ecoregion.csv", index=False)

    clustered, cluster_summary = assign_kmeans_clusters(base, k=4)
    clustered_save = clustered.copy()
    for col in clustered_save.select_dtypes(include=["category"]).columns:
        clustered_save[col] = clustered_save[col].astype(str)
    clustered_save.drop(columns="geometry").to_csv(
        output_dir / "kmeans4_constraint_clusters.csv", index=False
    )
    clustered_save.to_file(output_dir / "kmeans4_constraint_clusters.gpkg", driver="GPKG")
    cluster_summary.to_csv(output_dir / "kmeans4_constraint_cluster_summary.csv", index=False)
    save_map(
        clustered,
        "kmeans4_mechanism_cluster",
        CLUSTER_COLORS,
        "Data-driven constraint clusters (K-means, k=4)",
        output_dir / "fig_kmeans4_constraint_clusters",
    )

    metadata = {
        "input_gpkg": str(input_gpkg),
        "output_dir": str(output_dir),
        "effect_definition": "realized local effect: beta * standardized predictor / response standard deviation",
        "dominance_score": "max(forest_structure_constraint, climate_constraint_score, human_pressure_score)",
        "forest_dominance_score": "forest_structure_constraint",
        "forest_direction_ratio": FOREST_DIRECTION_RATIO,
        "resilience_composite": "equal-weight mean of within-ecoregion ranks for Resistance, IRI_good_pow2, and STAB_good_pow2",
        "main_dominance_quantile": MAIN_Q,
        "sensitivity_quantiles": SENSITIVITY_QS,
        "dominant_score_thresholds": thresholds,
        "resilience_priority_thresholds": r_thresholds[MAIN_Q],
        "n_l3_ecoregions": int(len(base)),
        "n_candidate_pixels": int(base["n_pixels"].sum()),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_notes(base, thresholds, sens, pca_summary, cluster_summary, output_dir)

    print(f"[constraint zoning 4/4] Wrote outputs to: {output_dir}")
    print(sens.to_string(index=False))
    print(pca_summary.to_string(index=False))
    print(cluster_summary.to_string(index=False))


if __name__ == "__main__":
    main()
