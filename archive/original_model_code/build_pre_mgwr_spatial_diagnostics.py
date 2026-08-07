#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import libpysal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "mgwr_all_resilience_stage5b_samples_2026-05-20"
SAMPLE = SAMPLE_DIR / "sample_n12000_seed42.parquet"
PREDICTOR_FILE = SAMPLE_DIR / "predictors_stage5b.txt"
MGWR_RESID = ROOT / "mgwr_outputs_stage5b_s12k" / "mgwr_residuals.parquet"
STATE_SHP = Path("/private/tmp/cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
OUT_DIR = ROOT / "pre_mgwr_spatial_diagnostics_2026-05-28"
OUT_DIR.mkdir(exist_ok=True)

OUT_PNG = OUT_DIR / "pre_mgwr_ols_residual_spatial_diagnostics.png"
OUT_PDF = OUT_DIR / "pre_mgwr_ols_residual_spatial_diagnostics.pdf"
OUT_RESID = OUT_DIR / "pre_mgwr_ols_residuals.parquet"
OUT_CLUSTERS = OUT_DIR / "pre_mgwr_ols_local_moran_cluster_counts.csv"
OUT_SUMMARY = OUT_DIR / "pre_mgwr_spatial_diagnostics_summary.csv"

WESTERN11 = [
    "Arizona",
    "California",
    "Colorado",
    "Idaho",
    "Montana",
    "Nevada",
    "New Mexico",
    "Oregon",
    "Utah",
    "Washington",
    "Wyoming",
]

CLUSTER_ORDER = ["HH", "LL", "HL", "LH", "NS"]
CLUSTER_LABELS = {
    "HH": "High-high",
    "LL": "Low-low",
    "HL": "High-low",
    "LH": "Low-high",
    "NS": "Not significant",
}
CLUSTER_COLORS = {
    "HH": "#d73027",
    "LL": "#4575b4",
    "HL": "#fdae61",
    "LH": "#74add1",
    "NS": "#c9c9c9",
}


def load_states() -> gpd.GeoDataFrame:
    states = gpd.read_file(STATE_SHP)
    states = states.loc[states["NAME"].isin(WESTERN11)].to_crs("EPSG:5070")
    states["geometry"] = states.geometry.simplify(2500, preserve_topology=True)
    return states


def make_weights(coords: np.ndarray, k: int = 8) -> libpysal.weights.KNN:
    weights = libpysal.weights.KNN.from_array(coords, k=k)
    weights.transform = "r"
    return weights


def classify_local_clusters(z: np.ndarray, lag_z: np.ndarray, p_sim: np.ndarray) -> np.ndarray:
    clusters = np.full(len(z), "NS", dtype=object)
    sig = p_sim < 0.05
    clusters[sig & (z > 0) & (lag_z > 0)] = "HH"
    clusters[sig & (z < 0) & (lag_z < 0)] = "LL"
    clusters[sig & (z > 0) & (lag_z < 0)] = "HL"
    clusters[sig & (z < 0) & (lag_z > 0)] = "LH"
    return clusters


def local_neighbor_lag(z: np.ndarray, coords: np.ndarray, k: int = 8) -> np.ndarray:
    tree = KDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    return z[idx[:, 1:]].mean(axis=1)


def draw_state_context(ax: plt.Axes, states: gpd.GeoDataFrame) -> None:
    states.boundary.plot(ax=ax, color="#5f5f5f", linewidth=0.45, zorder=5)
    states.dissolve().boundary.plot(ax=ax, color="#222222", linewidth=0.95, zorder=6)
    xmin, ymin, xmax, ymax = states.total_bounds
    ax.set_xlim(xmin - 90_000, xmax + 90_000)
    ax.set_ylim(ymin - 90_000, ymax + 90_000)
    ax.set_aspect("equal")
    ax.set_axis_off()


def main() -> None:
    predictors = [line.strip() for line in PREDICTOR_FILE.read_text().splitlines() if line.strip()]
    df = pd.read_parquet(SAMPLE)
    work = df[["x", "y", "Resistance"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    xmat = work[predictors].to_numpy(dtype=float)
    y = work["Resistance"].to_numpy(dtype=float)
    coords = work[["x", "y"]].to_numpy(dtype=float)

    ols = LinearRegression()
    ols.fit(xmat, y)
    pred = ols.predict(xmat)
    resid = y - pred
    resid_z = (resid - resid.mean()) / resid.std(ddof=1)

    weights = make_weights(coords, k=8)
    moran = Moran(resid, weights, permutations=999)
    local = Moran_Local(resid, weights, permutations=199, seed=42)
    lag_z = local_neighbor_lag(resid_z, coords, k=8)
    clusters = classify_local_clusters(resid_z, lag_z, local.p_sim)

    mgwr_moran_i = np.nan
    mgwr_moran_p = np.nan
    if MGWR_RESID.exists():
        mgwr = pd.read_parquet(MGWR_RESID)
        mgwr_coords = mgwr[["x", "y"]].to_numpy(dtype=float)
        mgwr_weights = make_weights(mgwr_coords, k=8)
        mgwr_moran = Moran(mgwr["residual"].to_numpy(dtype=float), mgwr_weights, permutations=999)
        mgwr_moran_i = float(mgwr_moran.I)
        mgwr_moran_p = float(mgwr_moran.p_sim)

    out = work[["x", "y", "Resistance"]].copy()
    out["ols_predicted"] = pred
    out["ols_residual"] = resid
    out["ols_residual_z"] = resid_z
    out["local_lag_residual_z"] = lag_z
    out["local_moran_i"] = local.Is
    out["local_moran_p"] = local.p_sim
    out["local_cluster"] = clusters
    out.to_parquet(OUT_RESID, index=False)

    cluster_counts = (
        pd.Series(clusters, name="cluster")
        .value_counts()
        .reindex(CLUSTER_ORDER, fill_value=0)
        .rename_axis("cluster")
        .reset_index(name="n")
    )
    cluster_counts["label"] = cluster_counts["cluster"].map(CLUSTER_LABELS)
    cluster_counts["percent"] = cluster_counts["n"] / len(out) * 100
    cluster_counts.to_csv(OUT_CLUSTERS, index=False)

    summary = pd.DataFrame(
        [
            {
                "diagnostic": "Global OLS residual Moran's I",
                "response": "Resistance",
                "n": len(out),
                "k_neighbors": 8,
                "moran_i": float(moran.I),
                "p_sim": float(moran.p_sim),
                "r2": float(r2_score(y, pred)),
                "rmse": float(mean_squared_error(y, pred) ** 0.5),
                "bias": float(resid.mean()),
                "max_abs_residual_z_for_plot": float(np.percentile(np.abs(resid_z), 98)),
            },
            {
                "diagnostic": "MGWR stage5b residual Moran's I",
                "response": "Resistance",
                "n": len(out),
                "k_neighbors": 8,
                "moran_i": mgwr_moran_i,
                "p_sim": mgwr_moran_p,
                "r2": np.nan,
                "rmse": np.nan,
                "bias": np.nan,
                "max_abs_residual_z_for_plot": np.nan,
            },
        ]
    )
    summary.to_csv(OUT_SUMMARY, index=False)

    states = load_states()
    vmax = min(3.0, float(np.percentile(np.abs(resid_z), 98)))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 5.0), dpi=300, constrained_layout=False)

    ax = axes[0]
    draw_state_context(ax, states)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    sc = ax.scatter(
        out["x"],
        out["y"],
        c=out["ols_residual_z"],
        s=5.5,
        cmap="RdBu_r",
        norm=norm,
        linewidths=0,
        alpha=0.78,
        zorder=3,
        rasterized=True,
    )
    draw_state_context(ax, states)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("OLS residual z-score")
    cb.ax.tick_params(labelsize=7)
    ax.set_title("(A) Global OLS residuals")
    p_txt = "< 0.001" if moran.p_sim <= 0.001 else f"= {moran.p_sim:.3f}"
    ax.text(
        0.02,
        -0.035,
        f"Moran's I = {moran.I:.3f}\np {p_txt}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox=dict(facecolor="white", edgecolor="#777777", linewidth=0.45, alpha=0.90, boxstyle="round,pad=0.25"),
        zorder=20,
        clip_on=False,
    )

    ax = axes[1]
    draw_state_context(ax, states)
    ns = out["local_cluster"] == "NS"
    ax.scatter(out.loc[ns, "x"], out.loc[ns, "y"], c=CLUSTER_COLORS["NS"], s=3.2, alpha=0.24, linewidths=0, zorder=2)
    for cl in ["LL", "LH", "HL", "HH"]:
        sel = out["local_cluster"] == cl
        if not sel.any():
            continue
        ax.scatter(
            out.loc[sel, "x"],
            out.loc[sel, "y"],
            c=CLUSTER_COLORS[cl],
            s=8.0,
            alpha=0.86,
            linewidths=0,
            label=f"{cl} ({CLUSTER_LABELS[cl]})",
            zorder=4,
            rasterized=True,
        )
    draw_state_context(ax, states)
    ax.set_title("(B) Local Moran residual clusters")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 0.98),
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    ax.text(
        0.98,
        -0.035,
        "k = 8 nearest neighbors\np < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        bbox=dict(facecolor="white", edgecolor="#777777", linewidth=0.45, alpha=0.90, boxstyle="round,pad=0.25"),
        zorder=20,
        clip_on=False,
    )

    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.10, wspace=0.08)
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_SUMMARY}")
    print(f"Saved {OUT_CLUSTERS}")
    print(summary.to_string(index=False))
    print(cluster_counts.to_string(index=False))


if __name__ == "__main__":
    main()
