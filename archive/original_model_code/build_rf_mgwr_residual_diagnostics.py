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
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
RF_RESID = ROOT / "residual_plot_best_rf_2026-05-01" / "m2_resid_localz5_noelev_cons_block_100km_predictions.csv"
MGWR_RESID = ROOT / "mgwr_outputs_stage5b_s12k" / "mgwr_residuals.parquet"
STATE_SHP = Path("/private/tmp/cb_2018_us_state_500k/cb_2018_us_state_500k.shp")
OUT_DIR = ROOT / "rf_mgwr_residual_diagnostics_2026-06-01"
OUT_DIR.mkdir(exist_ok=True)

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


def local_neighbor_lag(z: np.ndarray, coords: np.ndarray, k: int = 8) -> np.ndarray:
    tree = KDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    return z[idx[:, 1:]].mean(axis=1)


def classify_local_clusters(z: np.ndarray, lag_z: np.ndarray, p_sim: np.ndarray) -> np.ndarray:
    clusters = np.full(len(z), "NS", dtype=object)
    sig = p_sim < 0.05
    clusters[sig & (z > 0) & (lag_z > 0)] = "HH"
    clusters[sig & (z < 0) & (lag_z < 0)] = "LL"
    clusters[sig & (z > 0) & (lag_z < 0)] = "HL"
    clusters[sig & (z < 0) & (lag_z > 0)] = "LH"
    return clusters


def draw_state_context(ax: plt.Axes, states: gpd.GeoDataFrame) -> None:
    states.boundary.plot(ax=ax, color="#5f5f5f", linewidth=0.45, zorder=5)
    states.dissolve().boundary.plot(ax=ax, color="#222222", linewidth=0.95, zorder=6)
    xmin, ymin, xmax, ymax = states.total_bounds
    ax.set_xlim(xmin - 90_000, xmax + 90_000)
    ax.set_ylim(ymin - 90_000, ymax + 90_000)
    ax.set_aspect("equal")
    ax.set_axis_off()


def read_residuals(path: Path, model: str) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    needed = ["x", "y", "observed", "predicted", "residual"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    out["model"] = model
    out["residual_z"] = (out["residual"] - out["residual"].mean()) / out["residual"].std(ddof=1)
    return out


def add_spatial_diagnostics(df: pd.DataFrame, k: int = 8) -> tuple[pd.DataFrame, dict[str, float]]:
    coords = df[["x", "y"]].to_numpy(dtype=float)
    residual = df["residual"].to_numpy(dtype=float)
    residual_z = df["residual_z"].to_numpy(dtype=float)
    weights = make_weights(coords, k=k)
    moran = Moran(residual, weights, permutations=999)
    local = Moran_Local(residual, weights, permutations=199, seed=42)
    lag_z = local_neighbor_lag(residual_z, coords, k=k)
    out = df.copy()
    out["local_lag_residual_z"] = lag_z
    out["local_moran_i"] = local.Is
    out["local_moran_p"] = local.p_sim
    out["local_cluster"] = classify_local_clusters(residual_z, lag_z, local.p_sim)
    metrics = {
        "n": float(len(out)),
        "r2": float(r2_score(out["observed"], out["predicted"])),
        "rmse": float(mean_squared_error(out["observed"], out["predicted"]) ** 0.5),
        "bias": float(out["residual"].mean()),
        "moran_i": float(moran.I),
        "p_sim": float(moran.p_sim),
        "residual_z_plot_abs98": float(np.percentile(np.abs(out["residual_z"]), 98)),
    }
    return out, metrics


def format_p(p: float) -> str:
    if p <= 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def draw_residual_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    states: gpd.GeoDataFrame,
    norm: TwoSlopeNorm,
    label: str,
    colorbar_label: str,
    metrics: dict[str, float],
    add_colorbar: bool,
    fig: plt.Figure,
) -> None:
    draw_state_context(ax, states)
    sc = ax.scatter(
        df["x"],
        df["y"],
        c=df["residual_z"],
        s=4.7,
        cmap="RdBu_r",
        norm=norm,
        linewidths=0,
        alpha=0.78,
        zorder=3,
        rasterized=True,
    )
    draw_state_context(ax, states)
    ax.text(
        0.01,
        1.015,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.02,
        -0.035,
        f"Moran's I = {metrics['moran_i']:.3f}\np {format_p(metrics['p_sim'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        bbox=dict(facecolor="white", edgecolor="#777777", linewidth=0.45, alpha=0.90, boxstyle="round,pad=0.25"),
        zorder=20,
        clip_on=False,
    )
    if add_colorbar:
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.018)
        cb.set_label(colorbar_label, fontsize=7.6)
        cb.ax.tick_params(labelsize=7)


def draw_cluster_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    states: gpd.GeoDataFrame,
    label: str,
    add_legend: bool,
) -> None:
    draw_state_context(ax, states)
    ns = df["local_cluster"] == "NS"
    ax.scatter(
        df.loc[ns, "x"],
        df.loc[ns, "y"],
        c=CLUSTER_COLORS["NS"],
        s=2.6,
        alpha=0.24,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )
    for cluster in ["LL", "LH", "HL", "HH"]:
        sel = df["local_cluster"] == cluster
        if not sel.any():
            continue
        ax.scatter(
            df.loc[sel, "x"],
            df.loc[sel, "y"],
            c=CLUSTER_COLORS[cluster],
            s=7.0,
            alpha=0.86,
            linewidths=0,
            label=f"{cluster} ({CLUSTER_LABELS[cluster]})",
            zorder=4,
            rasterized=True,
        )
    draw_state_context(ax, states)
    ax.text(
        0.01,
        1.015,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        0.98,
        -0.035,
        "k = 8 nearest neighbors\np < 0.05",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        bbox=dict(facecolor="white", edgecolor="#777777", linewidth=0.45, alpha=0.90, boxstyle="round,pad=0.25"),
        zorder=20,
        clip_on=False,
    )
    if add_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 0.98),
            frameon=True,
            framealpha=0.92,
            borderpad=0.35,
            handletextpad=0.35,
            labelspacing=0.25,
            fontsize=7.2,
        )


def save_single_figure(
    model_key: str,
    df: pd.DataFrame,
    metrics: dict[str, float],
    states: gpd.GeoDataFrame,
    norm: TwoSlopeNorm,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.9), dpi=300, constrained_layout=False)
    draw_residual_panel(
        axes[0],
        df,
        states,
        norm,
        "(A)",
        "Residual z-score",
        metrics,
        True,
        fig,
    )
    draw_cluster_panel(axes[1], df, states, "(B)", True)
    fig.subplots_adjust(left=0.02, right=0.84, top=0.94, bottom=0.10, wspace=0.08)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{model_key}_residual_diagnostics_resistance.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_combined_figure(
    rf: pd.DataFrame,
    mgwr: pd.DataFrame,
    rf_metrics: dict[str, float],
    mgwr_metrics: dict[str, float],
    states: gpd.GeoDataFrame,
    norm: TwoSlopeNorm,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.8), dpi=300, constrained_layout=False)
    draw_residual_panel(
        axes[0, 0],
        rf,
        states,
        norm,
        "(A)",
        "Residual z-score",
        rf_metrics,
        True,
        fig,
    )
    draw_cluster_panel(axes[0, 1], rf, states, "(B)", True)
    draw_residual_panel(
        axes[1, 0],
        mgwr,
        states,
        norm,
        "(C)",
        "Residual z-score",
        mgwr_metrics,
        True,
        fig,
    )
    draw_cluster_panel(axes[1, 1], mgwr, states, "(D)", False)
    fig.subplots_adjust(left=0.02, right=0.84, top=0.97, bottom=0.06, wspace=0.08, hspace=0.22)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"rf_mgwr_residual_diagnostics_resistance_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_tables(rf: pd.DataFrame, mgwr: pd.DataFrame, metrics: dict[str, dict[str, float]]) -> None:
    summary = pd.DataFrame(
        [
            {
                "model": model,
                "response": "Resistance",
                "n": int(vals["n"]),
                "r2": vals["r2"],
                "rmse": vals["rmse"],
                "bias_observed_minus_predicted": vals["bias"],
                "moran_i": vals["moran_i"],
                "p_sim": vals["p_sim"],
                "k_neighbors": 8,
                "local_moran_permutations": 199,
            }
            for model, vals in metrics.items()
        ]
    )
    summary.to_csv(OUT_DIR / "rf_mgwr_residual_diagnostics_summary.csv", index=False)
    for model, df in [("best_rf", rf), ("mgwr", mgwr)]:
        counts = (
            df["local_cluster"]
            .value_counts()
            .reindex(CLUSTER_ORDER, fill_value=0)
            .rename_axis("cluster")
            .reset_index(name="n")
        )
        counts["label"] = counts["cluster"].map(CLUSTER_LABELS)
        counts["percent"] = counts["n"] / len(df) * 100
        counts.insert(0, "model", model)
        counts.to_csv(OUT_DIR / f"{model}_local_moran_cluster_counts.csv", index=False)
    rf.to_parquet(OUT_DIR / "best_rf_residual_diagnostics_resistance.parquet", index=False)
    mgwr.to_parquet(OUT_DIR / "mgwr_residual_diagnostics_resistance.parquet", index=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.2,
        }
    )

    states = load_states()
    rf_raw = read_residuals(RF_RESID, "Best RF")
    mgwr_raw = read_residuals(MGWR_RESID, "MGWR")
    rf, rf_metrics = add_spatial_diagnostics(rf_raw, k=8)
    mgwr, mgwr_metrics = add_spatial_diagnostics(mgwr_raw, k=8)

    vmax = min(3.0, float(np.percentile(np.abs(pd.concat([rf["residual_z"], mgwr["residual_z"]])), 98)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    save_single_figure("best_rf", rf, rf_metrics, states, norm)
    save_single_figure("mgwr", mgwr, mgwr_metrics, states, norm)
    save_combined_figure(rf, mgwr, rf_metrics, mgwr_metrics, states, norm)
    write_tables(rf, mgwr, {"best_rf": rf_metrics, "mgwr": mgwr_metrics})

    print(f"Saved outputs in: {OUT_DIR}")
    print(pd.read_csv(OUT_DIR / "rf_mgwr_residual_diagnostics_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
