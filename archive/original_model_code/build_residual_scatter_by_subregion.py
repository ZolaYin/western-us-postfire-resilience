#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
REGION_TABLE = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
OUT_DIR = ROOT / "residual_scatter_by_subregion_2026-06-04"
OUT_DIR.mkdir(exist_ok=True)

MODEL_FILES = [
    (
        "m2_baseline",
        "M2 Baseline RF",
        ROOT / "residual_plot_m2_baseline_2026-05-01" / "m2_baseline_block_100km_predictions.csv",
    ),
    (
        "best_rf",
        "M2-Based Residual-Corrected RF",
        ROOT / "residual_plot_best_rf_2026-05-01" / "m2_resid_localz5_noelev_cons_block_100km_predictions.csv",
    ),
]

REGION_ORDER = ["PNW", "CA_med", "S_Rockies", "N_Rockies", "SW_dry"]
REGION_LABELS = {
    "PNW": "Pacific Northwest",
    "CA_med": "California Mediterranean",
    "S_Rockies": "Southern Rockies",
    "N_Rockies": "Northern Rockies",
    "SW_dry": "Southwest dry forests",
}
REGION_COLORS = {
    "PNW": "#2b83ba",
    "CA_med": "#d7191c",
    "S_Rockies": "#fdae61",
    "N_Rockies": "#1a9641",
    "SW_dry": "#7b3294",
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.set_title(label, loc="left", fontsize=11, fontweight="bold", pad=4)


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def read_model_predictions(path: Path, model_key: str, model_label: str) -> pd.DataFrame:
    pred = pd.read_csv(path)
    regions = pd.read_parquet(REGION_TABLE, columns=["x", "y", "region"])
    out = pred.merge(regions, on=["x", "y"], how="left", validate="one_to_one")
    missing = int(out["region"].isna().sum())
    if missing:
        raise ValueError(f"{path} has {missing} rows without a region assignment")
    out["model_key"] = model_key
    out["model_label"] = model_label
    out["residual"] = out["observed"] - out["predicted"]
    out["residual_sign"] = np.where(out["residual"] >= 0, "underprediction", "overprediction")
    return out


def regional_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in REGION_ORDER:
        sub = df[df["region"] == region].copy()
        if sub.empty:
            continue
        obs = sub["observed"].to_numpy(dtype=float)
        pred = sub["predicted"].to_numpy(dtype=float)
        resid = sub["residual"].to_numpy(dtype=float)
        fit_obs = LinearRegression().fit(pred.reshape(-1, 1), obs)
        fit_resid = LinearRegression().fit(pred.reshape(-1, 1), resid)
        rows.append(
            {
                "model_key": sub["model_key"].iloc[0],
                "model_label": sub["model_label"].iloc[0],
                "region": region,
                "region_label": REGION_LABELS[region],
                "n": int(len(sub)),
                "r2": float(r2_score(obs, pred)),
                "rmse": float(mean_squared_error(obs, pred) ** 0.5),
                "mean_observed": float(np.mean(obs)),
                "mean_predicted": float(np.mean(pred)),
                "mean_residual": float(np.mean(resid)),
                "mean_abs_residual": float(np.mean(np.abs(resid))),
                "underprediction_pct": float(100 * np.mean(resid >= 0)),
                "overprediction_pct": float(100 * np.mean(resid < 0)),
                "obs_pred_slope": float(fit_obs.coef_[0]),
                "obs_pred_intercept": float(fit_obs.intercept_),
                "resid_pred_slope": float(fit_resid.coef_[0]),
                "resid_pred_intercept": float(fit_resid.intercept_),
            }
        )
    return pd.DataFrame(rows)


def plot_region_map(ax: plt.Axes, model_df: pd.DataFrame) -> None:
    for region in REGION_ORDER:
        sub = model_df[model_df["region"] == region]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=2.2,
            marker="s",
            linewidths=0,
            alpha=0.42,
            color=REGION_COLORS[region],
            rasterized=True,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    add_panel_label(ax, "(a)")


def plot_observed_predicted(ax: plt.Axes, df: pd.DataFrame, xlim: tuple[float, float]) -> None:
    rng = np.random.default_rng(42)
    order = rng.permutation(len(df))
    work = df.iloc[order]
    for region in REGION_ORDER:
        sub = work[work["region"] == region]
        ax.scatter(
            sub["predicted"],
            sub["observed"],
            s=5,
            linewidths=0,
            alpha=0.25,
            color=REGION_COLORS[region],
            label=REGION_LABELS[region],
            rasterized=True,
        )
        if len(sub) >= 30:
            pred = sub["predicted"].to_numpy(dtype=float)
            obs = sub["observed"].to_numpy(dtype=float)
            fit = LinearRegression().fit(pred.reshape(-1, 1), obs)
            xs = np.linspace(np.nanpercentile(pred, 2), np.nanpercentile(pred, 98), 80)
            ax.plot(xs, fit.predict(xs.reshape(-1, 1)), color=REGION_COLORS[region], linewidth=1.1, alpha=0.9)

    ax.plot(xlim, xlim, color="#1f1f1f", linestyle="--", linewidth=0.9, label="1:1")
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_xlabel("Predicted Resistance")
    ax.set_ylabel("Observed Resistance")
    add_grid(ax)
    add_panel_label(ax, "(b)")


def plot_residual_predicted(ax: plt.Axes, df: pd.DataFrame, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    rng = np.random.default_rng(43)
    order = rng.permutation(len(df))
    work = df.iloc[order]
    for region in REGION_ORDER:
        sub = work[work["region"] == region]
        ax.scatter(
            sub["predicted"],
            sub["residual"],
            s=5,
            linewidths=0,
            alpha=0.25,
            color=REGION_COLORS[region],
            label=REGION_LABELS[region],
            rasterized=True,
        )
        if len(sub) >= 30:
            pred = sub["predicted"].to_numpy(dtype=float)
            resid = sub["residual"].to_numpy(dtype=float)
            fit = LinearRegression().fit(pred.reshape(-1, 1), resid)
            xs = np.linspace(np.nanpercentile(pred, 2), np.nanpercentile(pred, 98), 80)
            ax.plot(xs, fit.predict(xs.reshape(-1, 1)), color=REGION_COLORS[region], linewidth=1.1, alpha=0.9)

    ax.axhline(0, color="#1f1f1f", linestyle="--", linewidth=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Predicted Resistance")
    ax.set_ylabel("Residual (observed - predicted)")
    add_grid(ax)
    add_panel_label(ax, "(c)")


def plot_spatial_residual_sign(ax: plt.Axes, df: pd.DataFrame) -> None:
    resid = df["residual"].to_numpy(dtype=float)
    residual_vmax = float(min(0.42, np.nanpercentile(np.abs(resid), 98.8)))
    norm = TwoSlopeNorm(vmin=-residual_vmax, vcenter=0.0, vmax=residual_vmax)
    rng = np.random.default_rng(44)
    order = rng.permutation(len(df))
    sc = ax.scatter(
        df["x"].to_numpy()[order],
        df["y"].to_numpy()[order],
        c=resid[order],
        cmap="RdBu",
        norm=norm,
        s=2.0,
        marker="s",
        linewidths=0,
        alpha=0.65,
        rasterized=True,
    )
    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#2166ac", markersize=6, label="Underprediction"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#b2182b", markersize=6, label="Overprediction"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=True)
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.038, pad=0.01)
    cbar.set_label("Residual", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.axis("off")
    add_panel_label(ax, "(d)")


def draw_model_figure(df: pd.DataFrame, summary: pd.DataFrame) -> Path:
    model_key = df["model_key"].iloc[0]
    model_label = df["model_label"].iloc[0]
    all_values = np.concatenate([df["observed"].to_numpy(dtype=float), df["predicted"].to_numpy(dtype=float)])
    xlim = (
        float(max(0.0, np.nanpercentile(all_values, 0.5) - 0.02)),
        float(min(1.12, np.nanpercentile(all_values, 99.5) + 0.02)),
    )
    resid = df["residual"].to_numpy(dtype=float)
    resid_limit = float(min(0.55, np.nanpercentile(np.abs(resid), 99) * 1.15))
    ylim = (-resid_limit, resid_limit)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), dpi=260, constrained_layout=True)
    plot_region_map(axes[0, 0], df)
    plot_observed_predicted(axes[0, 1], df, xlim)
    plot_residual_predicted(axes[1, 0], df, xlim, ylim)
    plot_spatial_residual_sign(axes[1, 1], df)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    region_handles = handles[: len(REGION_ORDER)]
    region_labels = labels[: len(REGION_ORDER)]
    axes[0, 1].legend(region_handles, region_labels, loc="upper left", fontsize=7.3, frameon=True, markerscale=2.4)

    png = OUT_DIR / f"{model_label} Subregion Residual Diagnostics.png"
    pdf = OUT_DIR / f"{model_label} Subregion Residual Diagnostics.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> None:
    all_summaries = []
    outputs = []
    for model_key, model_label, path in MODEL_FILES:
        df = read_model_predictions(path, model_key, model_label)
        summary = regional_summary(df)
        all_summaries.append(summary)
        outputs.append(draw_model_figure(df, summary))

    summary_all = pd.concat(all_summaries, ignore_index=True)
    summary_all.to_csv(OUT_DIR / "subregion_residual_scatter_summary.csv", index=False)
    print(f"Saved summary: {OUT_DIR / 'subregion_residual_scatter_summary.csv'}")
    for output in outputs:
        print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
