#!/usr/bin/env python3
"""Build residual point diagnostics colored by western US subregion."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
REGION_TABLE = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
OUT_DIR = ROOT / "residual_point_scatter_by_subregion_2026-06-04"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = [
    (
        "reduced_ols",
        "Reduced OLS",
        ROOT / "reduced_noevt_models_legacy_nlcd_20260411" / "reduced_model_ols_residuals.parquet",
    ),
    (
        "reduced_rf",
        "Reduced RF",
        ROOT / "reduced_noevt_models_legacy_nlcd_20260411" / "reduced_model_rf_residuals.parquet",
    ),
    (
        "reduced_xgb",
        "Reduced XGBoost",
        ROOT / "reduced_noevt_models_legacy_nlcd_20260411" / "reduced_model_xgb_residuals.parquet",
    ),
    (
        "best_rf_evt_poly",
        "Best RF EVT-poly",
        ROOT / "best_rf_evt_poly_diagnostics_20260412" / "best_rf_evt_poly_residuals.parquet",
    ),
    (
        "mgwr",
        "MGWR",
        ROOT / "mgwr_outputs_stage5b_s12k" / "mgwr_residuals.parquet",
    ),
]

REGION_ORDER = ["PNW", "CA_med", "S_Rockies", "N_Rockies", "SW_dry"]
REGION_LABELS_MAP = {
    "PNW": "Pacific Northwest",
    "CA_med": "California\nMediterranean",
    "S_Rockies": "Southern\nRockies",
    "N_Rockies": "Northern\nRockies",
    "SW_dry": "Southwest\ndry forests",
}
REGION_LABELS_ONE_LINE = {
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
UNDER_COLOR = "#2166ac"
OVER_COLOR = "#b2182b"
RANDOM_SEED = 42
SCATTER_MAX_PER_REGION = 2800
SPATIAL_MAX_PER_REGION = 8500


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.99,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, color="#e5e5e5", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def safe_filename(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def load_regions() -> pd.DataFrame:
    return pd.read_parquet(REGION_TABLE, columns=["x", "y", "region"])


def read_model_frame(model_key: str, model_label: str, path: Path, regions: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    out = pd.DataFrame()
    out["x"] = pd.to_numeric(df["x"], errors="coerce")
    out["y"] = pd.to_numeric(df["y"], errors="coerce")
    if "observed" in df.columns:
        out["observed"] = pd.to_numeric(df["observed"], errors="coerce")
    elif "Resistance" in df.columns:
        out["observed"] = pd.to_numeric(df["Resistance"], errors="coerce")
    else:
        raise ValueError(f"{path} has no observed response column")

    if "predicted" in df.columns:
        out["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")
    elif "prediction" in df.columns:
        out["predicted"] = pd.to_numeric(df["prediction"], errors="coerce")
    else:
        raise ValueError(f"{path} has no prediction column")

    if "residual" in df.columns:
        out["residual"] = pd.to_numeric(df["residual"], errors="coerce")
    else:
        out["residual"] = out["observed"] - out["predicted"]

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "observed", "predicted", "residual"])
    out = out.merge(regions, on=["x", "y"], how="left", validate="many_to_one")
    missing = int(out["region"].isna().sum())
    if missing:
        raise ValueError(f"{path} has {missing} rows without region labels")
    out["model_key"] = model_key
    out["model_label"] = model_label
    out["residual_sign"] = np.where(out["residual"] >= 0, "underprediction", "overprediction")
    return out


def stratified_sample(df: pd.DataFrame, max_per_region: int, seed: int) -> pd.DataFrame:
    samples = []
    rng = np.random.default_rng(seed)
    for region in REGION_ORDER:
        sub = df[df["region"].eq(region)]
        if sub.empty:
            continue
        if len(sub) > max_per_region:
            idx = rng.choice(sub.index.to_numpy(), size=max_per_region, replace=False)
            samples.append(sub.loc[idx])
        else:
            samples.append(sub)
    return pd.concat(samples, ignore_index=True)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 10:
        return None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_key = df["model_key"].iloc[0]
    model_label = df["model_label"].iloc[0]
    for region in ["ALL", *REGION_ORDER]:
        sub = df if region == "ALL" else df[df["region"].eq(region)]
        if sub.empty:
            continue
        obs = sub["observed"].to_numpy(dtype=float)
        pred = sub["predicted"].to_numpy(dtype=float)
        resid = sub["residual"].to_numpy(dtype=float)
        obs_fit = linear_fit(pred, obs)
        resid_fit = linear_fit(pred, resid)
        rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "region": region,
                "region_label": "All regions" if region == "ALL" else REGION_LABELS_ONE_LINE[region],
                "n": int(len(sub)),
                "r2": float(r2_score(obs, pred)),
                "rmse": float(mean_squared_error(obs, pred) ** 0.5),
                "mean_residual": float(np.mean(resid)),
                "mean_abs_residual": float(np.mean(np.abs(resid))),
                "underprediction_pct": float(100 * np.mean(resid >= 0)),
                "overprediction_pct": float(100 * np.mean(resid < 0)),
                "obs_pred_slope": np.nan if obs_fit is None else obs_fit[0],
                "obs_pred_intercept": np.nan if obs_fit is None else obs_fit[1],
                "resid_pred_slope": np.nan if resid_fit is None else resid_fit[0],
                "resid_pred_intercept": np.nan if resid_fit is None else resid_fit[1],
            }
        )
    return pd.DataFrame(rows)


def plot_region_map(ax: plt.Axes, regions: pd.DataFrame) -> None:
    for region in REGION_ORDER:
        sub = regions[regions["region"].eq(region)]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=0.75,
            marker="s",
            linewidths=0,
            alpha=0.42,
            color=REGION_COLORS[region],
            rasterized=True,
        )
        ax.text(
            sub["x"].median(),
            sub["y"].median(),
            REGION_LABELS_MAP[region],
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=2.0),
        )
    panel_label(ax, "(a)")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_observed_predicted(ax: plt.Axes, df: pd.DataFrame, sample: pd.DataFrame, xlim: tuple[float, float]) -> None:
    for region in REGION_ORDER:
        sub = sample[sample["region"].eq(region)]
        ax.scatter(
            sub["predicted"],
            sub["observed"],
            s=5.5,
            color=REGION_COLORS[region],
            alpha=0.25,
            linewidths=0,
            rasterized=True,
        )
        full = df[df["region"].eq(region)]
        fit = linear_fit(full["predicted"].to_numpy(dtype=float), full["observed"].to_numpy(dtype=float))
        if fit is not None:
            slope, intercept = fit
            xs = np.linspace(
                np.nanpercentile(full["predicted"], 2),
                np.nanpercentile(full["predicted"], 98),
                100,
            )
            ax.plot(xs, slope * xs + intercept, color=REGION_COLORS[region], linewidth=1.3, alpha=0.95)
    ax.plot(xlim, xlim, color="#222222", linestyle="--", linewidth=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_xlabel("Predicted Resistance", fontsize=8)
    ax.set_ylabel("Observed Resistance", fontsize=8)
    ax.tick_params(labelsize=7.4)
    panel_label(ax, "(b)")
    add_grid(ax)


def plot_residual_predicted(
    ax: plt.Axes,
    df: pd.DataFrame,
    sample: pd.DataFrame,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for region in REGION_ORDER:
        sub = sample[sample["region"].eq(region)]
        ax.scatter(
            sub["predicted"],
            sub["residual"],
            s=5.5,
            color=REGION_COLORS[region],
            alpha=0.25,
            linewidths=0,
            rasterized=True,
        )
        full = df[df["region"].eq(region)]
        fit = linear_fit(full["predicted"].to_numpy(dtype=float), full["residual"].to_numpy(dtype=float))
        if fit is not None:
            slope, intercept = fit
            xs = np.linspace(
                np.nanpercentile(full["predicted"], 2),
                np.nanpercentile(full["predicted"], 98),
                100,
            )
            ax.plot(xs, slope * xs + intercept, color=REGION_COLORS[region], linewidth=1.3, alpha=0.95)
    ax.axhline(0, color="#222222", linestyle="--", linewidth=0.9)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Predicted Resistance", fontsize=8)
    ax.set_ylabel("Residual (observed - predicted)", fontsize=8)
    ax.tick_params(labelsize=7.4)
    panel_label(ax, "(c)")
    add_grid(ax)


def plot_spatial_sign(ax: plt.Axes, sample: pd.DataFrame) -> None:
    plot_order = sample.sample(frac=1.0, random_state=RANDOM_SEED)
    colors = np.where(plot_order["residual"].to_numpy(dtype=float) >= 0, UNDER_COLOR, OVER_COLOR)
    ax.scatter(
        plot_order["x"],
        plot_order["y"],
        s=1.45,
        marker="s",
        linewidths=0,
        alpha=0.48,
        color=colors,
        rasterized=True,
    )
    panel_label(ax, "(d)")
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=UNDER_COLOR, markeredgecolor="none", markersize=5.5, label="underprediction"),
        Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=OVER_COLOR, markeredgecolor="none", markersize=5.5, label="overprediction"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7.2, frameon=True)


def model_limits(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    values = np.concatenate([df["observed"].to_numpy(dtype=float), df["predicted"].to_numpy(dtype=float)])
    lo = float(max(0.0, np.nanpercentile(values, 0.5) - 0.025))
    hi = float(min(1.15, np.nanpercentile(values, 99.5) + 0.025))
    resid = df["residual"].to_numpy(dtype=float)
    rlim = float(min(0.62, np.nanpercentile(np.abs(resid), 99.0) * 1.08))
    return (lo, hi), (-rlim, rlim)


def draw_model_figure(df: pd.DataFrame, regions: pd.DataFrame) -> tuple[Path, Path]:
    model_label = df["model_label"].iloc[0]
    scatter_sample = stratified_sample(df, SCATTER_MAX_PER_REGION, RANDOM_SEED)
    spatial_sample = stratified_sample(df, SPATIAL_MAX_PER_REGION, RANDOM_SEED + 1)
    xlim, ylim = model_limits(df)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), dpi=300, constrained_layout=False)
    plot_region_map(axes[0, 0], regions)
    plot_observed_predicted(axes[0, 1], df, scatter_sample, xlim)
    plot_residual_predicted(axes[1, 0], df, scatter_sample, xlim, ylim)
    plot_spatial_sign(axes[1, 1], spatial_sample)

    region_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=REGION_COLORS[region],
            markerfacecolor=REGION_COLORS[region],
            markeredgecolor="none",
            markersize=5,
            linewidth=1.2,
            label=REGION_LABELS_ONE_LINE[region],
        )
        for region in REGION_ORDER
    ]
    fig.legend(
        handles=region_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=5,
        frameon=False,
        fontsize=7.6,
        columnspacing=1.1,
        handletextpad=0.35,
    )
    fig.text(0.5, 0.972, model_label, ha="center", va="top", fontsize=12, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.065, top=0.865, wspace=0.23, hspace=0.28)

    stem = f"Residual Point Diagnostics - {model_label}"
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    regions = load_regions()
    summaries = []
    outputs = []
    for model_key, model_label, path in MODEL_FILES:
        df = read_model_frame(model_key, model_label, path, regions)
        summaries.append(summarize(df))
        outputs.extend(draw_model_figure(df, regions))
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = OUT_DIR / "residual_point_scatter_by_subregion_summary.csv"
    summary.to_csv(summary_path, index=False)
    for path in outputs:
        print(f"Wrote: {path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
