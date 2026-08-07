#!/usr/bin/env python3
"""Compare model residual bias by named western US subregions."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
REGION_TABLE = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
OUT_DIR = ROOT / "regional_residual_model_comparison_2026-06-04"
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
REGION_LABELS_ONE_LINE = {
    "PNW": "Pacific Northwest",
    "CA_med": "California Mediterranean",
    "S_Rockies": "Southern Rockies",
    "N_Rockies": "Northern Rockies",
    "SW_dry": "Southwest dry forests",
}
REGION_LABELS_MAP = {
    "PNW": "Pacific Northwest",
    "CA_med": "California\nMediterranean",
    "S_Rockies": "Southern\nRockies",
    "N_Rockies": "Northern\nRockies",
    "SW_dry": "Southwest\ndry forests",
}
REGION_COLORS = {
    "PNW": "#2b83ba",
    "CA_med": "#d7191c",
    "S_Rockies": "#fdae61",
    "N_Rockies": "#1a9641",
    "SW_dry": "#7b3294",
}
OVER_COLOR = "#e8751a"
UNDER_COLOR = "#36aa7b"
DOT_COLOR = "#1f1f1f"


def load_region_table() -> pd.DataFrame:
    return pd.read_parquet(REGION_TABLE, columns=["x", "y", "region"])


def normalize_model_frame(model_key: str, model_label: str, path: Path, regions: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_parquet(path)
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
        raise ValueError(f"{path} has no predicted response column")

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
    out["abs_residual"] = out["residual"].abs()
    return out


def summarize_models(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in predictions:
        model_key = df["model_key"].iloc[0]
        model_label = df["model_label"].iloc[0]
        obs = df["observed"].to_numpy(dtype=float)
        pred = df["predicted"].to_numpy(dtype=float)
        resid = df["residual"].to_numpy(dtype=float)
        rows.append(
            {
                "model_key": model_key,
                "model_label": model_label,
                "region": "ALL",
                "region_label": "All regions",
                "n": int(len(df)),
                "r2": float(r2_score(obs, pred)),
                "rmse": float(mean_squared_error(obs, pred) ** 0.5),
                "residual_mean": float(np.mean(resid)),
                "abs_residual_mean": float(np.mean(np.abs(resid))),
                "residual_sd": float(np.std(resid, ddof=1)),
                "underprediction_pct": float(100 * np.mean(resid >= 0)),
            }
        )
        for region in REGION_ORDER:
            sub = df[df["region"] == region]
            if sub.empty:
                rows.append(
                    {
                        "model_key": model_key,
                        "model_label": model_label,
                        "region": region,
                        "region_label": REGION_LABELS_ONE_LINE[region],
                        "n": 0,
                        "r2": np.nan,
                        "rmse": np.nan,
                        "residual_mean": np.nan,
                        "abs_residual_mean": np.nan,
                        "residual_sd": np.nan,
                        "underprediction_pct": np.nan,
                    }
                )
                continue
            obs = sub["observed"].to_numpy(dtype=float)
            pred = sub["predicted"].to_numpy(dtype=float)
            resid = sub["residual"].to_numpy(dtype=float)
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "region": region,
                    "region_label": REGION_LABELS_ONE_LINE[region],
                    "n": int(len(sub)),
                    "r2": float(r2_score(obs, pred)),
                    "rmse": float(mean_squared_error(obs, pred) ** 0.5),
                    "residual_mean": float(np.mean(resid)),
                    "abs_residual_mean": float(np.mean(np.abs(resid))),
                    "residual_sd": float(np.std(resid, ddof=1)),
                    "underprediction_pct": float(100 * np.mean(resid >= 0)),
                }
            )
    return pd.DataFrame(rows)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.01,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def plot_region_map(ax: plt.Axes, regions: pd.DataFrame) -> None:
    for region in REGION_ORDER:
        sub = regions[regions["region"] == region]
        ax.scatter(
            sub["x"],
            sub["y"],
            s=0.8,
            marker="s",
            linewidths=0,
            alpha=0.45,
            color=REGION_COLORS[region],
            rasterized=True,
        )
        ax.text(
            sub["x"].median(),
            sub["y"].median(),
            REGION_LABELS_MAP[region],
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=2.0),
        )
    panel_label(ax, "(a)")
    ax.set_aspect("equal")
    ax.axis("off")


def annotate_bias(ax: plt.Axes, x: float, y: float, xmin: float, xmax: float) -> None:
    span = xmax - xmin
    if x < -0.012:
        text_x = x + 0.014 * span
        ax.text(text_x, y, f"{x:+.3f}", va="center", ha="left", fontsize=7.0, color="white", fontweight="bold")
    elif x < 0:
        text_x = x - 0.012 * span
        ha = "right"
        if text_x < xmin + 0.02 * span:
            text_x = x + 0.014 * span
            ha = "left"
        ax.text(text_x, y, f"{x:+.3f}", va="center", ha=ha, fontsize=7.0, color="#222222")
    else:
        text_x = x + 0.012 * span
        ha = "left"
        if text_x > xmax - 0.02 * span:
            text_x = x - 0.014 * span
            ha = "right"
        ax.text(text_x, y, f"{x:+.3f}", va="center", ha=ha, fontsize=7.0, color="#222222")


def plot_model_panel(ax: plt.Axes, summary: pd.DataFrame, model_key: str, label: str, xlim: tuple[float, float]) -> None:
    sub = summary[summary["model_key"].eq(model_key) & summary["region"].isin(REGION_ORDER)].set_index("region")
    sub = sub.reindex(REGION_ORDER)
    y = np.arange(len(sub))
    residual_mean = sub["residual_mean"].to_numpy(dtype=float)
    abs_mean = sub["abs_residual_mean"].to_numpy(dtype=float)
    colors = np.where(residual_mean < 0, OVER_COLOR, UNDER_COLOR)

    ax.barh(y, residual_mean, color=colors, height=0.58, edgecolor="none", zorder=2)
    ax.scatter(abs_mean, y, s=22, color=DOT_COLOR, zorder=3)
    ax.axvline(0, color="#777777", linewidth=0.8, zorder=1)
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(*xlim)
    ax.set_yticks(y)
    ax.set_yticklabels([REGION_LABELS_ONE_LINE[r] for r in REGION_ORDER], fontsize=7.2)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=7.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Residual in Resistance units", fontsize=7.6)

    panel_label(ax, label)
    ax.text(
        0.985,
        0.985,
        sub["model_label"].iloc[0],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color="#222222",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=1.7),
    )

    xmin, xmax = xlim
    for yi, (_, row) in enumerate(sub.iterrows()):
        if not np.isfinite(row["residual_mean"]) or not np.isfinite(row["abs_residual_mean"]):
            ax.text(
                0.01,
                yi,
                "n=0",
                va="center",
                ha="left",
                fontsize=6.8,
                color="#666666",
            )
            continue
        annotate_bias(ax, float(row["residual_mean"]), yi, xmin, xmax)
        dot_x = float(row["abs_residual_mean"])
        dot_label_x = dot_x + 0.018 * (xmax - xmin)
        if dot_label_x > xmax - 0.025 * (xmax - xmin):
            dot_label_x = dot_x - 0.018 * (xmax - xmin)
            ha = "right"
        else:
            ha = "left"
        ax.text(dot_label_x, yi, f"|e|={dot_x:.3f}", va="center", ha=ha, fontsize=6.8, color="#333333")
        ax.text(
            xmax - 0.015 * (xmax - xmin),
            yi,
            f"n={int(row['n']):,}",
            va="center",
            ha="right",
            fontsize=6.4,
            color="#666666",
        )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#dddddd")


def draw_figure(regions: pd.DataFrame, summary: pd.DataFrame) -> tuple[Path, Path]:
    model_keys = [item[0] for item in MODEL_FILES]
    values = summary[summary["region"].isin(REGION_ORDER)][["residual_mean", "abs_residual_mean"]].to_numpy(dtype=float)
    xmin = min(float(np.nanmin(values[:, 0])) * 1.45, -0.02)
    xmax = max(float(np.nanmax(values[:, 1])) * 1.62, 0.08)
    xlim = (xmin, xmax)

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.9), dpi=300, constrained_layout=False)
    plot_region_map(axes[0, 0], regions)
    labels = ["(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, model_key, label in zip(axes.flat[1:], model_keys, labels):
        plot_model_panel(ax, summary, model_key, label, xlim)

    handles = [
        mpatches.Patch(color=OVER_COLOR, label="overprediction"),
        mpatches.Patch(color=UNDER_COLOR, label="underprediction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DOT_COLOR, markersize=5, label="mean absolute residual"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=3, frameon=False, fontsize=8.4)
    fig.subplots_adjust(left=0.052, right=0.985, bottom=0.07, top=0.91, wspace=0.34, hspace=0.33)

    png = OUT_DIR / "Regional Residual Bias - Four Models and MGWR.png"
    pdf = OUT_DIR / "Regional Residual Bias - Four Models and MGWR.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    regions = load_region_table()
    predictions = [normalize_model_frame(*item, regions=regions) for item in MODEL_FILES]
    summary = summarize_models(predictions)
    summary_path = OUT_DIR / "regional_residual_model_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    png, pdf = draw_figure(regions, summary)
    print(f"Wrote: {png}")
    print(f"Wrote: {pdf}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
