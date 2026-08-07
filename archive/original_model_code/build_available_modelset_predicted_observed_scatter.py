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
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, r2_score

from build_predicted_observed_scatter_by_region import (
    REGION_COLORS,
    REGION_LABELS,
    REGION_ORDER,
    add_grid,
    add_subregions,
    draw_scatter_panel,
    read_prediction_file,
    sample_by_group,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "predicted_observed_modelset_scatter_2026-06-10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RF_MODELS = [
    (
        "m2_baseline",
        "M2 baseline",
        {
            "random": ROOT
            / "predicted_observed_modelset_inputs_2026-06-10"
            / "m2_baseline"
            / "m2_baseline_random_predictions.csv",
            "block": ROOT
            / "residual_plot_m2_baseline_2026-05-01"
            / "m2_baseline_block_100km_predictions.csv",
        },
    ),
    (
        "m2_mgwr",
        "M2 + MGWR-scale smooth",
        {
            "random": ROOT
            / "predicted_observed_modelset_inputs_2026-06-10"
            / "softmoe_fastplot_sample30k_maxk300"
            / "m2_mgwr_random_predictions.csv",
            "block": ROOT
            / "predicted_observed_modelset_inputs_2026-06-10"
            / "softmoe_fastplot_sample30k_maxk300"
            / "m2_mgwr_block_predictions.csv",
        },
    ),
    (
        "m2_soft3_rawsmooth",
        "M2 + Soft3 raw-smooth",
        {
            "random": ROOT
            / "predicted_observed_modelset_inputs_2026-06-10"
            / "softmoe_fastplot_sample30k_maxk300"
            / "m2_soft3_rawsmooth_random_predictions.csv",
            "block": ROOT
            / "predicted_observed_modelset_inputs_2026-06-10"
            / "softmoe_fastplot_sample30k_maxk300"
            / "m2_soft3_rawsmooth_block_predictions.csv",
        },
    ),
    (
        "m2_resid_localz5_noelev_cons",
        "M2 + residual local-z context",
        {
            "random": ROOT
            / "residual_plot_best_rf_2026-05-01"
            / "m2_resid_localz5_noelev_cons_random_predictions.csv",
            "block": ROOT
            / "residual_plot_best_rf_2026-05-01"
            / "m2_resid_localz5_noelev_cons_block_100km_predictions.csv",
        },
    ),
]

MGWR_FILE = ROOT / "mgwr_outputs_stage5b_s12k" / "mgwr_residuals.parquet"


def load_frame(model_key: str, model_label: str, split: str, path: Path) -> pd.DataFrame:
    df = read_prediction_file(model_key, model_label, path)
    df = add_subregions(df)
    df["split"] = split
    return df


def global_limits(frames: list[pd.DataFrame]) -> tuple[float, float]:
    vals = np.concatenate([frame[["observed", "predicted"]].to_numpy().ravel() for frame in frames])
    lo = float(max(0.0, np.nanpercentile(vals, 0.5) - 0.03))
    hi = float(min(1.15, np.nanpercentile(vals, 99.5) + 0.03))
    return lo, hi


def draw_missing(ax: plt.Axes, title: str, reason: str) -> None:
    ax.set_title(title, fontsize=9.2, fontweight="bold", pad=7)
    ax.text(
        0.5,
        0.55,
        "prediction CSV missing",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#555555",
    )
    ax.text(
        0.5,
        0.42,
        reason,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.5,
        color="#777777",
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d0d0d0")
    add_grid(ax)


def model_metrics(df: pd.DataFrame) -> dict[str, float]:
    return {
        "n": len(df),
        "r2": float(r2_score(df["observed"], df["predicted"])),
        "rmse": float(mean_squared_error(df["observed"], df["predicted"]) ** 0.5),
        "mean_residual": float(df["residual"].mean()),
    }


def main() -> None:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    manifest_rows = []
    for model_key, model_label, paths in RF_MODELS:
        for split, path in paths.items():
            exists = bool(path and path.exists())
            manifest_rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "split": split,
                    "status": "available" if exists else "missing_prediction_csv",
                    "path": "" if path is None else str(path),
                }
            )
            if exists:
                frames[(model_key, split)] = load_frame(model_key, model_label, split, path)

    mgwr = load_frame("mgwr", "MGWR fitted sample", "fitted", MGWR_FILE)
    manifest_rows.append(
        {
            "model_key": "mgwr",
            "model_label": "MGWR fitted sample",
            "split": "fitted",
            "status": "available_fitted_only",
            "path": str(MGWR_FILE),
        }
    )

    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "modelset_prediction_availability.csv", index=False)

    available = list(frames.values())
    lim = global_limits(available + [mgwr])
    fig, axes = plt.subplots(2, len(RF_MODELS), figsize=(18, 8.4), dpi=300, sharex=True, sharey=True)
    for row_idx, split in enumerate(["random", "block"]):
        for col_idx, (model_key, model_label, _) in enumerate(RF_MODELS):
            ax = axes[row_idx, col_idx]
            key = (model_key, split)
            title = f"{model_label}\n{split}"
            if key not in frames:
                draw_missing(
                    ax,
                    title,
                    "Need rerun with --save-predictions for this split.",
                )
                continue
            df = frames[key]
            sample = sample_by_group(df, "region", 2400)
            draw_scatter_panel(
                ax,
                df,
                sample,
                "region",
                REGION_ORDER,
                REGION_LABELS,
                REGION_COLORS,
                lim,
                title,
            )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=REGION_COLORS[region],
            markerfacecolor=REGION_COLORS[region],
            markeredgecolor="none",
            linewidth=1.1,
            markersize=5,
            label=REGION_LABELS[region],
        )
        for region in REGION_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    grid_png = OUT_DIR / "predicted_vs_observed_rf_modelset_random_block_available.png"
    grid_pdf = OUT_DIR / "predicted_vs_observed_rf_modelset_random_block_available.pdf"
    fig.savefig(grid_png, bbox_inches="tight")
    fig.savefig(grid_pdf, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    sample = sample_by_group(mgwr, "region", 2400)
    draw_scatter_panel(
        ax,
        mgwr,
        sample,
        "region",
        REGION_ORDER,
        REGION_LABELS,
        REGION_COLORS,
        lim,
        "MGWR fitted sample",
    )
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.98, 0.5), frameon=False)
    fig.tight_layout()
    mgwr_png = OUT_DIR / "predicted_vs_observed_mgwr_fitted_sample_subregion.png"
    mgwr_pdf = OUT_DIR / "predicted_vs_observed_mgwr_fitted_sample_subregion.pdf"
    fig.savefig(mgwr_png, bbox_inches="tight")
    fig.savefig(mgwr_pdf, bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for (model_key, split), df in frames.items():
        row = {"model_key": model_key, "model_label": df["model_label"].iloc[0], "split": split}
        row.update(model_metrics(df))
        summary_rows.append(row)
    row = {"model_key": "mgwr", "model_label": "MGWR fitted sample", "split": "fitted"}
    row.update(model_metrics(mgwr))
    summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "modelset_available_metrics.csv", index=False)

    for path in [grid_png, grid_pdf, mgwr_png, mgwr_pdf, OUT_DIR / "modelset_prediction_availability.csv", OUT_DIR / "modelset_available_metrics.csv"]:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
