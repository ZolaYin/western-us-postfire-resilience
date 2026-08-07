#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from textwrap import shorten

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "predicted_observed_scatter_by_region_2026-06-10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGION_TABLE = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
EPA_L3_SHP = ROOT.parent / "EPA_Ecoregions" / "us_eco_l3" / "us_eco_l3.shp"

MODEL_FILES = [
    (
        "rf_block100km",
        "RF block validation",
        ROOT / "residual_plot_best_rf_2026-05-01" / "m2_resid_localz5_noelev_cons_block_100km_predictions.csv",
    ),
    (
        "mgwr",
        "MGWR fitted sample",
        ROOT / "mgwr_outputs_stage5b_s12k" / "mgwr_residuals.parquet",
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

RANDOM_SEED = 42
MAX_POINTS_PER_SUBREGION = 3500
MAX_POINTS_PER_ECOREGION = 1300
TOP_ECOREGIONS = 12


def read_prediction_file(model_key: str, model_label: str, path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    observed_col = "observed" if "observed" in df.columns else "Resistance"
    if observed_col not in df.columns:
        raise ValueError(f"{path} has no observed or Resistance column")

    predicted_col = None
    for candidate in ["predicted", "prediction", "fitted"]:
        if candidate in df.columns:
            predicted_col = candidate
            break
    if predicted_col is None:
        raise ValueError(f"{path} has no predicted, prediction, or fitted column")

    out = pd.DataFrame(
        {
            "x": pd.to_numeric(df["x"], errors="coerce"),
            "y": pd.to_numeric(df["y"], errors="coerce"),
            "observed": pd.to_numeric(df[observed_col], errors="coerce"),
            "predicted": pd.to_numeric(df[predicted_col], errors="coerce"),
        }
    )
    out["residual"] = out["observed"] - out["predicted"]
    if "residual" in df.columns:
        out["residual"] = pd.to_numeric(df["residual"], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out["model_key"] = model_key
    out["model_label"] = model_label
    return out


def add_subregions(df: pd.DataFrame) -> pd.DataFrame:
    regions = pd.read_parquet(REGION_TABLE, columns=["x", "y", "region"])
    out = df.merge(regions, on=["x", "y"], how="left", validate="many_to_one")
    missing = int(out["region"].isna().sum())
    if missing:
        raise ValueError(f"{missing} prediction rows did not match the subregion table")
    return out


def add_epa_l3(df: pd.DataFrame) -> pd.DataFrame:
    epa = gpd.read_file(EPA_L3_SHP)[["US_L3CODE", "US_L3NAME", "geometry"]]
    points = gpd.GeoDataFrame(
        df.reset_index(drop=True).assign(_row_id=lambda x: x.index),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=epa.crs,
    )
    joined = gpd.sjoin(points, epa, how="left", predicate="within")
    joined = joined.sort_values(["_row_id", "US_L3CODE"], na_position="last").drop_duplicates("_row_id")
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["eco_id"] = np.where(
        out["US_L3CODE"].notna(),
        out["US_L3CODE"].astype(str) + " " + out["US_L3NAME"].astype(str),
        "Outside mapped EPA L3",
    )
    return out


def summarize(df: pd.DataFrame, group_col: str, label_map: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []
    groups = [("ALL", df)]
    groups.extend((str(name), sub) for name, sub in df.groupby(group_col, sort=False))
    for name, sub in groups:
        if sub.empty:
            continue
        obs = sub["observed"].to_numpy(dtype=float)
        pred = sub["predicted"].to_numpy(dtype=float)
        residual = sub["residual"].to_numpy(dtype=float)
        rows.append(
            {
                "model_key": sub["model_key"].iloc[0],
                "model_label": sub["model_label"].iloc[0],
                "grouping": group_col,
                "group": name,
                "group_label": "All" if name == "ALL" else (label_map or {}).get(name, name),
                "n": int(len(sub)),
                "r2": float(r2_score(obs, pred)),
                "rmse": float(mean_squared_error(obs, pred) ** 0.5),
                "mean_observed": float(np.mean(obs)),
                "mean_predicted": float(np.mean(pred)),
                "mean_residual_obs_minus_pred": float(np.mean(residual)),
                "mean_abs_residual": float(np.mean(np.abs(residual))),
                "underprediction_pct": float(100 * np.mean(residual > 0)),
                "overprediction_pct": float(100 * np.mean(residual < 0)),
            }
        )
    return pd.DataFrame(rows)


def sample_by_group(df: pd.DataFrame, group_col: str, max_per_group: int) -> pd.DataFrame:
    parts = []
    rng = np.random.default_rng(RANDOM_SEED)
    for _, sub in df.groupby(group_col, sort=False):
        if len(sub) > max_per_group:
            idx = rng.choice(sub.index.to_numpy(), size=max_per_group, replace=False)
            parts.append(sub.loc[idx])
        else:
            parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def plot_limits(frames: list[pd.DataFrame]) -> tuple[float, float]:
    values = np.concatenate(
        [frame[["observed", "predicted"]].to_numpy(dtype=float).ravel() for frame in frames]
    )
    lo = float(max(0.0, np.nanpercentile(values, 0.5) - 0.03))
    hi = float(min(1.15, np.nanpercentile(values, 99.5) + 0.03))
    return lo, hi


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, color="#e7e7e7", linewidth=0.65, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def draw_scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    sample: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    lim: tuple[float, float],
    title: str,
    fit_lines: bool = True,
) -> None:
    for group in group_order:
        sub = sample[sample[group_col].eq(group)]
        if sub.empty:
            continue
        ax.scatter(
            sub["observed"],
            sub["predicted"],
            s=8.5,
            color=colors[group],
            alpha=0.28,
            linewidths=0,
            rasterized=True,
        )
        full = df[df[group_col].eq(group)]
        if fit_lines and len(full) >= 20:
            slope, intercept = np.polyfit(full["observed"], full["predicted"], 1)
            xs = np.linspace(
                np.nanpercentile(full["observed"], 2),
                np.nanpercentile(full["observed"], 98),
                100,
            )
            ax.plot(xs, slope * xs + intercept, color=colors[group], linewidth=1.25, alpha=0.95)

    ax.plot(lim, lim, color="#222222", linestyle="--", linewidth=0.9)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Observed resistance", fontsize=9)
    ax.set_ylabel("Predicted resistance", fontsize=9)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8)
    add_grid(ax)

    r2 = r2_score(df["observed"], df["predicted"])
    rmse = mean_squared_error(df["observed"], df["predicted"]) ** 0.5
    bias = float(np.mean(df["residual"]))
    ax.text(
        0.035,
        0.965,
        f"n = {len(df):,}\nR² = {r2:.3f}\nRMSE = {rmse:.3f}\nmean residual = {bias:+.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        bbox=dict(facecolor="white", edgecolor="#bdbdbd", linewidth=0.5, alpha=0.88, pad=3.0),
    )


def save_single_scatter(
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    max_per_group: int,
    stem: str,
    subtitle: str,
) -> list[Path]:
    sample = sample_by_group(df, group_col, max_per_group)
    lim = plot_limits([df])
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    draw_scatter_panel(ax, df, sample, group_col, group_order, labels, colors, lim, subtitle)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=colors[group],
            markerfacecolor=colors[group],
            markeredgecolor="none",
            linewidth=1.1,
            markersize=5,
            label=f"{labels[group]} (n={(df[group_col] == group).sum():,})",
        )
        for group in group_order
        if (df[group_col] == group).any()
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=7.5,
        handletextpad=0.35,
        borderaxespad=0,
    )
    fig.tight_layout()
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def save_model_comparison_subregions(frames: list[pd.DataFrame]) -> list[Path]:
    lim = plot_limits(frames)
    fig, axes = plt.subplots(1, len(frames), figsize=(12.2, 5.3), dpi=300, sharex=True, sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for ax, df in zip(axes, frames):
        sample = sample_by_group(df, "region", MAX_POINTS_PER_SUBREGION)
        draw_scatter_panel(
            ax,
            df,
            sample,
            "region",
            REGION_ORDER,
            REGION_LABELS,
            REGION_COLORS,
            lim,
            df["model_label"].iloc[0],
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
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=5,
        frameon=False,
        fontsize=8,
        columnspacing=1.15,
        handletextpad=0.35,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    stem = "predicted_vs_observed_subregion_rf_mgwr_comparison"
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def ecoregion_plot_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str], dict[str, str]]:
    counts = df["eco_id"].value_counts()
    top = counts.head(TOP_ECOREGIONS).index.tolist()
    out = df.copy()
    out["eco_plot"] = np.where(out["eco_id"].isin(top), out["eco_id"], "Other EPA L3 ecoregions")
    order = top + ["Other EPA L3 ecoregions"]
    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(i % 20) for i, name in enumerate(top)}
    colors["Other EPA L3 ecoregions"] = "#9e9e9e"
    labels = {name: shorten(name, width=48, placeholder="...") for name in order}
    return out, order, labels, colors


def main() -> None:
    frames = []
    summaries = []
    outputs = []

    for model_key, model_label, path in MODEL_FILES:
        df = add_subregions(read_prediction_file(model_key, model_label, path))
        frames.append(df)
        summaries.append(summarize(df, "region", REGION_LABELS))
        outputs.extend(
            save_single_scatter(
                df,
                "region",
                REGION_ORDER,
                REGION_LABELS,
                REGION_COLORS,
                MAX_POINTS_PER_SUBREGION,
                f"predicted_vs_observed_subregion_{model_key}",
                f"{model_label}: observed vs predicted by subregion",
            )
        )

        epa_df, eco_order, eco_labels, eco_colors = ecoregion_plot_groups(add_epa_l3(df))
        summaries.append(summarize(epa_df, "eco_id"))
        outputs.extend(
            save_single_scatter(
                epa_df,
                "eco_plot",
                eco_order,
                eco_labels,
                eco_colors,
                MAX_POINTS_PER_ECOREGION,
                f"predicted_vs_observed_epa_l3_top{TOP_ECOREGIONS}_{model_key}",
                f"{model_label}: observed vs predicted by EPA Level III ecoregion",
            )
        )

    outputs.extend(save_model_comparison_subregions(frames))
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = OUT_DIR / "predicted_vs_observed_region_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)

    for path in outputs:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
