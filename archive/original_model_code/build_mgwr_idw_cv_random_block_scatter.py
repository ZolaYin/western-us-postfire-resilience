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
from scipy.spatial import KDTree
from sklearn.metrics import mean_squared_error, r2_score

from build_predicted_observed_scatter_by_region import (
    REGION_COLORS,
    REGION_LABELS,
    REGION_ORDER,
    add_subregions,
    draw_scatter_panel,
    read_prediction_file,
    sample_by_group,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "predicted_observed_modelset_scatter_2026-06-10"
INPUT_DIR = ROOT / "predicted_observed_modelset_inputs_2026-06-10" / "mgwr_idw_cv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_TABLE = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
MGWR_COEF = ROOT / "mgwr_outputs_stage5b_s12k" / "mgwr_coefficients.parquet"
MGWR_RESID = ROOT / "mgwr_outputs_stage5b_s12k" / "mgwr_residuals.parquet"

RANDOM_SEED = 42
TEST_FRACTION = 0.2
BLOCK_SIZE = 100_000.0
KNN = 10

RAW_PREDICTOR_MAP = {
    "TS_elev_m_z": "TS_elev_m",
    "TS_slope_deg_z": "TS_slope_deg",
    "TS_SOC_0_30cm_clean_z": "TS_SOC_0_30cm",
    "FS_TCC_t0_z": "FS_TCC_t0",
    "FS_CBH_t0agg_z": "FS_CBH_t0agg",
    "HUM_viirs_near_t0_log_z": "HUM_viirs_near_t0",
    "HUM_imperv_near_t0_z": "HUM_imperv_near_t0",
    "HUM_roaddens_r5km_z": "HUM_roaddens_r5km",
    "HUM_traildens_r10km_z": "HUM_traildens_r10km",
    "CLIM_pr_sum_pre_z": "CLIM_pr_sum_pre",
    "CLIM_tmmn_mean_pre_z": "CLIM_tmmn_mean_pre",
}

RF_MODELS = [
    (
        "mgwr_idw_cv",
        "MGWR IDW-CV",
        {
            "random": INPUT_DIR / "mgwr_idw_cv_random_predictions.csv",
            "block": INPUT_DIR / "mgwr_idw_cv_block_predictions.csv",
        },
    ),
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


def raw_predictor_values(df: pd.DataFrame, term: str) -> np.ndarray:
    raw_col = RAW_PREDICTOR_MAP[term]
    values = pd.to_numeric(df[raw_col], errors="coerce").astype(float).to_numpy()
    if term == "HUM_viirs_near_t0_log_z":
        values = np.log1p(np.clip(values, 0, None))
    return values


def beta_column(data: pd.DataFrame, term: str) -> str:
    suffixed = f"{term}_beta"
    return suffixed if suffixed in data.columns else term


def recover_affine_predictor_transforms(
    data: pd.DataFrame, terms: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    features = []
    for term in terms:
        beta = data[beta_column(data, term)].to_numpy(dtype=float)
        raw_values = raw_predictor_values(data, term)
        features.append(beta * raw_values)
        features.append(beta)

    design = np.column_stack(features)
    target = data["stored_predicted"].to_numpy(dtype=float) - data["Intercept"].to_numpy(dtype=float)
    good = np.isfinite(design).all(axis=1) & np.isfinite(target)
    params, *_ = np.linalg.lstsq(design[good], target[good], rcond=None)

    rows = []
    for i, term in enumerate(terms):
        scale = float(params[2 * i])
        offset = float(params[2 * i + 1])
        rows.append(
            {
                "term": term,
                "raw_column": RAW_PREDICTOR_MAP[term],
                "scale": scale,
                "offset": offset,
                "implied_mean": float(-offset / scale) if scale != 0 else np.nan,
                "implied_sd": float(1.0 / scale) if scale != 0 else np.nan,
            }
        )
    return params, pd.DataFrame(rows)


def build_design_matrix(data: pd.DataFrame, terms: list[str], params: np.ndarray) -> np.ndarray:
    cols = [np.ones(len(data), dtype=float)]
    for i, term in enumerate(terms):
        scale = params[2 * i]
        offset = params[2 * i + 1]
        cols.append(scale * raw_predictor_values(data, term) + offset)
    return np.column_stack(cols)


def idw_interpolate(
    train_coords: np.ndarray,
    train_coefs: np.ndarray,
    test_coords: np.ndarray,
    k: int = KNN,
    power: float = 2.0,
) -> np.ndarray:
    tree = KDTree(train_coords)
    dists, idxs = tree.query(test_coords, k=min(k, len(train_coords)))
    if idxs.ndim == 1:
        idxs = idxs[:, None]
        dists = dists[:, None]
    dists = np.maximum(dists, 1e-10)
    weights = 1.0 / dists**power
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, train_coefs[idxs])


def load_mgwr_data() -> tuple[pd.DataFrame, list[str], np.ndarray, pd.DataFrame]:
    coef = pd.read_parquet(MGWR_COEF)
    resid = pd.read_parquet(MGWR_RESID)
    terms = [c for c in coef.columns if c not in {"x", "y", "Intercept"}]

    raw_cols = sorted(set(RAW_PREDICTOR_MAP.values()))
    raw = pd.read_parquet(RAW_TABLE, columns=["x", "y", "Resistance", "region", *raw_cols])
    data = (
        raw.merge(coef, on=["x", "y"], how="inner", suffixes=("", "_beta"))
        .merge(
            resid[["x", "y", "observed", "predicted"]],
            on=["x", "y"],
            how="inner",
        )
        .rename(columns={"predicted": "stored_predicted"})
        .reset_index(drop=True)
    )
    if len(data) != len(coef):
        raise ValueError(f"Expected {len(coef)} MGWR coefficient rows, matched {len(data)} rows")

    params, transform_table = recover_affine_predictor_transforms(data, terms)
    design = build_design_matrix(data, terms, params)
    coef_matrix = np.column_stack(
        [data["Intercept"].to_numpy(dtype=float)]
        + [data[beta_column(data, term)].to_numpy(dtype=float) for term in terms]
    )
    fitted_check = (design * coef_matrix).sum(axis=1)
    diff = fitted_check - data["stored_predicted"].to_numpy(dtype=float)
    transform_table["fitted_check_rmse"] = float(np.sqrt(np.nanmean(diff**2)))
    transform_table["fitted_check_max_abs"] = float(np.nanmax(np.abs(diff)))

    data["observed"] = data["observed"].astype(float)
    return data, terms, params, transform_table


def split_masks(data: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    if split == "random":
        is_test = rng.random(len(data)) < TEST_FRACTION
    elif split == "block":
        coords = data[["x", "y"]].to_numpy(dtype=float)
        bx = np.floor(coords[:, 0] / BLOCK_SIZE).astype(int).astype(str)
        by = np.floor(coords[:, 1] / BLOCK_SIZE).astype(int).astype(str)
        block_ids = np.char.add(np.char.add(bx, "_"), by)
        unique_blocks = np.unique(block_ids)
        n_test_blocks = max(1, round(len(unique_blocks) * TEST_FRACTION))
        test_blocks = set(rng.choice(unique_blocks, size=n_test_blocks, replace=False))
        is_test = np.array([block_id in test_blocks for block_id in block_ids], dtype=bool)
    else:
        raise ValueError(f"Unknown split: {split}")
    return ~is_test, is_test


def compute_cv_predictions(data: pd.DataFrame, terms: list[str], params: np.ndarray, split: str) -> pd.DataFrame:
    train_mask, test_mask = split_masks(data, split)
    coords = data[["x", "y"]].to_numpy(dtype=float)
    design = build_design_matrix(data, terms, params)
    coef_matrix = np.column_stack(
        [data["Intercept"].to_numpy(dtype=float)]
        + [data[beta_column(data, term)].to_numpy(dtype=float) for term in terms]
    )
    interp_coefs = idw_interpolate(
        coords[train_mask],
        coef_matrix[train_mask],
        coords[test_mask],
        k=KNN,
    )
    predicted = (design[test_mask] * interp_coefs).sum(axis=1)
    out = data.loc[test_mask, ["x", "y", "observed", "region"]].copy()
    out["predicted"] = predicted
    out["residual"] = out["observed"] - out["predicted"]
    out["split"] = split
    out["model_key"] = "mgwr_idw_cv"
    out["model_label"] = "MGWR IDW-CV"
    return out.reset_index(drop=True)


def metrics(df: pd.DataFrame) -> dict[str, float]:
    return {
        "n": int(len(df)),
        "r2": float(r2_score(df["observed"], df["predicted"])),
        "rmse": float(mean_squared_error(df["observed"], df["predicted"]) ** 0.5),
        "mean_residual": float(df["residual"].mean()),
    }


def global_limits(frames: list[pd.DataFrame]) -> tuple[float, float]:
    values = np.concatenate([frame[["observed", "predicted"]].to_numpy().ravel() for frame in frames])
    lo = float(max(0.0, np.nanpercentile(values, 0.5) - 0.03))
    hi = float(min(1.15, np.nanpercentile(values, 99.5) + 0.03))
    return lo, hi


def legend_handles() -> list[Line2D]:
    return [
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


def plot_mgwr_only(frames: dict[str, pd.DataFrame]) -> list[Path]:
    lim = global_limits([frames["random"], frames["block"]])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), dpi=300, sharex=True, sharey=True)
    for ax, split in zip(axes, ["random", "block"]):
        df = frames[split]
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
            f"MGWR IDW-CV\n{split}",
        )
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = OUT_DIR / "predicted_vs_observed_mgwr_idw_cv_random_block_subregion.png"
    pdf = OUT_DIR / "predicted_vs_observed_mgwr_idw_cv_random_block_subregion.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def load_prediction_frame(model_key: str, model_label: str, split: str, path: Path) -> pd.DataFrame:
    df = read_prediction_file(model_key, model_label, path)
    df = add_subregions(df)
    df["split"] = split
    return df


def plot_combined_modelset() -> tuple[list[Path], pd.DataFrame]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    for model_key, model_label, paths in RF_MODELS:
        for split, path in paths.items():
            df = load_prediction_frame(model_key, model_label, split, path)
            frames[(model_key, split)] = df
            row = {"model_key": model_key, "model_label": model_label, "split": split}
            row.update(metrics(df))
            rows.append(row)

    lim = global_limits(list(frames.values()))
    fig, axes = plt.subplots(2, len(RF_MODELS), figsize=(22.5, 8.4), dpi=300, sharex=True, sharey=True)
    for row_idx, split in enumerate(["random", "block"]):
        for col_idx, (model_key, model_label, _) in enumerate(RF_MODELS):
            ax = axes[row_idx, col_idx]
            df = frames[(model_key, split)]
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
                f"{model_label}\n{split}",
            )

    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    png = OUT_DIR / "predicted_vs_observed_modelset_with_mgwr_idw_cv_random_block_subregion.png"
    pdf = OUT_DIR / "predicted_vs_observed_modelset_with_mgwr_idw_cv_random_block_subregion.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "modelset_with_mgwr_idw_cv_metrics.csv", index=False)
    return [png, pdf], metrics_df


def main() -> None:
    data, terms, params, transform_table = load_mgwr_data()
    transform_table.to_csv(INPUT_DIR / "mgwr_recovered_predictor_transforms.csv", index=False)

    frames = {}
    metric_rows = []
    for split in ["random", "block"]:
        frame = compute_cv_predictions(data, terms, params, split)
        frame.to_csv(INPUT_DIR / f"mgwr_idw_cv_{split}_predictions.csv", index=False)
        frames[split] = frame
        row = {"model_key": "mgwr_idw_cv", "model_label": "MGWR IDW-CV", "split": split}
        row.update(metrics(frame))
        metric_rows.append(row)

    mgwr_metrics = pd.DataFrame(metric_rows)
    mgwr_metrics.to_csv(INPUT_DIR / "mgwr_idw_cv_metrics.csv", index=False)
    paths = plot_mgwr_only(frames)
    combined_paths, _ = plot_combined_modelset()

    for path in [
        INPUT_DIR / "mgwr_idw_cv_random_predictions.csv",
        INPUT_DIR / "mgwr_idw_cv_block_predictions.csv",
        INPUT_DIR / "mgwr_idw_cv_metrics.csv",
        INPUT_DIR / "mgwr_recovered_predictor_transforms.csv",
        *paths,
        *combined_paths,
        OUT_DIR / "modelset_with_mgwr_idw_cv_metrics.csv",
    ]:
        print(f"Wrote: {path}")
    print(mgwr_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
