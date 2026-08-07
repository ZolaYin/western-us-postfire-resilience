#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
TODAY = date.today().strftime("%Y-%m-%d")
RANDOM_STATE = 42

FEATURE_Z_COLS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_twi_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "FS_EVT_resistance_proxy_z",
    "FS_EVT_regeneration_proxy_z",
    "HUM_popdens_win10km_log_z",
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
    "HUM_roaddens_r5km_z",
    "HUM_traildens_r10km_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_tmmx_std_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_vpd_mean_pre_z",
    "CLIM_vpd_std_pre_z",
]

BASE_TO_Z = {
    "TS_elev_m_z": "TS_elev_m",
    "TS_slope_deg_z": "TS_slope_deg",
    "TS_twi_z": "TS_twi",
    "TS_SOC_0_30cm_z": "TS_SOC_0_30cm",
    "FS_TCC_t0_z": "FS_TCC_t0",
    "FS_CBH_t0agg_z": "FS_CBH_t0agg",
    "FS_EVT_resistance_proxy_z": "FS_EVT_resistance_proxy",
    "FS_EVT_regeneration_proxy_z": "FS_EVT_regeneration_proxy",
    "HUM_imperv_near_t0_z": "HUM_imperv_near_t0",
    "HUM_roaddens_r5km_z": "HUM_roaddens_r5km",
    "HUM_traildens_r10km_z": "HUM_traildens_r10km",
    "CLIM_pr_sum_pre_z": "CLIM_pr_sum_pre",
    "CLIM_tmmn_mean_pre_z": "CLIM_tmmn_mean_pre",
    "CLIM_hot_days_35C_pre_z": "CLIM_hot_days_35C_pre",
    "CLIM_tmmx_std_pre_z": "CLIM_tmmx_std_pre",
    "CLIM_aridity_pre_z": "CLIM_aridity_pre",
    "CLIM_vpd_mean_pre_z": "CLIM_vpd_mean_pre",
    "CLIM_vpd_std_pre_z": "CLIM_vpd_std_pre",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA-based latent environmental regime discovery for Western US."
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--pca-mode", choices=["fixed", "var90"], default="fixed")
    parser.add_argument("--n-pcs", type=int, default=8)
    parser.add_argument("--clusters", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def log1p_z(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float).clip(lower=0)
    return zscore(np.log1p(clean))


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for z_col, raw_col in BASE_TO_Z.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])
    if "HUM_popdens_win10km_log_z" not in out.columns and "HUM_popdens_win10km" in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns and "HUM_viirs_near_t0" in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    return out


def prepare_work(df: pd.DataFrame, sample_n: int | None) -> pd.DataFrame:
    keep_cols = ["pixel_id", "row", "col", "x", "y", "lon_wgs84", "lat_wgs84", "region", "Resistance"] + FEATURE_Z_COLS
    keep_cols = [c for c in keep_cols if c in df.columns]
    work = (
        df[keep_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=FEATURE_Z_COLS)
        .reset_index(drop=True)
    )
    if sample_n is not None and sample_n < len(work):
        work = work.sample(n=sample_n, random_state=RANDOM_STATE).reset_index(drop=True)
    return work


def fit_pca(X: np.ndarray, mode: str, n_pcs: int) -> tuple[PCA, np.ndarray]:
    if mode == "var90":
        pca = PCA(n_components=0.90, random_state=RANDOM_STATE)
    else:
        pca = PCA(n_components=n_pcs, random_state=RANDOM_STATE)
    scores = pca.fit_transform(X)
    return pca, scores


def evaluate_labels(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "silhouette": float(silhouette_score(scores, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(scores, labels)),
        "davies_bouldin": float(davies_bouldin_score(scores, labels)),
    }


def summarize_clusters(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(cluster_col)
        .agg(
            n=("Resistance", "size"),
            resistance_mean=("Resistance", "mean"),
            x_mean=("x", "mean"),
            y_mean=("y", "mean"),
        )
        .reset_index()
    )
    env_means = (
        df.groupby(cluster_col)[FEATURE_Z_COLS]
        .mean()
        .reset_index()
    )
    return summary.merge(env_means, on=cluster_col, how="left")


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or ROOT / f"latent_regime_pca_{TODAY}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = ensure_columns(pd.read_parquet(args.input))
    work = prepare_work(df, args.sample_n)
    X = work[FEATURE_Z_COLS].to_numpy(dtype=float)
    pca, scores = fit_pca(X, args.pca_mode, args.n_pcs)

    pca_scores_df = pd.DataFrame(
        scores,
        columns=[f"PC{i+1}" for i in range(scores.shape[1])],
        index=work.index,
    )
    pd.concat([work[["pixel_id", "x", "y", "region", "Resistance"]], pca_scores_df], axis=1).to_csv(
        out_dir / "pca_scores.csv", index=False
    )

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=FEATURE_Z_COLS,
        columns=[f"PC{i+1}" for i in range(scores.shape[1])],
    )
    loadings_df.to_csv(out_dir / "pca_loadings.csv")

    explained_df = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(scores.shape[1])],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    explained_df.to_csv(out_dir / "pca_explained_variance.csv", index=False)

    metrics_rows: list[dict] = []
    summary_paths: list[dict] = []

    for k in args.clusters:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        km_labels = km.fit_predict(scores)
        km_metrics = evaluate_labels(scores, km_labels)
        km_col = f"kmeans_k{k}"
        km_assign = work[["pixel_id", "row", "col", "x", "y", "lon_wgs84", "lat_wgs84", "region", "Resistance"]].copy()
        km_assign[km_col] = km_labels
        km_assign.to_csv(out_dir / f"{km_col}_assignments.csv", index=False)
        km_summary = summarize_clusters(km_assign.merge(work[FEATURE_Z_COLS], left_index=True, right_index=True), km_col)
        km_summary.to_csv(out_dir / f"{km_col}_summary.csv", index=False)
        metrics_rows.append({"method": "kmeans", "k": k, **km_metrics})
        summary_paths.append({"method": "kmeans", "k": k, "assignments": f"{km_col}_assignments.csv", "summary": f"{km_col}_summary.csv"})

        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE)
        gmm_labels = gmm.fit_predict(scores)
        gmm_metrics = evaluate_labels(scores, gmm_labels)
        gmm_col = f"gmm_k{k}"
        gmm_assign = work[["pixel_id", "row", "col", "x", "y", "lon_wgs84", "lat_wgs84", "region", "Resistance"]].copy()
        gmm_assign[gmm_col] = gmm_labels
        gmm_assign.to_csv(out_dir / f"{gmm_col}_assignments.csv", index=False)
        gmm_summary = summarize_clusters(gmm_assign.merge(work[FEATURE_Z_COLS], left_index=True, right_index=True), gmm_col)
        gmm_summary.to_csv(out_dir / f"{gmm_col}_summary.csv", index=False)
        metrics_rows.append({"method": "gmm", "k": k, **gmm_metrics})
        summary_paths.append({"method": "gmm", "k": k, "assignments": f"{gmm_col}_assignments.csv", "summary": f"{gmm_col}_summary.csv"})

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["method", "k"]).reset_index(drop=True)
    metrics_df.to_csv(out_dir / "clustering_metrics.csv", index=False)
    pd.DataFrame(summary_paths).to_csv(out_dir / "outputs_index.csv", index=False)

    metadata = {
        "input": str(args.input),
        "rows_used": int(len(work)),
        "feature_count": len(FEATURE_Z_COLS),
        "features": FEATURE_Z_COLS,
        "pca_mode": args.pca_mode,
        "n_pcs_requested": args.n_pcs,
        "n_pcs_actual": int(scores.shape[1]),
        "clusters": args.clusters,
        "sample_n": args.sample_n,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    lines = [
        f"# Latent Environmental Regime Discovery - PCA baseline ({TODAY})",
        "",
        f"**Input:** `{args.input.name}`  ",
        f"**Rows used:** {len(work):,}  ",
        f"**Features:** {len(FEATURE_Z_COLS)}  ",
        f"**PCA mode:** `{args.pca_mode}`  ",
        f"**PCs retained:** {scores.shape[1]}",
        "",
        "## 1. PCA explained variance",
        "",
        "| Component | Explained variance ratio | Cumulative explained variance |",
        "|---|---|---|",
    ]
    for _, row in explained_df.iterrows():
        lines.append(
            f"| {row['component']} | {row['explained_variance_ratio']:.4f} | {row['cumulative_explained_variance']:.4f} |"
        )

    lines += [
        "",
        "## 2. Clustering metrics",
        "",
        "| Method | k | Silhouette | Calinski-Harabasz | Davies-Bouldin |",
        "|---|---|---|---|---|",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"| {row['method']} | {int(row['k'])} | {row['silhouette']:.4f} | "
            f"{row['calinski_harabasz']:.2f} | {row['davies_bouldin']:.4f} |"
        )
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))

    print(json.dumps({"out_dir": str(out_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
