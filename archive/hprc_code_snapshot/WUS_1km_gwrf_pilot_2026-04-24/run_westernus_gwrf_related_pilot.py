#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
TODAY = date.today().strftime("%Y-%m-%d")

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_TREES = 300
DEFAULT_BLOCK_KM = 100.0
DEFAULT_RESPONSE = "Resistance"
DEFAULT_VARIANT = "E_EVT_fireregime_postclim"
DEFAULT_NEIGHBOR_K = 16
MORAN_K = 8

FIRE_REGIME_MAP = {
    7053: "surface_fire", 7054: "surface_fire", 7031: "surface_fire",
    7017: "surface_fire", 7019: "surface_fire", 7020: "surface_fire",
    7022: "surface_fire", 7035: "surface_fire", 7036: "surface_fire",
    7063: "surface_fire",
    7050: "stand_replacing", 7055: "stand_replacing", 7056: "stand_replacing",
    7046: "stand_replacing", 7041: "stand_replacing", 7044: "stand_replacing",
    7032: "stand_replacing", 7033: "stand_replacing", 7058: "stand_replacing",
    7061: "stand_replacing", 7062: "stand_replacing", 7113: "stand_replacing",
    7114: "stand_replacing", 7118: "stand_replacing",
    7045: "mixed_severity", 7047: "mixed_severity", 7166: "mixed_severity",
    7051: "mixed_severity", 7018: "mixed_severity", 7027: "mixed_severity",
    7028: "mixed_severity", 7172: "mixed_severity", 7030: "mixed_severity",
    7265: "mixed_severity",
    7037: "coastal_mixed", 7039: "coastal_mixed", 7042: "coastal_mixed",
    7174: "coastal_mixed", 7043: "coastal_mixed", 7014: "coastal_mixed",
    7015: "coastal_mixed",
    7011: "hardwood", 7029: "hardwood", 7010: "hardwood",
    7008: "hardwood", 7021: "hardwood",
}

BASE_PREDS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z",
    "TS_twi_z", "TS_roughness_z", "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z", "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_z", "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_pre_z", "CLIM_eto_sum_pre_z", "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z", "CLIM_aridity_pre_z", "CLIM_tmmx_std_pre_z",
    "CLIM_vpd_mean_pre_z", "CLIM_vpd_std_pre_z",
    "x", "y", "x_sq_z", "y_sq_z", "xy_z",
]
EVT_PREDS = ["FS_EVT_resistance_proxy_z", "FS_EVT_regeneration_proxy_z"]

BASE_TO_Z = {
    "TS_elev_m_z": "TS_elev_m",
    "TS_slope_deg_z": "TS_slope_deg",
    "TS_northness_z": "TS_northness",
    "TS_eastness_z": "TS_eastness",
    "TS_twi_z": "TS_twi",
    "TS_roughness_z": "TS_roughness",
    "TS_SOC_0_30cm_z": "TS_SOC_0_30cm",
    "FS_TCC_t0_z": "FS_TCC_t0",
    "FS_CBH_t0agg_z": "FS_CBH_t0agg",
    "HUM_roaddens_r5km_z": "HUM_roaddens_r5km",
    "HUM_traildens_r10km_z": "HUM_traildens_r10km",
    "HUM_imperv_near_t0_z": "HUM_imperv_near_t0",
    "CLIM_pr_sum_pre_z": "CLIM_pr_sum_pre",
    "CLIM_eto_sum_pre_z": "CLIM_eto_sum_pre",
    "CLIM_tmmn_mean_pre_z": "CLIM_tmmn_mean_pre",
    "CLIM_hot_days_35C_pre_z": "CLIM_hot_days_35C_pre",
    "CLIM_aridity_pre_z": "CLIM_aridity_pre",
    "CLIM_tmmx_std_pre_z": "CLIM_tmmx_std_pre",
    "CLIM_vpd_mean_pre_z": "CLIM_vpd_mean_pre",
    "CLIM_vpd_std_pre_z": "CLIM_vpd_std_pre",
    "FS_EVT_resistance_proxy_z": "FS_EVT_resistance_proxy",
    "FS_EVT_regeneration_proxy_z": "FS_EVT_regeneration_proxy",
}
SPACE_COLS = {"x", "y", "x_sq_z", "y_sq_z", "xy_z"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GW-RF-related pilot: compare global RF vs spatial RF "
            "with neighbor-summary features under random and block validation."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--response", default=DEFAULT_RESPONSE)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--trees", type=int, default=DEFAULT_TREES)
    parser.add_argument("--block-km", type=float, default=DEFAULT_BLOCK_KM)
    parser.add_argument("--neighbor-k", type=int, default=DEFAULT_NEIGHBOR_K)
    parser.add_argument("--sample-n", type=int, default=None)
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


def ensure_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    for z_col, raw_col in BASE_TO_Z.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    for col, expr in [
        ("x_sq_z", out["x"] ** 2),
        ("y_sq_z", out["y"] ** 2),
        ("xy_z", out["x"] * out["y"]),
    ]:
        if col not in out.columns:
            out[col] = zscore(expr)

    regime = out["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    dummies = pd.get_dummies(regime, prefix="EVT_regime").astype(np.float32)
    if "EVT_regime_other" in dummies.columns:
        dummies = dummies.drop(columns=["EVT_regime_other"])
    regime_cols = list(dummies.columns)
    out = pd.concat([out, dummies], axis=1)
    return out, regime_cols


def build_variants(regime_cols: list[str]) -> OrderedDict[str, dict]:
    return OrderedDict(
        [
            (
                "C_EVTproxy_postclim",
                {
                    "label": "C: baseline + EVT proxy",
                    "predictors": BASE_PREDS + EVT_PREDS,
                },
            ),
            (
                "E_EVT_fireregime_postclim",
                {
                    "label": "E: baseline + EVT proxy + fire-regime dummies",
                    "predictors": BASE_PREDS + EVT_PREDS + regime_cols,
                },
            ),
        ]
    )


def prepare_work(
    df: pd.DataFrame, response: str, predictors: list[str], sample_n: int | None
) -> pd.DataFrame:
    cols = list(dict.fromkeys([response, "x", "y", *predictors]))
    work = (
        df[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    if sample_n is not None and sample_n < len(work):
        work = work.sample(n=sample_n, random_state=RANDOM_STATE).reset_index(drop=True)
    return work


def block_groups(df: pd.DataFrame, block_km: float) -> pd.Series:
    block_m = block_km * 1000.0
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    labels = [
        f"{int(np.floor(xi / block_m))}_{int(np.floor(yi / block_m))}"
        for xi, yi in zip(x, y)
    ]
    return pd.Series(labels, index=df.index)


def build_splits(
    work: pd.DataFrame, block_km: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(work))
    random_train_idx, random_test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    groups = block_groups(work, block_km)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    block_train_idx, block_test_idx = next(gss.split(idx, groups=groups))
    return {
        "random": (random_train_idx, random_test_idx),
        "block": (block_train_idx, block_test_idx),
    }


def fit_rf(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def compute_moran(coords_df: pd.DataFrame, residuals: np.ndarray) -> float:
    k = min(MORAN_K, len(coords_df) - 1)
    if k < 1:
        return float("nan")
    weights = KNN.from_array(coords_df[["x", "y"]].to_numpy(), k=k)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return float(moran.I)


def neighbor_mean_features(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: list[str],
    neighbor_k: int,
    is_train_target: bool,
) -> pd.DataFrame:
    train_coords = train_df[["x", "y"]].to_numpy(dtype=float)
    target_coords = target_df[["x", "y"]].to_numpy(dtype=float)
    train_values = train_df[feature_cols].to_numpy(dtype=float)

    n_train = len(train_df)
    if n_train <= 1:
        raise RuntimeError("Not enough training rows to build neighborhood features.")

    if is_train_target:
        n_neighbors = min(neighbor_k + 1, n_train)
        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
        knn.fit(train_coords)
        idx = knn.kneighbors(train_coords, return_distance=False)
        if n_neighbors > 1:
            idx = idx[:, 1:]
        else:
            idx = idx[:, :0]
    else:
        n_neighbors = min(neighbor_k, n_train)
        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
        knn.fit(train_coords)
        idx = knn.kneighbors(target_coords, return_distance=False)

    if idx.shape[1] == 0:
        raise RuntimeError("Neighborhood index matrix is empty.")

    features = {
        f"nn{neighbor_k}_mean_{col}": train_values[idx, j].mean(axis=1)
        for j, col in enumerate(feature_cols)
    }
    return pd.DataFrame(features, index=target_df.index, dtype=np.float32)


def top_importance_rows(
    model: RandomForestRegressor, predictors: list[str], top_n: int = 15
) -> list[dict]:
    imp = (
        pd.Series(model.feature_importances_, index=predictors)
        .sort_values(ascending=False)
        .head(top_n)
    )
    rows: list[dict] = []
    for rank, (feature, importance) in enumerate(imp.items(), 1):
        rows.append(
            {
                "rank": rank,
                "feature": feature,
                "importance": float(importance),
            }
        )
    return rows


def evaluate_split(
    work: pd.DataFrame,
    predictors: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_estimators: int,
    neighbor_k: int,
) -> tuple[dict, dict, list[dict], list[dict]]:
    train = work.iloc[train_idx].copy()
    test = work.iloc[test_idx].copy()

    y_train = train["response"]
    y_test = test["response"].to_numpy()

    global_model = fit_rf(train[predictors], y_train, n_estimators)
    global_pred = global_model.predict(test[predictors])
    global_metrics = metric_dict(y_test, global_pred)
    global_metrics["moran_i"] = compute_moran(
        test[["x", "y"]], y_test - global_pred
    )

    spatial_feature_cols = [c for c in predictors if c not in SPACE_COLS]
    train_spatial = neighbor_mean_features(
        train, train, spatial_feature_cols, neighbor_k, is_train_target=True
    )
    test_spatial = neighbor_mean_features(
        train, test, spatial_feature_cols, neighbor_k, is_train_target=False
    )
    spatial_train = pd.concat([train.reset_index(drop=True), train_spatial.reset_index(drop=True)], axis=1)
    spatial_test = pd.concat([test.reset_index(drop=True), test_spatial.reset_index(drop=True)], axis=1)
    extra_cols = list(train_spatial.columns)
    spatial_predictors = predictors + extra_cols

    spatial_model = fit_rf(
        spatial_train[spatial_predictors],
        spatial_train["response"],
        n_estimators,
    )
    spatial_pred = spatial_model.predict(spatial_test[spatial_predictors])
    spatial_metrics = metric_dict(y_test, spatial_pred)
    spatial_metrics["moran_i"] = compute_moran(
        test[["x", "y"]], y_test - spatial_pred
    )

    global_importance = top_importance_rows(global_model, predictors)
    spatial_importance = top_importance_rows(spatial_model, spatial_predictors)
    return global_metrics, spatial_metrics, global_importance, spatial_importance


def build_report(
    response: str,
    variant: str,
    label: str,
    predictors: list[str],
    rows_used: int,
    neighbor_k: int,
    block_km: float,
    n_estimators: int,
    summary_df: pd.DataFrame,
    importance_global: pd.DataFrame,
    importance_spatial: pd.DataFrame,
) -> str:
    lines = [
        f"# GW-RF-Related Pilot - {response} - {variant} ({TODAY})",
        "",
        f"**Input:** `{INPUT.name}`  ",
        f"**Response:** `{response}`  ",
        f"**Variant:** `{variant}`  ",
        f"**Variant label:** {label}  ",
        f"**Rows used:** {rows_used:,}  ",
        f"**Base predictors:** {len(predictors)}  ",
        f"**Neighbor k:** {neighbor_k}  ",
        f"**Block size:** {block_km:g} km  ",
        f"**Trees per RF:** {n_estimators}",
        "",
        "This pilot uses a spatial-RF-style neighborhood summary design as a GW-RF-related first pass.",
        "",
        "## 1. Global RF vs Spatial RF summary",
        "",
        "| Split | Model | R2 | RMSE | Residual Moran's I |",
        "|---|---|---|---|---|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['split']} | {row['model']} | {row['r2']:.4f} | "
            f"{row['rmse']:.4f} | {row['moran_i']:.4f} |"
        )

    lines += [
        "",
        "## 2. Top importances - global RF (block split)",
        "",
        "| Rank | Feature | Importance |",
        "|---|---|---|",
    ]
    for _, row in importance_global.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['feature']} | {row['importance']:.6f} |"
        )

    lines += [
        "",
        "## 3. Top importances - spatial RF (block split)",
        "",
        "| Rank | Feature | Importance |",
        "|---|---|---|",
    ]
    for _, row in importance_spatial.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['feature']} | {row['importance']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or ROOT / (
        f"gwrf_related_pilot_{args.variant}_k{args.neighbor_k}_{TODAY}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    df, regime_cols = ensure_columns(df)
    variants = build_variants(regime_cols)
    if args.variant not in variants:
        raise ValueError(f"Unknown variant: {args.variant}")

    config = variants[args.variant]
    predictors = config["predictors"]
    work = prepare_work(df, args.response, predictors, args.sample_n).rename(
        columns={args.response: "response"}
    )
    splits = build_splits(work, args.block_km)

    summary_rows: list[dict] = []
    global_importance_rows: list[dict] = []
    spatial_importance_rows: list[dict] = []

    for split_name, (train_idx, test_idx) in splits.items():
        global_metrics, spatial_metrics, global_imp, spatial_imp = evaluate_split(
            work,
            predictors,
            train_idx,
            test_idx,
            args.trees,
            args.neighbor_k,
        )
        summary_rows.append({"split": split_name, "model": "global_rf", **global_metrics})
        summary_rows.append({"split": split_name, "model": f"spatial_rf_knn{args.neighbor_k}", **spatial_metrics})
        if split_name == "block":
            for row in global_imp:
                row["split"] = split_name
                global_importance_rows.append(row)
            for row in spatial_imp:
                row["split"] = split_name
                spatial_importance_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    global_importance_df = pd.DataFrame(global_importance_rows)
    spatial_importance_df = pd.DataFrame(spatial_importance_rows)

    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)
    global_importance_df.to_csv(out_dir / "block_global_importance.csv", index=False)
    spatial_importance_df.to_csv(out_dir / "block_spatial_importance.csv", index=False)

    metadata = {
        "input": str(args.input),
        "response": args.response,
        "variant": args.variant,
        "variant_label": config["label"],
        "rows_used": int(len(work)),
        "neighbor_k": args.neighbor_k,
        "trees": args.trees,
        "block_km": args.block_km,
        "sample_n": args.sample_n,
        "predictors": predictors,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    report = build_report(
        args.response,
        args.variant,
        config["label"],
        predictors,
        len(work),
        args.neighbor_k,
        args.block_km,
        args.trees,
        summary_df.sort_values(["split", "model"]).reset_index(drop=True),
        global_importance_df,
        spatial_importance_df,
    )
    (out_dir / "report.md").write_text(report)
    print(json.dumps({"out_dir": str(out_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
