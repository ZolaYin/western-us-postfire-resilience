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

ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
LATENT_ASSIGN = ROOT / "stage5b_latent_k6_followup_2026-04-26" / "latent_k6_assignments.csv"
OUT_DIR = ROOT / f"latent_k6_rf_compare_{date.today().strftime('%Y-%m-%d')}"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BLOCK_KM = 100.0
RF_TREES = 200
MORAN_K = 8

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--trees", type=int, default=RF_TREES)
    p.add_argument(
        "--models",
        nargs="+",
        default=["E_global", "E_plus_region", "E_plus_latent_k6", "E_plus_region_and_latent_k6"],
        choices=["E_global", "E_plus_region", "E_plus_latent_k6", "E_plus_region_and_latent_k6"],
    )
    return p.parse_args()


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
    if "HUM_popdens_win10km_log_z" not in out.columns and "HUM_popdens_win10km" in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns and "HUM_viirs_near_t0" in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    for col, expr in [("x_sq_z", out["x"] ** 2), ("y_sq_z", out["y"] ** 2), ("xy_z", out["x"] * out["y"])]:
        if col not in out.columns:
            out[col] = zscore(expr)
    regime = out["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    dummies = pd.get_dummies(regime, prefix="EVT_regime").astype(np.float32)
    if "EVT_regime_other" in dummies.columns:
        dummies = dummies.drop(columns=["EVT_regime_other"])
    out = pd.concat([out, dummies], axis=1)
    return out, list(dummies.columns)


def block_groups(df: pd.DataFrame) -> pd.Series:
    block_m = BLOCK_KM * 1000.0
    labels = [
        f"{int(np.floor(x / block_m))}_{int(np.floor(y / block_m))}"
        for x, y in zip(df["x"], df["y"])
    ]
    return pd.Series(labels, index=df.index)


def compute_moran(coords_df: pd.DataFrame, residuals: np.ndarray) -> float:
    k = min(MORAN_K, len(coords_df) - 1)
    if k < 1:
        return float("nan")
    weights = KNN.from_array(coords_df[["x", "y"]].to_numpy(), k=k)
    weights.transform = "R"
    return float(Moran(residuals.astype(float), weights, permutations=0).I)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, coords_df: pd.DataFrame) -> dict[str, float]:
    residuals = y_true - y_pred
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "moran_i": compute_moran(coords_df, residuals),
    }


def eta_squared_from_groups(values: pd.Series, groups: pd.Series) -> float:
    work = pd.DataFrame({"value": values.astype(float), "group": groups}).dropna()
    grand_mean = work["value"].mean()
    ss_total = ((work["value"] - grand_mean) ** 2).sum()
    if ss_total <= 0:
        return float("nan")
    group_stats = work.groupby("group")["value"].agg(["mean", "size"])
    ss_between = ((group_stats["mean"] - grand_mean) ** 2 * group_stats["size"]).sum()
    return float(ss_between / ss_total)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_parquet(INPUT)
    df, evt_cols = ensure_columns(df_raw)
    latent = pd.read_csv(LATENT_ASSIGN, usecols=["pixel_id", "latent_k6"])

    predictors_e = [c for c in BASE_PREDS + EVT_PREDS + evt_cols if c in df.columns]
    work = (
        df[list(dict.fromkeys(["pixel_id", "Resistance", "x", "y", "region"] + predictors_e))]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .merge(latent, on="pixel_id", how="inner")
        .reset_index(drop=True)
    )

    region_dummies = pd.get_dummies(work["region"], prefix="region", drop_first=True).astype(np.float32)
    latent_dummies = pd.get_dummies(work["latent_k6"], prefix="latent_k6", drop_first=True).astype(np.float32)
    work = pd.concat([work, region_dummies, latent_dummies], axis=1)

    all_model_specs = OrderedDict(
        [
            ("E_global", predictors_e),
            ("E_plus_region", predictors_e + list(region_dummies.columns)),
            ("E_plus_latent_k6", predictors_e + list(latent_dummies.columns)),
            ("E_plus_region_and_latent_k6", predictors_e + list(region_dummies.columns) + list(latent_dummies.columns)),
        ]
    )
    model_specs = OrderedDict((name, all_model_specs[name]) for name in args.models)

    idx = np.arange(len(work))
    rand_tr, rand_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    blk_tr, blk_te = next(gss.split(idx, groups=block_groups(work)))

    metrics_rows = []
    residual_rows = []
    eta_rows = []

    for model_name, predictors in model_specs.items():
        print(f"Running {model_name} ...", flush=True)
        pred_cache = {}
        for split_name, (tr, te) in [("random", (rand_tr, rand_te)), ("block", (blk_tr, blk_te))]:
            train = work.iloc[tr]
            test = work.iloc[te]
            model = RandomForestRegressor(n_estimators=args.trees, random_state=RANDOM_STATE, n_jobs=1)
            model.fit(train[predictors], train["Resistance"])
            pred = model.predict(test[predictors])
            y = test["Resistance"].to_numpy(dtype=float)
            metrics = metric_dict(y, pred, test[["x", "y"]])
            metrics_rows.append(
                {"model": model_name, "split": split_name, "rows": int(len(test)), "predictors": int(len(predictors)), **metrics}
            )
            pred_cache[split_name] = pred

        test_block = work.iloc[blk_te].copy()
        test_block["pred"] = pred_cache["block"]
        test_block["residual"] = test_block["Resistance"] - test_block["pred"]
        test_block["abs_residual"] = np.abs(test_block["residual"])

        for group_type, col in [("latent_k6", "latent_k6"), ("region", "region")]:
            summary = (
                test_block.groupby(col)
                .agg(
                    n=("Resistance", "size"),
                    resistance_mean=("Resistance", "mean"),
                    pred_mean=("pred", "mean"),
                    residual_mean=("residual", "mean"),
                    abs_residual_mean=("abs_residual", "mean"),
                )
                .reset_index()
                .rename(columns={col: "group"})
            )
            summary["model"] = model_name
            summary["group_type"] = group_type
            residual_rows.append(summary)

        eta_rows.append(
            {
                "model": model_name,
                "eta2_resistance_by_latent_k6": eta_squared_from_groups(test_block["Resistance"], test_block["latent_k6"]),
                "eta2_resistance_by_region": eta_squared_from_groups(test_block["Resistance"], test_block["region"]),
                "eta2_abs_residual_by_latent_k6": eta_squared_from_groups(test_block["abs_residual"], test_block["latent_k6"]),
                "eta2_abs_residual_by_region": eta_squared_from_groups(test_block["abs_residual"], test_block["region"]),
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)
    residual_df = pd.concat(residual_rows, ignore_index=True)
    eta_df = pd.DataFrame(eta_rows)

    metrics_df.to_csv(OUT_DIR / "latent_vs_region_rf_metrics.csv", index=False)
    residual_df.to_csv(OUT_DIR / "latent_vs_region_block_residual_summary.csv", index=False)
    eta_df.to_csv(OUT_DIR / "latent_vs_region_effect_sizes.csv", index=False)

    lines = [
        f"# Latent k=6 RF comparison ({date.today().strftime('%Y-%m-%d')})",
        "",
        "| Model | Split | Rows | Predictors | R2 | RMSE | Moran's I |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.sort_values(["split", "r2"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {row['model']} | {row['split']} | {int(row['rows'])} | {int(row['predictors'])} | {row['r2']:.4f} | {row['rmse']:.4f} | {row['moran_i']:.4f} |"
        )
    (OUT_DIR / "report.md").write_text("\n".join(lines))

    print(json.dumps({"out_dir": str(OUT_DIR), "rows_used": len(work)}))


if __name__ == "__main__":
    main()
