#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
BASE_METRICS = (
    ROOT
    / "westernus_rf_xgb_complete_corrected_diagnostics_20260411"
    / "westernus_rf_xgb_complete_corrected_metrics.csv"
)
OUT_DIR = ROOT / "xgb_tuning_complete_corrected_20260411"
OUT_TRIALS = OUT_DIR / "xgb_tuning_trials.csv"
OUT_BEST = OUT_DIR / "xgb_tuning_best_metrics.json"
OUT_REPORT = OUT_DIR / "xgb_tuning_report.md"

RESPONSE = "Resistance"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z",
    "HUM_roaddens_r5km_z",
    "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_eto_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_tmmx_std_pre_z",
    "x",
    "y",
]
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
}


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
    for z_col, base_col in BASE_TO_Z.items():
        if z_col not in out.columns:
            out[z_col] = zscore(out[base_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    return out


def build_model(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **params,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT).copy()
    df = ensure_columns(df)
    work = df[[RESPONSE] + PREDICTORS].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[PREDICTORS]
    y = work[RESPONSE]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    grid = list(
        itertools.product(
            [300, 500, 800],
            [3, 5, 7],
            [0.03, 0.05],
            [0.7, 0.85],
            [0.7, 0.9],
        )
    )
    trials = []
    for n_estimators, max_depth, learning_rate, subsample, colsample_bytree in grid:
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
        model = build_model(params)
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        trials.append(
            {
                **params,
                "val_r2": float(r2_score(y_val, pred_val)),
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
            }
        )

    trials_df = pd.DataFrame(trials).sort_values(["val_r2", "val_rmse"], ascending=[False, True]).reset_index(drop=True)
    trials_df.to_csv(OUT_TRIALS, index=False)
    best = trials_df.iloc[0].to_dict()
    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
    }

    best_model = build_model(best_params)
    best_model.fit(X_trainval, y_trainval)
    pred_test = best_model.predict(X_test)
    pred_full = best_model.predict(X)

    base_metrics = pd.read_csv(BASE_METRICS)
    base_xgb = base_metrics.loc[base_metrics["model"] == "XGBoost"].iloc[0].to_dict()
    result = {
        "input_table": str(INPUT),
        "rows_used": int(len(work)),
        "predictors": PREDICTORS,
        "best_params": best_params,
        "validation_r2": float(best["val_r2"]),
        "validation_rmse": float(best["val_rmse"]),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_r2": float(r2_score(y, pred_full)),
        "full_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
        "baseline_complete_xgb_test_r2": float(base_xgb["test_r2"]),
        "baseline_complete_xgb_test_rmse": float(base_xgb["test_rmse"]),
        "old_coast_xgb_plusxy_test_r2": 0.550707,
    }
    OUT_BEST.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# XGBoost Tuning On Complete Corrected Predictor Set",
        "",
        f"- Input table: `{INPUT}`",
        f"- Rows used: `{len(work)}`",
        f"- Best params: `{best_params}`",
        f"- Validation R2: `{result['validation_r2']:.6f}`",
        f"- Test R2: `{result['test_r2']:.6f}`",
        f"- Test RMSE: `{result['test_rmse']:.6f}`",
        f"- Baseline complete corrected XGB test R2: `{result['baseline_complete_xgb_test_r2']:.6f}`",
        f"- Old Coast XGB_plusXY reference: `{result['old_coast_xgb_plusxy_test_r2']:.6f}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
