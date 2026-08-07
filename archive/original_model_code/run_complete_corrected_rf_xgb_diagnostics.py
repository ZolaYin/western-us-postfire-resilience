#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
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
RF_COMPLETE_METRICS = (
    ROOT
    / "westernus_rf_noevt_plusxy_complete_corrected_20260411"
    / "westernus_rf_noevt_plusxy_complete_corrected_metrics.json"
)
OUT_DIR = ROOT / "westernus_rf_xgb_complete_corrected_diagnostics_20260411"
OUT_METRICS = OUT_DIR / "westernus_rf_xgb_complete_corrected_metrics.csv"
OUT_MORAN = OUT_DIR / "westernus_rf_xgb_complete_corrected_moran.csv"
OUT_XGB_IMPORTANCE = OUT_DIR / "westernus_xgb_noevt_plusxy_complete_corrected_feature_importance.csv"
OUT_RF_RESID = OUT_DIR / "westernus_rf_noevt_plusxy_complete_corrected_residuals.parquet"
OUT_XGB_RESID = OUT_DIR / "westernus_xgb_noevt_plusxy_complete_corrected_residuals.parquet"
OUT_REPORT = OUT_DIR / "westernus_rf_xgb_complete_corrected_report.md"

RESPONSE = "Resistance"
RANDOM_STATE = 42
TEST_SIZE = 0.2
K_MORAN = 8
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


def build_xgb(random_state: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )


def compute_moran(work_xy: pd.DataFrame, residuals: np.ndarray, k: int = K_MORAN) -> dict:
    weights = KNN.from_array(work_xy[["x", "y"]].to_numpy(), k=k)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return {
        "k": int(k),
        "n_obs": int(len(work_xy)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT).copy()
    df = ensure_columns(df)
    cols = list(dict.fromkeys(["pixel_id", "row", "col", "x", "y", RESPONSE] + PREDICTORS))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = work[PREDICTORS]
    y = work[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rf_full_metrics = json.loads(RF_COMPLETE_METRICS.read_text())

    rf_full_model = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_full_model.fit(X, y)
    rf_full_pred = rf_full_model.predict(X)
    rf_resid = y.to_numpy() - rf_full_pred
    rf_moran = compute_moran(work[["x", "y"]], rf_resid)
    rf_resid_df = work[["pixel_id", "row", "col", "x", "y", RESPONSE]].copy()
    rf_resid_df["prediction"] = rf_full_pred.astype(np.float32)
    rf_resid_df["residual"] = rf_resid.astype(np.float32)
    rf_resid_df.to_parquet(OUT_RF_RESID, index=False)

    xgb_eval = build_xgb(RANDOM_STATE)
    xgb_eval.fit(X_train, y_train)
    xgb_test_pred = xgb_eval.predict(X_test)
    xgb_full = build_xgb(RANDOM_STATE)
    xgb_full.fit(X, y)
    xgb_full_pred = xgb_full.predict(X)
    xgb_resid = y.to_numpy() - xgb_full_pred
    xgb_moran = compute_moran(work[["x", "y"]], xgb_resid)
    xgb_resid_df = work[["pixel_id", "row", "col", "x", "y", RESPONSE]].copy()
    xgb_resid_df["prediction"] = xgb_full_pred.astype(np.float32)
    xgb_resid_df["residual"] = xgb_resid.astype(np.float32)
    xgb_resid_df.to_parquet(OUT_XGB_RESID, index=False)

    score = xgb_full.get_booster().get_score(importance_type="gain")
    (
        pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in score.items()])
        .sort_values("importance_gain", ascending=False)
        .to_csv(OUT_XGB_IMPORTANCE, index=False)
    )

    coast_xgb_plusxy_test_r2 = 0.550707
    metrics = pd.DataFrame(
        [
            {
                "model": "RF",
                "variant": "noevt_plusxy_complete_corrected_westernus",
                "rows_used": int(len(work)),
                "test_r2": float(rf_full_metrics["test_r2"]),
                "test_rmse": float(rf_full_metrics["test_rmse"]),
                "full_r2": float(rf_full_metrics["full_r2"]),
                "full_rmse": float(rf_full_metrics["full_rmse"]),
            },
            {
                "model": "XGBoost",
                "variant": "noevt_plusxy_complete_corrected_westernus",
                "rows_used": int(len(work)),
                "test_r2": float(r2_score(y_test, xgb_test_pred)),
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, xgb_test_pred))),
                "full_r2": float(r2_score(y, xgb_full_pred)),
                "full_rmse": float(np.sqrt(mean_squared_error(y, xgb_full_pred))),
            },
        ]
    )
    moran_df = pd.DataFrame(
        [
            {"model": "RF", **rf_moran},
            {"model": "XGBoost", **xgb_moran},
        ]
    )
    metrics.to_csv(OUT_METRICS, index=False)
    moran_df.to_csv(OUT_MORAN, index=False)

    xgb_test_r2 = float(metrics.loc[metrics["model"] == "XGBoost", "test_r2"].iloc[0])
    lines = [
        "# Complete Corrected RF/XGBoost Diagnostics",
        "",
        f"- Input table: `{INPUT}`",
        f"- Rows used: `{len(work)}`",
        "",
        "RF:",
        f"- test R2: `{rf_full_metrics['test_r2']:.6f}`",
        f"- test RMSE: `{rf_full_metrics['test_rmse']:.6f}`",
        f"- full-fit residual Moran's I: `{rf_moran['moran_i']:.6f}`",
        "",
        "XGBoost:",
        f"- test R2: `{xgb_test_r2:.6f}`",
        f"- test RMSE: `{metrics.loc[metrics['model'] == 'XGBoost', 'test_rmse'].iloc[0]:.6f}`",
        f"- full-fit residual Moran's I: `{xgb_moran['moran_i']:.6f}`",
        "",
        "Old Coast reference:",
        "- RF_noEVT_plusXY test R2: `0.695254`",
        f"- XGB_plusXY test R2: `{coast_xgb_plusxy_test_r2:.6f}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(metrics.to_csv(index=False))
    print(moran_df.to_csv(index=False))


if __name__ == "__main__":
    main()
