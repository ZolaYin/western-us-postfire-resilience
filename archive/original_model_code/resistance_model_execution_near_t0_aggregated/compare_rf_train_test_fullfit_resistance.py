from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "resistance_model_execution_near_t0_aggregated" / "MGWR_ready_table_near_t0_aggregated.parquet"
FULL_SAF_META = ROOT / "resistance_full_saf_models_2026-03-30" / "full_saf_metadata.json"
OUT_CSV = ROOT / "resistance_model_execution_near_t0_aggregated" / "rf_train_test_fullfit_comparison_2026-04-01.csv"
OUT_JSON = ROOT / "resistance_model_execution_near_t0_aggregated" / "rf_train_test_fullfit_comparison_2026-04-01.json"

RESPONSE = "Resistance"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500

BASE_WITH_EVT_PROXY = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "FS_EVT_t0agg_resistance_proxy",
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
]

BASE_NO_EVT = [
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
]


def load_full_saf_predictors() -> list[str]:
    meta = json.loads(FULL_SAF_META.read_text())
    return meta["base_predictors"] + meta["full_saf_indicator_columns"]


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=float), index=vals.index)
    return (vals - vals.mean()) / std


def build_full_saf(df: pd.DataFrame, saf_indicator_columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    saf = pd.to_numeric(out["FS_EVT_t0agg_SAF_code"], errors="coerce").astype("Int64")
    for col in saf_indicator_columns:
        code = int(col.split("_")[-1])
        out[col] = (saf == code).fillna(False).astype(int)
    return out


def add_poly_xy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x_sq_z"] = zscore(out["x"] ** 2)
    out["y_sq_z"] = zscore(out["y"] ** 2)
    out["xy_z"] = zscore(out["x"] * out["y"])
    return out


def evaluate_variant(df: pd.DataFrame, model_name: str, predictors: list[str], evt_mode: str, xy_mode: str) -> dict:
    cols = [RESPONSE] + predictors
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[predictors]
    y = work[RESPONSE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model_split = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model_split.fit(X_train, y_train)
    pred_train = model_split.predict(X_train)
    pred_test = model_split.predict(X_test)

    model_full = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model_full.fit(X, y)
    pred_full = model_full.predict(X)

    return {
        "model_name": model_name,
        "evt_mode": evt_mode,
        "xy_mode": xy_mode,
        "n_rows_used": int(len(work)),
        "predictor_count": int(len(predictors)),
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "full_fit_r2": float(r2_score(y, pred_full)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_fit_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
        "predictors": predictors,
    }


def main() -> None:
    df = pd.read_parquet(INPUT)
    full_saf_predictors = load_full_saf_predictors()
    df = build_full_saf(df, [col for col in full_saf_predictors if col.startswith("FS_EVT_t0agg_SAF_")])
    df = add_poly_xy(df)

    variants = [
        ("RF_noEVT_noXY", BASE_NO_EVT, "none", "none"),
        ("RF_noEVT_plusXY", BASE_NO_EVT + ["x", "y"], "none", "xy"),
        ("RF_EVTproxy_noXY", BASE_WITH_EVT_PROXY, "proxy", "none"),
        ("RF_EVTproxy_plusXY", BASE_WITH_EVT_PROXY + ["x", "y"], "proxy", "xy"),
        ("RF_fullSAF_noXY", full_saf_predictors, "full_saf", "none"),
        ("RF_fullSAF_plusXY", full_saf_predictors + ["x", "y"], "full_saf", "xy"),
        ("RF_fullSAF_plusXY_poly", full_saf_predictors + ["x", "y", "x_sq_z", "y_sq_z", "xy_z"], "full_saf", "xy_poly"),
    ]

    rows = [
        evaluate_variant(df=df, model_name=name, predictors=predictors, evt_mode=evt_mode, xy_mode=xy_mode)
        for name, predictors, evt_mode, xy_mode in variants
    ]

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(out_df[["model_name", "train_r2", "test_r2", "full_fit_r2", "test_rmse"]].to_string(index=False))


if __name__ == "__main__":
    main()
