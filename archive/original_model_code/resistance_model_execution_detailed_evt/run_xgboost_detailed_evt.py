"""
XGBoost script for the detailed EVT Resistance pathway.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


BASE_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/MGWR_ready_table.parquet"
)
DETAIL_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_modeling_detailed_evt/detailed_evt_candidate_table.parquet"
)

BASE_PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_1km_z",
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

DETAILED_EVT_PREDICTORS = [
    "EVT2022_group_is_shrub",
    "EVT2022_group_is_deciduous",
    "EVT2022_group_is_mixed",
    "EVT2022_group_is_conifer",
    "EVT2022_is_code_7028",
    "EVT2022_is_code_7043",
    "EVT2022_is_code_7045",
    "EVT2022_is_code_7027",
    "EVT2022_is_code_7037",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-input", default=str(BASE_INPUT))
    parser.add_argument("--detail-input", default=str(DETAIL_INPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    base_input = Path(args.base_input)
    detail_input = Path(args.detail_input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cols = ["pixel_id", "Resistance"] + BASE_PREDICTORS
    detail_cols = ["pixel_id"] + DETAILED_EVT_PREDICTORS

    base_df = pd.read_parquet(base_input, columns=base_cols)
    detail_df = pd.read_parquet(detail_input, columns=detail_cols)
    work = base_df.merge(detail_df, on="pixel_id", how="inner", validate="one_to_one")
    predictors = BASE_PREDICTORS + DETAILED_EVT_PREDICTORS
    work = work[["Resistance"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = work[predictors]
    y = work["Resistance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    imp = pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in score.items()]).sort_values(
        "importance_gain", ascending=False
    )
    imp.to_csv(out_dir / "xgb_detailed_evt_feature_importance.csv", index=False)

    metrics = {
        "base_input_file": str(base_input),
        "detail_input_file": str(detail_input),
        "join_key": "pixel_id",
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "base_predictors": BASE_PREDICTORS,
        "detailed_evt_predictors": DETAILED_EVT_PREDICTORS,
        "predictors": predictors,
        "detailed_evt_strategy": "grouped_indicators_plus_top5_code_indicators",
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    (out_dir / "xgb_detailed_evt_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
