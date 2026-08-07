from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_model_execution_near_t0_evt_cbh/MGWR_ready_table_near_t0_evt_cbh.parquet"
)
PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0_z",
    "FS_EVT_t0_resistance_proxy",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    work = df[["Resistance"] + PREDICTORS].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[PREDICTORS]
    y = work["Resistance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state
    )

    model = RandomForestRegressor(n_estimators=500, random_state=args.random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    pd.DataFrame({"predictor": PREDICTORS, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(out_dir / "rf_near_t0_evt_cbh_feature_importance.csv", index=False)
    metrics = {
        "input_file": str(input_path),
        "response": "Resistance",
        "n_rows_used": int(len(work)),
        "predictors": PREDICTORS,
        "replaced_predictors": {
            "evt": {"old": "FS_EVT_resistance_proxy", "new": "FS_EVT_t0_resistance_proxy"},
            "cbh": {"old": "FS_CBH_1km_z", "new": "FS_CBH_t0_z"},
        },
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    (out_dir / "rf_near_t0_evt_cbh_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
