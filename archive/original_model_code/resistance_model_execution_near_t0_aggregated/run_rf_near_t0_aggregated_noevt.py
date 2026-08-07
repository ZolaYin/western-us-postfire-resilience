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


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_model_execution_near_t0_aggregated/MGWR_ready_table_near_t0_aggregated.parquet"
)
OUT_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_model_execution_near_t0_aggregated/rf_run_noevt_noxy_2026-04-01"
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


def compute_residual_moran(work: pd.DataFrame, predictors: list[str], random_state: int = 42, k: int = 8) -> dict:
    weights = KNN.from_array(work[["x", "y"]].to_numpy(), k=k)
    weights.transform = "R"
    model = RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    model.fit(work[predictors], work["Resistance"])
    resid = work["Resistance"] - model.predict(work[predictors])
    moran = Moran(resid.to_numpy(), weights, permutations=0)
    return {
        "k": int(k),
        "n_obs": int(len(work)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DEFAULT_INPUT)
    cols = ["Resistance", "x", "y"] + PREDICTORS
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[PREDICTORS]
    y = work["Resistance"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    pd.DataFrame({"predictor": PREDICTORS, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(OUT_DIR / "rf_near_t0_aggregated_noevt_feature_importance.csv", index=False)

    metrics = {
        "input_file": str(DEFAULT_INPUT),
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "predictors": PREDICTORS,
        "extra_predictors": [],
        "evt_included": False,
        "xy_included": False,
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "full_fit_residual_moran": compute_residual_moran(work, PREDICTORS, 42, 8),
    }
    (OUT_DIR / "rf_near_t0_aggregated_noevt_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
