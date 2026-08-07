from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_model_execution_near_t0_aggregated/MGWR_ready_table_near_t0_aggregated.parquet"
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


def build_model(random_state: int) -> XGBRegressor:
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


def compute_residual_moran(work: pd.DataFrame, predictors: list[str], random_state: int, k: int) -> dict:
    weights = KNN.from_array(work[["x", "y"]].to_numpy(), k=k)
    weights.transform = "R"
    model = build_model(random_state)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--extra-predictors", nargs="*", default=[])
    parser.add_argument("--compute-full-fit-moran", action="store_true")
    parser.add_argument("--moran-k", type=int, default=8)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(Path(args.input))
    predictors = list(PREDICTORS)
    for col in args.extra_predictors:
        if col not in predictors:
            predictors.append(col)
    cols = ["Resistance"] + predictors
    if args.compute_full_fit_moran:
        for col in ("x", "y"):
            if col not in cols:
                cols.append(col)
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[predictors]
    y = work["Resistance"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)
    model = build_model(args.random_state)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.get_booster().get_score(importance_type="gain")
    pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in score.items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(out_dir / "xgb_near_t0_aggregated_feature_importance.csv", index=False)
    metrics = {
        "input_file": str(args.input),
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "predictors": predictors,
        "extra_predictors": args.extra_predictors,
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    if args.compute_full_fit_moran:
        metrics["full_fit_residual_moran"] = compute_residual_moran(work, predictors, args.random_state, args.moran_k)
    (out_dir / "xgb_near_t0_aggregated_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
