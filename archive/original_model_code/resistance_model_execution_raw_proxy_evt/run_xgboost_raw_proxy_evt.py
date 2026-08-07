"""
XGBoost script for the raw proxy EVT Resistance pathway.

Intended use:
- local machine or Google Colab

Defaults:
- input file: MGWR_ready_table.parquet
- response: Resistance
- predictor source-of-truth: predictors_rf_xgb_proxy_evt.txt from resistance_modeling_plan

Key difference from proxy-z package:
- FS_EVT_resistance_proxy_z is replaced by raw/original FS_EVT_resistance_proxy

Outputs:
- xgb_raw_proxy_evt_feature_importance.csv
- xgb_raw_proxy_evt_metrics.json

Dependencies:
- pandas
- numpy
- xgboost
- scikit-learn
- pyarrow or fastparquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/MGWR_ready_table.parquet"
)
DEFAULT_PREDICTORS = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_modeling_plan/predictors_rf_xgb_proxy_evt.txt"
)


def read_predictors(path: Path):
    predictors = [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return [
        "FS_EVT_resistance_proxy" if c == "FS_EVT_resistance_proxy_z" else c
        for c in predictors
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--predictors-file", default=str(DEFAULT_PREDICTORS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    predictors_file = Path(args.predictors_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    predictors = read_predictors(predictors_file)
    cols = ["Resistance"] + predictors
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

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
    imp = pd.DataFrame(
        [{"predictor": k, "importance_gain": v} for k, v in score.items()]
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(out_dir / "xgb_raw_proxy_evt_feature_importance.csv", index=False)

    metrics = {
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "predictors": predictors,
        "replaced_predictor": {
            "old": "FS_EVT_resistance_proxy_z",
            "new": "FS_EVT_resistance_proxy",
        },
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    (out_dir / "xgb_raw_proxy_evt_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
