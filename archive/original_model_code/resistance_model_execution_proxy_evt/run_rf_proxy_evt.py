"""
Random Forest script for the proxy EVT Resistance pathway.

Intended use:
- local machine or Google Colab

Defaults:
- input file: MGWR_model_input.parquet
- response: Resistance
- predictor file: predictors_rf_xgb_proxy_evt.txt from resistance_modeling_plan

Outputs:
- rf_proxy_evt_feature_importance.csv
- rf_proxy_evt_metrics.json

Dependencies:
- pandas
- numpy
- scikit-learn
- pyarrow or fastparquet
"""

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
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/MGWR_model_input.parquet"
)
DEFAULT_PREDICTORS = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_modeling_plan/predictors_rf_xgb_proxy_evt.txt"
)


def read_predictors(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


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

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    imp = pd.DataFrame(
        {"predictor": predictors, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp.to_csv(out_dir / "rf_proxy_evt_feature_importance.csv", index=False)

    metrics = {
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "predictors": predictors,
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    (out_dir / "rf_proxy_evt_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
