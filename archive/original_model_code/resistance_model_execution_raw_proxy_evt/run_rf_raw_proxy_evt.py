"""
Random Forest script for the raw proxy EVT Resistance pathway.

Intended use:
- local machine or Google Colab

Defaults:
- input file: MGWR_ready_table.parquet
- response: Resistance
- predictor source-of-truth: predictors_rf_xgb_proxy_evt.txt from resistance_modeling_plan

Key difference from proxy-z package:
- FS_EVT_resistance_proxy_z is replaced by raw/original FS_EVT_resistance_proxy

Outputs:
- rf_raw_proxy_evt_feature_importance.csv
- rf_raw_proxy_evt_metrics.json

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
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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


def compute_residual_moran(work: pd.DataFrame, predictors: list[str], random_state: int, k: int):
    coords = work[["x", "y"]].to_numpy()
    weights = KNN.from_array(coords, k=k)
    weights.transform = "R"

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(work[predictors], work["Resistance"])
    residuals = work["Resistance"] - model.predict(work[predictors])
    moran = Moran(residuals.to_numpy(), weights, permutations=0)
    return {
        "k": int(k),
        "n_obs": int(len(work)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--predictors-file", default=str(DEFAULT_PREDICTORS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--extra-predictors",
        nargs="*",
        default=[],
        help="Optional extra predictors to append, e.g. x y",
    )
    parser.add_argument(
        "--compute-full-fit-moran",
        action="store_true",
        help="Refit on the full table and compute residual Moran's I using x/y coordinates",
    )
    parser.add_argument(
        "--moran-k",
        type=int,
        default=8,
        help="Neighbor count for KNN weights used in residual Moran's I",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    predictors_file = Path(args.predictors_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    predictors = read_predictors(predictors_file)
    for col in args.extra_predictors:
        if col not in predictors:
            predictors.append(col)

    cols = ["Resistance"] + predictors
    needs_coords = args.compute_full_fit_moran and not {"x", "y"}.issubset(cols)
    if needs_coords:
        cols.extend(["x", "y"])
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
    imp.to_csv(out_dir / "rf_raw_proxy_evt_feature_importance.csv", index=False)

    metrics = {
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "predictors": predictors,
        "extra_predictors": args.extra_predictors,
        "replaced_predictor": {
            "old": "FS_EVT_resistance_proxy_z",
            "new": "FS_EVT_resistance_proxy",
        },
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    if args.compute_full_fit_moran:
        metrics["full_fit_residual_moran"] = compute_residual_moran(
            work=work,
            predictors=predictors,
            random_state=args.random_state,
            k=args.moran_k,
        )
    (out_dir / "rf_raw_proxy_evt_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
