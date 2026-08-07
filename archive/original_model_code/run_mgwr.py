#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MGWR on a parquet input file.")
    parser.add_argument(
        "--input",
        default="MGWR_model_input.parquet",
        help="Path to input parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="mgwr_outputs",
        help="Directory where MGWR outputs will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    response_col = "Resistance"
    coord_cols = ["x", "y"]
    predictor_cols = [col for col in df.columns if col not in {response_col, *coord_cols}]

    missing_required = [col for col in [response_col, *coord_cols] if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    if not predictor_cols:
        raise ValueError("No predictor columns found after excluding response and coordinate columns.")

    required_frame = df[[response_col, *coord_cols, *predictor_cols]]
    if required_frame.isna().any().any():
        raise ValueError("Input contains missing values. Refusing to modify or drop data.")

    numeric_values = required_frame.to_numpy(dtype=float)
    if not np.isfinite(numeric_values).all():
        raise ValueError("Input contains non-finite values. Refusing to modify or drop data.")

    y = df[[response_col]].to_numpy(dtype=float)
    X = df[predictor_cols].to_numpy(dtype=float)
    coords = df[coord_cols].to_numpy(dtype=float)

    selector = Sel_BW(coords, y, X, multi=True)
    bandwidths = selector.search()

    model = MGWR(coords, y, X, selector=selector)
    results = model.fit()

    coef_columns = ["Intercept", *predictor_cols]
    coefficients = pd.DataFrame(results.params, columns=coef_columns)
    coefficient_output = pd.concat(
        [df[coord_cols].reset_index(drop=True), coefficients.reset_index(drop=True)],
        axis=1,
    )
    coefficient_output.to_parquet(output_dir / "mgwr_coefficients.parquet", index=False)

    residual_output = df[coord_cols].copy()
    residual_output["observed"] = y.ravel()
    residual_output["predicted"] = results.predy.ravel()
    residual_output["residual"] = results.resid_response.ravel()
    residual_output.to_parquet(output_dir / "mgwr_residuals.parquet", index=False)

    bandwidth_output = pd.DataFrame(
        {"term": coef_columns, "bandwidth": np.asarray(bandwidths).ravel()}
    )
    bandwidth_output.to_csv(output_dir / "mgwr_bandwidths.csv", index=False)

    metadata = {
        "input_path": str(input_path),
        "n_rows": int(df.shape[0]),
        "n_predictors": int(len(predictor_cols)),
        "predictor_columns": predictor_cols,
        "response_column": response_col,
        "coordinate_columns": coord_cols,
    }
    (output_dir / "mgwr_run_metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
