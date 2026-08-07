"""
Simplified GWR script for the proxy EVT Resistance pathway.

Intended use:
- local machine or Google Colab
- this is a simplified GWR, not MGWR

Defaults:
- input file: MGWR_model_input.parquet
- response: Resistance
- predictor file: predictors_gwr_proxy_evt.txt from resistance_modeling_plan

Outputs:
- gwr_proxy_evt_summary.txt
- gwr_proxy_evt_metadata.json
- gwr_proxy_evt_coefficients.parquet

Dependencies:
- pandas
- numpy
- mgwr
- pyarrow or fastparquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/MGWR_model_input.parquet"
)
DEFAULT_PREDICTORS = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_modeling_plan/predictors_gwr_proxy_evt.txt"
)


def read_predictors(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--predictors-file", default=str(DEFAULT_PREDICTORS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bw-min", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input)
    predictors_file = Path(args.predictors_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    predictors = read_predictors(predictors_file)
    cols = ["pixel_id", "row", "col", "x", "y", "Resistance"] + predictors
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    coords = work[["x", "y"]].to_numpy(dtype=float)
    y = work[["Resistance"]].to_numpy(dtype=float)
    X = work[predictors].to_numpy(dtype=float)

    selector = Sel_BW(coords, y, X, multi=False)
    bw = selector.search(bw_min=args.bw_min)
    model = GWR(coords, y, X, bw=bw, fixed=False, kernel="bisquare")
    results = model.fit()

    (out_dir / "gwr_proxy_evt_summary.txt").write_text(str(results.summary()))
    meta = {
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "response": "Resistance",
        "predictors": predictors,
        "n_rows_used": int(len(work)),
        "bw_min": int(args.bw_min),
        "selected_bw": float(bw),
    }
    (out_dir / "gwr_proxy_evt_metadata.json").write_text(json.dumps(meta, indent=2))

    coef_df = work[["pixel_id", "row", "col", "x", "y", "Resistance"]].copy()
    params = results.params
    coef_df["intercept"] = params[:, 0]
    for i, col in enumerate(predictors):
        coef_df[f"coef_{col}"] = params[:, i + 1]
    coef_df.to_parquet(out_dir / "gwr_proxy_evt_coefficients.parquet", index=False)


if __name__ == "__main__":
    main()
