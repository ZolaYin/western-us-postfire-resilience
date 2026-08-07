"""
Local-safe simplified GWR script for the raw proxy EVT Resistance pathway.

Why this file exists:
- preserves the original run_gwr_raw_proxy_evt.py unchanged
- forces n_jobs=1 to avoid local joblib/loky semaphore permission errors

Intended use:
- local machine or Google Colab
- multivariable GWR, not MGWR
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
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/MGWR_ready_table.parquet"
)
DEFAULT_PREDICTORS = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_modeling_plan/predictors_gwr_proxy_evt.txt"
)


def read_predictors(path: Path):
    predictors = [
        line.strip()
        for line in path.read_text().splitlines()
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

    selector = Sel_BW(coords, y, X, multi=False, n_jobs=1)
    bw = selector.search(bw_min=args.bw_min)
    model = GWR(coords, y, X, bw=bw, fixed=False, kernel="bisquare", n_jobs=1)
    results = model.fit()

    (out_dir / "gwr_raw_proxy_evt_summary.txt").write_text(str(results.summary()))
    meta = {
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "response": "Resistance",
        "predictors": predictors,
        "replaced_predictor": {
            "old": "FS_EVT_resistance_proxy_z",
            "new": "FS_EVT_resistance_proxy",
        },
        "n_rows_used": int(len(work)),
        "bw_min": int(args.bw_min),
        "selected_bw": float(bw),
        "n_jobs": 1,
        "script_variant": "local_single_job",
    }
    (out_dir / "gwr_raw_proxy_evt_metadata.json").write_text(json.dumps(meta, indent=2))

    coef_df = work[["pixel_id", "row", "col", "x", "y", "Resistance"]].copy()
    if hasattr(results, "localR2") and results.localR2 is not None:
        coef_df["localR2"] = results.localR2.flatten()

    params = results.params
    coef_df["intercept"] = params[:, 0]
    for i, col in enumerate(predictors):
        coef_df[f"coef_{col}"] = params[:, i + 1]

    if hasattr(results, "tvalues") and results.tvalues is not None:
        tv = results.tvalues
        coef_df["intercept_t"] = tv[:, 0]
        for i, col in enumerate(predictors):
            coef_df[f"t_{col}"] = tv[:, i + 1]

    coef_df.to_parquet(out_dir / "gwr_raw_proxy_evt_coefficients.parquet", index=False)


if __name__ == "__main__":
    main()
