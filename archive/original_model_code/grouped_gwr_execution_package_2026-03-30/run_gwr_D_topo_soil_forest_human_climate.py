import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW


DEFAULT_INPUT = Path("/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/resistance_model_execution_near_t0_aggregated/MGWR_ready_table_near_t0_aggregated.parquet")
DEFAULT_PREDICTORS = Path("/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/grouped_gwr_execution_package_2026-03-30/predictors_D_topo_soil_forest_human_climate.txt")


def read_predictors(path: Path):
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
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
    cols = ["pixel_id", "row", "col", "x", "y", "t0_year", "Resistance"] + predictors
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    coords = work[["x", "y"]].to_numpy(dtype=float)
    y = work[["Resistance"]].to_numpy(dtype=float)
    X = work[predictors].to_numpy(dtype=float)

    selector = Sel_BW(coords, y, X, fixed=False, kernel="bisquare")
    bw = selector.search(bw_min=args.bw_min)
    model = GWR(coords, y, X, bw=bw, fixed=False, kernel="bisquare")
    results = model.fit()

    meta = {
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "n_rows_used": int(len(work)),
        "predictors": predictors,
        "bandwidth": float(np.atleast_1d(bw)[0]),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "r2": float(results.R2),
        "adj_r2": float(results.adj_R2),
    }
    (out_dir / "gwr_metrics.json").write_text(json.dumps(meta, indent=2))

    coef_df = work[["pixel_id", "row", "col", "x", "y", "t0_year", "Resistance"]].copy()
    coef_df["fitted"] = results.predy.flatten()
    coef_df["residual"] = results.resid_response.flatten()
    if hasattr(results, "localR2"):
        coef_df["localR2"] = results.localR2.flatten()

    params = results.params
    if params.shape[1] == len(predictors) + 1:
        coef_df["intercept"] = params[:, 0]
        for i, col in enumerate(predictors):
            coef_df[f"coef_{col}"] = params[:, i + 1]
    else:
        for i, col in enumerate(predictors):
            coef_df[f"coef_{col}"] = params[:, i]

    coef_df.to_parquet(out_dir / "gwr_coefficients.parquet", index=False)
    coef_df.to_csv(out_dir / "gwr_coefficients.csv", index=False)


if __name__ == "__main__":
    main()
