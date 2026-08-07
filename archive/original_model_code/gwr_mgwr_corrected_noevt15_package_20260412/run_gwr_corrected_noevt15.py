#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.metrics import mean_squared_error, r2_score


PACKAGE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/gwr_mgwr_corrected_noevt15_package_20260412"
)
DEFAULT_INPUT = PACKAGE_DIR / "GWR_MGWR_ready_table_corrected_noevt15.parquet"
DEFAULT_PREDICTORS = PACKAGE_DIR / "predictors_noevt15_inferred_from_reports.txt"


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--predictors-file", default=str(DEFAULT_PREDICTORS))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-n", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bw-min", type=int, default=100)
    parser.add_argument("--moran-k", type=int, default=8)
    return parser.parse_args()


def compute_moran(coords: np.ndarray, residuals: np.ndarray, k: int) -> dict:
    weights = KNN.from_array(coords, k=k)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return {
        "k": int(k),
        "n_obs": int(len(coords)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    predictors_file = Path(args.predictors_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictors = read_predictors(predictors_file)
    df = pd.read_parquet(input_path)
    cols = ["pixel_id", "row", "col", "x", "y", "t0_year", "Resistance"] + predictors
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if args.sample_n and len(work) > args.sample_n:
        work = work.sample(n=args.sample_n, random_state=args.seed).reset_index(drop=True)

    coords = work[["x", "y"]].to_numpy(dtype=float)
    y = work[["Resistance"]].to_numpy(dtype=float)
    X = work[predictors].to_numpy(dtype=float)

    selector = Sel_BW(coords, y, X, fixed=False, kernel="bisquare", n_jobs=1)
    bw = selector.search(bw_min=args.bw_min)
    model = GWR(coords, y, X, bw=bw, fixed=False, kernel="bisquare", n_jobs=1)
    results = model.fit()

    fitted = results.predy.flatten()
    residuals = results.resid_response.flatten()
    moran = compute_moran(coords, residuals, args.moran_k)

    metrics = {
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "rows_used": int(len(work)),
        "sample_n_requested": int(args.sample_n),
        "predictor_count": int(len(predictors)),
        "predictors": predictors,
        "bandwidth": float(np.atleast_1d(bw)[0]),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "r2": float(results.R2),
        "adj_r2": float(results.adj_R2),
        "rmse": float(np.sqrt(mean_squared_error(y.flatten(), fitted))),
        "moran": moran,
        "historical_exact_15var_noevt_list": "uncertain",
    }
    (out_dir / "gwr_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "gwr_summary.txt").write_text(str(results.summary()), encoding="utf-8")

    coef_df = work[["pixel_id", "row", "col", "x", "y", "t0_year", "Resistance"]].copy()
    coef_df["fitted"] = fitted
    coef_df["residual"] = residuals
    if hasattr(results, "localR2"):
        coef_df["localR2"] = results.localR2.flatten()
    params = results.params
    coef_df["intercept"] = params[:, 0]
    for i, col in enumerate(predictors):
        coef_df[f"coef_{col}"] = params[:, i + 1]
    coef_df.to_parquet(out_dir / "gwr_coefficients.parquet", index=False)
    coef_df.to_csv(out_dir / "gwr_coefficients.csv", index=False)

    print(json.dumps({"gwr_metrics": str(out_dir / "gwr_metrics.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
