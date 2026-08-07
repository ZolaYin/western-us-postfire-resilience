#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--predictors-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    predictors_path = Path(args.predictors_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictors = read_predictors(predictors_path)
    df = pd.read_parquet(input_path)
    required = [args.response, "x", "y", *predictors]
    work = df[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if args.sample_n is not None and args.sample_n < len(work):
        work = work.sample(n=args.sample_n, random_state=args.random_state).reset_index(drop=True)

    y = work[[args.response]].to_numpy(dtype=float)
    X = work[predictors].to_numpy(dtype=float)
    coords = work[["x", "y"]].to_numpy(dtype=float)

    selector = Sel_BW(coords, y, X, multi=True)
    bandwidths = selector.search()

    model = MGWR(coords, y, X, selector=selector)
    results = model.fit()

    fitted = results.predy.ravel()
    residual = results.resid_response.ravel()
    observed = y.ravel()
    rmse = float(np.sqrt(np.mean((observed - fitted) ** 2)))

    coef_columns = ["Intercept", *predictors]
    coef_df = pd.DataFrame(results.params, columns=coef_columns)
    coef_out = pd.concat([work[["x", "y"]].reset_index(drop=True), coef_df], axis=1)
    coef_out.to_parquet(output_dir / "mgwr_coefficients.parquet", index=False)

    resid_out = work[["x", "y"]].copy()
    resid_out["observed"] = observed
    resid_out["predicted"] = fitted
    resid_out["residual"] = residual
    resid_out.to_parquet(output_dir / "mgwr_residuals.parquet", index=False)

    pd.DataFrame({"term": coef_columns, "bandwidth": np.asarray(bandwidths).ravel()}).to_csv(
        output_dir / "mgwr_bandwidths.csv", index=False
    )

    metrics = {
        "response": args.response,
        "input_path": Path(args.input).as_posix(),
        "predictors_file": Path(args.predictors_file).as_posix(),
        "rows_used": int(len(work)),
        "sample_n": None if args.sample_n is None else int(args.sample_n),
        "random_state": int(args.random_state),
        "predictor_count": int(len(predictors)),
        "predictors": predictors,
        "bandwidths": np.asarray(bandwidths).ravel().astype(float).tolist(),
        "aic": float(results.aic),
        "bic": float(results.bic),
        "r2": float(results.R2),
        "adj_r2": float(results.adj_R2),
        "rmse": rmse,
    }
    (output_dir / "mgwr_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "mgwr_summary.txt").write_text(str(results.summary()), encoding="utf-8")

    print(json.dumps({"mgwr_output_dir": str(output_dir), "response": args.response}, ensure_ascii=False))


if __name__ == "__main__":
    main()
