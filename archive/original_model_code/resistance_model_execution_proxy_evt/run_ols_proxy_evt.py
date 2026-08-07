"""
OLS-ready script for the proxy EVT Resistance pathway.

Intended use:
- local machine or Google Colab
- not an HPRC submission script

Defaults:
- input file: MGWR_model_input.parquet
- response: Resistance
- predictor file: predictors_ols_proxy_evt.txt from resistance_modeling_plan

Outputs:
- ols_proxy_evt_summary.txt
- ols_proxy_evt_coefficients.csv
- ols_proxy_evt_metrics.json

Dependencies:
- pandas
- numpy
- statsmodels
- pyarrow or fastparquet for parquet reads
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/MGWR_model_input.parquet"
)
DEFAULT_PREDICTORS = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_modeling_plan/predictors_ols_proxy_evt.txt"
)


def read_predictors(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--predictors-file", default=str(DEFAULT_PREDICTORS))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    predictors_file = Path(args.predictors_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    predictors = read_predictors(predictors_file)
    cols = ["Resistance"] + predictors
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = sm.add_constant(work[predictors], has_constant="add")
    y = work["Resistance"]
    model = sm.OLS(y, X).fit()

    (out_dir / "ols_proxy_evt_summary.txt").write_text(model.summary().as_text())
    coef_df = pd.DataFrame({
        "term": model.params.index,
        "coef": model.params.values,
        "pvalue": model.pvalues.values,
    })
    coef_df.to_csv(out_dir / "ols_proxy_evt_coefficients.csv", index=False)
    metrics = {
        "input_file": str(input_path),
        "predictors_file": str(predictors_file),
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "predictors": predictors,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }
    (out_dir / "ols_proxy_evt_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
