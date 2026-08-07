from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire/MGWR_ready_table_t80_iri_stability_near_t0_postfire.parquet"
)
DEFAULT_PREDICTORS = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire/predictors_ols_recovery_baseline.txt"
)


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def compute_standardized_coefficients(work: pd.DataFrame, response: str, predictors: list[str], model) -> pd.DataFrame:
    y_std = work[response].std(ddof=0)
    rows = []
    for predictor in predictors:
        x_std = work[predictor].std(ddof=0)
        std_coef = np.nan if x_std == 0 or y_std == 0 else model.params[predictor] * x_std / y_std
        rows.append(
            {
                "term": predictor,
                "coef": float(model.params[predictor]),
                "pvalue": float(model.pvalues[predictor]),
                "abs_t": float(abs(model.tvalues[predictor])),
                "std_coef": float(std_coef),
                "abs_std_coef": float(abs(std_coef)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_std_coef", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--predictors-file", default=str(DEFAULT_PREDICTORS))
    parser.add_argument("--response", required=True, choices=["T80_revised", "IRI_good_10yr", "STAB_10yr"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    predictors = read_predictors(Path(args.predictors_file))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    work = df[[args.response] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = sm.add_constant(work[predictors], has_constant="add")
    y = work[args.response]
    model = sm.OLS(y, X).fit()

    prefix = f"ols_{args.response}"
    (out_dir / f"{prefix}_summary.txt").write_text(model.summary().as_text())
    pd.DataFrame({"term": model.params.index, "coef": model.params.values, "pvalue": model.pvalues.values}).to_csv(
        out_dir / f"{prefix}_coefficients.csv", index=False
    )
    compute_standardized_coefficients(work, args.response, predictors, model).to_csv(
        out_dir / f"{prefix}_standardized_importance.csv", index=False
    )
    metrics = {
        "input_file": str(input_path),
        "predictors_file": str(args.predictors_file),
        "n_rows_used": int(len(work)),
        "response": args.response,
        "predictors": predictors,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }
    (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
