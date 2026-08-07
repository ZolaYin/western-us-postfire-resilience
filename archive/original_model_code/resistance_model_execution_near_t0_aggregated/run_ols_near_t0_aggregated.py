from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
import statsmodels.api as sm


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_model_execution_near_t0_aggregated/MGWR_ready_table_near_t0_aggregated.parquet"
)
PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z",
    "HUM_roaddens_r5km_z",
    "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_tmmx_std_pre_z",
]


def compute_standardized_coefficients(work: pd.DataFrame, predictors: list[str], model) -> pd.DataFrame:
    y_std = work["Resistance"].std(ddof=0)
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


def compute_residual_moran(work: pd.DataFrame, model, k: int) -> dict:
    weights = KNN.from_array(work[["x", "y"]].to_numpy(), k=k)
    weights.transform = "R"
    resid = model.resid.to_numpy()
    moran = Moran(resid, weights, permutations=0)
    return {
        "k": int(k),
        "n_obs": int(len(work)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra-predictors", nargs="*", default=[])
    parser.add_argument("--compute-full-fit-moran", action="store_true")
    parser.add_argument("--moran-k", type=int, default=8)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(Path(args.input))
    predictors = list(PREDICTORS)
    for col in args.extra_predictors:
        if col not in predictors:
            predictors.append(col)
    cols = ["Resistance"] + predictors
    if args.compute_full_fit_moran:
        for col in ("x", "y"):
            if col not in cols:
                cols.append(col)
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = sm.add_constant(work[predictors], has_constant="add")
    y = work["Resistance"]
    model = sm.OLS(y, X).fit()
    (out_dir / "ols_near_t0_aggregated_summary.txt").write_text(model.summary().as_text())
    pd.DataFrame({"term": model.params.index, "coef": model.params.values, "pvalue": model.pvalues.values}).to_csv(
        out_dir / "ols_near_t0_aggregated_coefficients.csv", index=False
    )
    compute_standardized_coefficients(work, predictors, model).to_csv(
        out_dir / "ols_near_t0_aggregated_standardized_importance.csv", index=False
    )
    metrics = {
        "input_file": str(args.input),
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "predictors": predictors,
        "extra_predictors": args.extra_predictors,
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }
    if args.compute_full_fit_moran:
        metrics["full_fit_residual_moran"] = compute_residual_moran(work, model, args.moran_k)
    (out_dir / "ols_near_t0_aggregated_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
