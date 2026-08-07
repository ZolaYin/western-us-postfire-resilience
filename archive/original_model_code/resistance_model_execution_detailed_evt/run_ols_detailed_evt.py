"""
OLS script for the detailed EVT Resistance pathway.

Uses:
- MGWR_ready_table.parquet as the current Resistance-only driver table
- detailed_evt_candidate_table.parquet as the detailed EVT source

Join strategy:
- inner join on pixel_id only
- no parquet files are modified in place
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/MGWR_ready_table.parquet"
)
DETAIL_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "resistance_modeling_detailed_evt/detailed_evt_candidate_table.parquet"
)

BASE_PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_1km_z",
    "HUM_popdens_win10km_log_z",
    "HUM_roaddens_r5km_z",
    "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_tmmx_std_pre_z",
]

DETAILED_EVT_PREDICTORS = [
    "EVT2022_group_is_shrub",
    "EVT2022_group_is_deciduous",
    "EVT2022_group_is_mixed",
    "EVT2022_group_is_conifer",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-input", default=str(BASE_INPUT))
    parser.add_argument("--detail-input", default=str(DETAIL_INPUT))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_input = Path(args.base_input)
    detail_input = Path(args.detail_input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cols = ["pixel_id", "Resistance"] + BASE_PREDICTORS
    detail_cols = ["pixel_id"] + DETAILED_EVT_PREDICTORS

    base_df = pd.read_parquet(base_input, columns=base_cols)
    detail_df = pd.read_parquet(detail_input, columns=detail_cols)
    work = base_df.merge(detail_df, on="pixel_id", how="inner", validate="one_to_one")
    predictors = BASE_PREDICTORS + DETAILED_EVT_PREDICTORS
    work = work[["Resistance"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = sm.add_constant(work[predictors], has_constant="add")
    y = work["Resistance"]
    model = sm.OLS(y, X).fit()

    (out_dir / "ols_detailed_evt_summary.txt").write_text(model.summary().as_text())
    coef_df = pd.DataFrame(
        {"term": model.params.index, "coef": model.params.values, "pvalue": model.pvalues.values}
    )
    coef_df.to_csv(out_dir / "ols_detailed_evt_coefficients.csv", index=False)
    metrics = {
        "base_input_file": str(base_input),
        "detail_input_file": str(detail_input),
        "join_key": "pixel_id",
        "n_rows_used": int(len(work)),
        "response": "Resistance",
        "base_predictors": BASE_PREDICTORS,
        "detailed_evt_predictors": DETAILED_EVT_PREDICTORS,
        "predictors": predictors,
        "detailed_evt_strategy": "grouped_indicators_reference_nonwoody",
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }
    (out_dir / "ols_detailed_evt_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
