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
    "resistance_model_execution_near_t0_evt_cbh/MGWR_ready_table_near_t0_evt_cbh.parquet"
)
PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0_z",
    "FS_EVT_t0_resistance_proxy",
    "HUM_popdens_win10km_log_z",
    "HUM_roaddens_r5km_z",
    "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_tmmx_std_pre_z",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    work = df[["Resistance"] + PREDICTORS].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = sm.add_constant(work[PREDICTORS], has_constant="add")
    y = work["Resistance"]
    model = sm.OLS(y, X).fit()

    (out_dir / "ols_near_t0_evt_cbh_summary.txt").write_text(model.summary().as_text())
    pd.DataFrame(
        {"term": model.params.index, "coef": model.params.values, "pvalue": model.pvalues.values}
    ).to_csv(out_dir / "ols_near_t0_evt_cbh_coefficients.csv", index=False)
    metrics = {
        "input_file": str(input_path),
        "response": "Resistance",
        "n_rows_used": int(len(work)),
        "predictors": PREDICTORS,
        "replaced_predictors": {
            "evt": {"old": "FS_EVT_resistance_proxy", "new": "FS_EVT_t0_resistance_proxy"},
            "cbh": {"old": "FS_CBH_1km_z", "new": "FS_CBH_t0_z"},
        },
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }
    (out_dir / "ols_near_t0_evt_cbh_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
