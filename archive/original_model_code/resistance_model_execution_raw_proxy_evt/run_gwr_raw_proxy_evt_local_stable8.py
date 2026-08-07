"""
Very conservative local-safe GWR script for the raw proxy EVT Resistance pathway.

Why this file exists:
- preserves all existing scripts unchanged
- uses a smaller multivariable system model after singular matrix failure
- still keeps topo, forest, human, and climate components together
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import GWR


DEFAULT_INPUT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/MGWR_ready_table.parquet"
)

PREDICTORS = [
    "TS_elev_m_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_1km_z",
    "FS_EVT_resistance_proxy",
    "HUM_popdens_win10km_log_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_hot_days_35C_pre_z",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bw", type=int, default=300)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    cols = ["pixel_id", "row", "col", "x", "y", "Resistance"] + PREDICTORS
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    coords = work[["x", "y"]].to_numpy(dtype=float)
    y = work[["Resistance"]].to_numpy(dtype=float)
    X = work[PREDICTORS].to_numpy(dtype=float)

    model = GWR(coords, y, X, bw=args.bw, fixed=False, kernel="bisquare", n_jobs=1)
    results = model.fit()

    (out_dir / "gwr_raw_proxy_evt_summary.txt").write_text(str(results.summary()))
    meta = {
        "input_file": str(input_path),
        "response": "Resistance",
        "predictors": PREDICTORS,
        "n_rows_used": int(len(work)),
        "specified_bw": int(args.bw),
        "bw_type": "adaptive_nearest_neighbor",
        "n_jobs": 1,
        "script_variant": "local_single_job_stable8",
    }
    (out_dir / "gwr_raw_proxy_evt_metadata.json").write_text(json.dumps(meta, indent=2))

    coef_df = work[["pixel_id", "row", "col", "x", "y", "Resistance"]].copy()
    if hasattr(results, "predy") and results.predy is not None:
        coef_df["fitted"] = results.predy.flatten()
    if hasattr(results, "resid_response") and results.resid_response is not None:
        coef_df["residual"] = results.resid_response.flatten()
    if hasattr(results, "localR2") and results.localR2 is not None:
        coef_df["localR2"] = results.localR2.flatten()

    params = results.params
    coef_df["intercept"] = params[:, 0]
    for i, col in enumerate(PREDICTORS):
        coef_df[f"coef_{col}"] = params[:, i + 1]

    if hasattr(results, "tvalues") and results.tvalues is not None:
        tv = results.tvalues
        coef_df["intercept_t"] = tv[:, 0]
        for i, col in enumerate(PREDICTORS):
            coef_df[f"t_{col}"] = tv[:, i + 1]

    coef_df.to_parquet(out_dir / "gwr_raw_proxy_evt_coefficients.parquet", index=False)


if __name__ == "__main__":
    main()
