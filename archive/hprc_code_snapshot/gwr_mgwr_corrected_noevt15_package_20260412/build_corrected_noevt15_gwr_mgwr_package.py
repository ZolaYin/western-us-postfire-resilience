#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
OUT_DIR = ROOT / "gwr_mgwr_corrected_noevt15_package_20260412"
OUT_TABLE = OUT_DIR / "GWR_MGWR_ready_table_corrected_noevt15.parquet"
OUT_SAMPLE = OUT_DIR / "GWR_MGWR_ready_table_corrected_noevt15_sample.csv"
OUT_PREDICTORS = OUT_DIR / "predictors_noevt15_inferred_from_reports.txt"
OUT_REPORT = OUT_DIR / "package_build_report.md"
OUT_AUDIT = OUT_DIR / "package_build_audit.json"

# This set is the most defensible noEVT launch bundle I could reconstruct from
# verified project artifacts. The exact historical full 15-variable HPRC list
# is uncertain in surviving scripts, so the package report labels that clearly.
PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_twi_z",
    "TS_SOC_0_30cm_clean_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z",
    "HUM_roaddens_r5km_z",
    "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_tmmx_std_pre_z",
]


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def build_table() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(INPUT).copy()

    df["TS_SOC_0_30cm_clean"] = pd.to_numeric(df["TS_SOC_0_30cm"], errors="coerce").astype(float)
    soc_bad = int((df["TS_SOC_0_30cm_clean"] == -9999).sum())
    df.loc[df["TS_SOC_0_30cm_clean"] == -9999, "TS_SOC_0_30cm_clean"] = np.nan

    df["TS_elev_m_z"] = zscore(df["TS_elev_m"])
    df["TS_slope_deg_z"] = zscore(df["TS_slope_deg"])
    df["TS_twi_z"] = zscore(df["TS_twi"])
    df["TS_SOC_0_30cm_clean_z"] = zscore(df["TS_SOC_0_30cm_clean"])
    df["FS_TCC_t0_z"] = zscore(df["FS_TCC_t0"])
    df["FS_CBH_t0agg_z"] = zscore(df["FS_CBH_t0agg"])
    df["HUM_popdens_win10km_log_z"] = zscore(
        np.log1p(pd.to_numeric(df["HUM_popdens_win10km"], errors="coerce").clip(lower=0))
    )
    df["HUM_roaddens_r5km_z"] = zscore(df["HUM_roaddens_r5km"])
    df["HUM_traildens_r10km_z"] = zscore(df["HUM_traildens_r10km"])
    df["HUM_imperv_near_t0_z"] = zscore(df["HUM_imperv_near_t0"])
    df["HUM_viirs_near_t0_log_z"] = zscore(
        np.log1p(pd.to_numeric(df["HUM_viirs_near_t0"], errors="coerce").clip(lower=0))
    )
    df["CLIM_pr_sum_pre_z"] = zscore(df["CLIM_pr_sum_pre"])
    df["CLIM_tmmn_mean_pre_z"] = zscore(df["CLIM_tmmn_mean_pre"])
    df["CLIM_hot_days_35C_pre_z"] = zscore(df["CLIM_hot_days_35C_pre"])
    df["CLIM_tmmx_std_pre_z"] = zscore(df["CLIM_tmmx_std_pre"])

    cols = ["pixel_id", "row", "col", "x", "y", "t0_year", "Resistance"] + PREDICTORS
    work = (
        df[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
        .copy()
    )

    audit = {
        "input_table": str(INPUT),
        "rows_input": int(len(df)),
        "rows_output": int(len(work)),
        "predictor_count": int(len(PREDICTORS)),
        "predictors": PREDICTORS,
        "ts_soc_invalid_eq_minus9999": soc_bad,
        "historical_exact_15var_noevt_list": "uncertain",
        "inference_basis": [
            "Verified grouped GWR package uses A_topo_soil = elev+slope+twi+SOC.",
            "Verified formal report states a noEVT stable extension exists and all-in noEVT reaches R2=0.7023.",
            "Verified intermediate success row mentions adding TCC, CBH, popdens, tmmx_std, tmmn, and viirs.",
            "This package completes the noEVT launch set with road, trail, impervious, and pre-fire pr_sum + hot_days.",
        ],
    }
    return work, audit


def write_predictor_file() -> None:
    lines = [
        "# corrected Western US noEVT 15-variable GWR/MGWR launch bundle",
        "# exact historical full noEVT HPRC list: uncertain",
        "# this is the most defensible noEVT reconstruction from verified project artifacts",
        *PREDICTORS,
        "",
    ]
    OUT_PREDICTORS.write_text("\n".join(lines), encoding="utf-8")


def write_report(audit: dict) -> None:
    lines = [
        "# Corrected noEVT GWR/MGWR Package",
        "",
        f"- Source table: `{INPUT}`",
        f"- Output table: `{OUT_TABLE}`",
        f"- Rows kept: `{audit['rows_output']}`",
        f"- Predictor count: `{audit['predictor_count']}`",
        f"- TS_SOC invalid `-9999` handled as missing: `{audit['ts_soc_invalid_eq_minus9999']}`",
        f"- Exact historical full 15-variable noEVT list: `{audit['historical_exact_15var_noevt_list']}`",
        "",
        "Predictors:",
        *[f"- `{name}`" for name in PREDICTORS],
        "",
        "Evidence basis:",
        *[f"- {item}" for item in audit["inference_basis"]],
        "",
        "Local run entrypoints in this package:",
        "- `run_gwr_corrected_noevt15.py`",
        "- `run_mgwr_corrected_noevt15.py`",
        "- `submit_mgwr_corrected_noevt15.sbatch`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work, audit = build_table()
    work.to_parquet(OUT_TABLE, index=False)
    work.head(5000).to_csv(OUT_SAMPLE, index=False)
    write_predictor_file()
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_report(audit)
    print(
        json.dumps(
            {
                "output_table": str(OUT_TABLE),
                "predictor_file": str(OUT_PREDICTORS),
                "rows_output": int(len(work)),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
