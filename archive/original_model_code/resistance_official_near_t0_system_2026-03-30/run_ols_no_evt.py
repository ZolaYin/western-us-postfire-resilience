import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "resistance_model_execution_near_t0_aggregated"
    / "MGWR_ready_table_near_t0_aggregated.parquet"
)
OUT = ROOT / "resistance_official_near_t0_system_2026-03-30" / "no_evt"

ID_COLS = ["pixel_id", "row", "col", "x", "y", "t0_year"]
RESPONSE = "Resistance"

PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z",
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_eto_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_tmmx_std_pre_z",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT)
    model_df = df[ID_COLS + [RESPONSE] + PREDICTORS].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = sm.add_constant(model_df[PREDICTORS], has_constant="add")
    y = model_df[RESPONSE]
    fit = sm.OLS(y, X).fit()

    metrics = {
        "model_type": "OLS",
        "evt_variant": "no_evt",
        "rows_used": int(len(model_df)),
        "predictor_count": int(len(PREDICTORS)),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "rmse": float(np.sqrt(np.mean((y - fit.predict(X)) ** 2))),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }

    pd.DataFrame(
        {"term": fit.params.index, "coef": fit.params.values, "p_value": fit.pvalues.values}
    ).to_csv(OUT / "ols_no_evt_coefficients.csv", index=False)
    (OUT / "ols_no_evt_summary.txt").write_text(fit.summary().as_text(), encoding="utf-8")
    (OUT / "ols_no_evt_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
