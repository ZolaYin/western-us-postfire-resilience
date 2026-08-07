from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.model_selection import train_test_split
from spreg import ML_Error, ML_Lag, OLS


RECOVERY_BASE = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/recovery_model_execution_near_t0_postfire"
)
RESISTANCE_BASE = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/resistance_model_execution_near_t0_aggregated"
)
RECOVERY_TABLE = RECOVERY_BASE / "MGWR_ready_table_increment_recovery_near_t0_postfire.parquet"
RESISTANCE_TABLE = RESISTANCE_BASE / "MGWR_ready_table_near_t0_aggregated.parquet"
OUT_DIR = RECOVERY_BASE / "sar_sem_all_y_screen_2026-03-31"

RANDOM_STATE = 42
MORAN_K = 8
WEIGHT_K = 8
WORK_SAMPLE_N = 4000

RECOVERY_PREDICTORS = [
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
    "CLIM_pr_sum_post_z",
    "CLIM_hot_days_35C_post_z",
    "CLIM_tmmx_std_post_z",
]
RESISTANCE_PREDICTORS = [
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


def sample_df(df: pd.DataFrame, n: int, stratify_col: str | None = None) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    if stratify_col is None:
        return df.sample(n=n, random_state=RANDOM_STATE).copy()
    frac = n / len(df)
    parts = []
    for _, grp in df.groupby(stratify_col):
        take = max(1, int(round(len(grp) * frac)))
        parts.append(grp.sample(n=min(take, len(grp)), random_state=RANDOM_STATE))
    out = pd.concat(parts, axis=0)
    if len(out) > n:
        out = out.sample(n=n, random_state=RANDOM_STATE)
    return out.copy()


def compute_moran(coords: np.ndarray, residuals: np.ndarray) -> float:
    w = KNN.from_array(coords, k=MORAN_K)
    w.transform = "R"
    return float(Moran(residuals.astype(float), w, permutations=0).I)


def run_models(df: pd.DataFrame, response: str, predictors: list[str], stratify: bool = False) -> pd.DataFrame:
    cols = [response] + predictors + ["x", "y"]
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = sample_df(work, WORK_SAMPLE_N, stratify_col=response if stratify else None)
    coords = work[["x", "y"]].to_numpy()
    w = KNN.from_array(coords, k=WEIGHT_K)
    w.transform = "R"
    y = work[[response]].to_numpy()
    x = work[predictors].to_numpy()

    print("Running", response, flush=True)
    ols = OLS(y, x, name_y=response, name_x=predictors)
    lag = ML_Lag(y, x, w, method="ord", name_y=response, name_x=predictors)
    err = ML_Error(y, x, w, method="full", name_y=response, name_x=predictors)

    rows = [
        {
            "y": response,
            "model": "OLS",
            "n_rows_used": int(len(work)),
            "pr2": float(ols.r2),
            "aic": float(ols.aic),
            "bic": float(ols.schwarz),
            "spatial_param": np.nan,
            "residual_moran_i": compute_moran(coords, ols.u.flatten()),
        },
        {
            "y": response,
            "model": "SAR",
            "n_rows_used": int(len(work)),
            "pr2": float(lag.pr2),
            "aic": float(lag.aic),
            "bic": float(lag.schwarz),
            "spatial_param": float(lag.rho),
            "residual_moran_i": compute_moran(coords, lag.u.flatten()),
        },
        {
            "y": response,
            "model": "SEM",
            "n_rows_used": int(len(work)),
            "pr2": float(err.pr2),
            "aic": float(err.aic),
            "bic": float(err.schwarz),
            "spatial_param": float(err.lam),
            "residual_moran_i": compute_moran(coords, err.u.flatten()),
        },
    ]

    run_dir = OUT_DIR / response
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / f"{response}_sar_sem_models.csv", index=False)

    coef_rows = []
    for name, model, extra_name in [("OLS", ols, None), ("SAR", lag, "rho"), ("SEM", err, "lambda")]:
        coef_names = ["CONSTANT"] + predictors
        for i, coef_name in enumerate(coef_names):
            coef_rows.append(
                {
                    "y": response,
                    "model": name,
                    "term": coef_name,
                    "coef": float(model.betas[i][0]),
                }
            )
        if extra_name is not None:
            coef_rows.append(
                {
                    "y": response,
                    "model": name,
                    "term": extra_name,
                    "coef": float(model.betas[-1][0]),
                }
            )
    pd.DataFrame(coef_rows).to_csv(run_dir / f"{response}_sar_sem_coefficients.csv", index=False)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recovery_df = pd.read_parquet(RECOVERY_TABLE)
    resistance_df = pd.read_parquet(RESISTANCE_TABLE)

    frames = [
        run_models(resistance_df, "Resistance", RESISTANCE_PREDICTORS),
        run_models(recovery_df, "T80_revised", RECOVERY_PREDICTORS, stratify=True),
        run_models(recovery_df, "IRI_good_10yr", RECOVERY_PREDICTORS),
        run_models(recovery_df, "STAB_10yr", RECOVERY_PREDICTORS),
        run_models(recovery_df, "INC_end_rel_10obs", RECOVERY_PREDICTORS),
        run_models(recovery_df, "INC_cum_rel_10obs", RECOVERY_PREDICTORS),
    ]
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(OUT_DIR / "sar_sem_all_y_screen_summary.csv", index=False)
    (OUT_DIR / "sar_sem_all_y_screen_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = ["SAR / SEM screening across all y", ""]
    for y, sub in summary.groupby("y"):
        lines.append(y)
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['model']}: pr2={row['pr2']:.4f}, aic={row['aic']:.2f}, "
                f"bic={row['bic']:.2f}, spatial={row['spatial_param']}, moran={row['residual_moran_i']:.4f}"
            )
    (OUT_DIR / "sar_sem_all_y_screen_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
