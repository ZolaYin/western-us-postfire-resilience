from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from mgwr.gwr import GWR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/recovery_model_execution_near_t0_postfire"
)
RES_BASE = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/resistance_model_execution_near_t0_aggregated"
)

RECOVERY_TABLE = BASE / "MGWR_ready_table_increment_recovery_near_t0_postfire.parquet"
RESISTANCE_TABLE = RES_BASE / "MGWR_ready_table_near_t0_aggregated.parquet"
OUT = BASE / "gwr_residual_correction_all_y_screen_2026-03-31"

RANDOM_STATE = 42
MORAN_K = 8
WORK_SAMPLE_N = 12000
GWR_FIT_N = 300
GWR_MORAN_N = 3000
ADAPTIVE_BW = 60

RECOVERY_RF_PREDICTORS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z", "TS_twi_z", "TS_roughness_z",
    "TS_SOC_0_30cm_z", "FS_TCC_t0_z", "FS_CBH_t0agg_z", "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z", "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z", "CLIM_pr_sum_post_z", "CLIM_eto_sum_post_z", "CLIM_tmmn_mean_post_z",
    "CLIM_hot_days_35C_post_z", "CLIM_aridity_post_z", "CLIM_tmmx_std_post_z", "x", "y", "x_sq_z", "y_sq_z", "xy_z",
]
RECOVERY_GWR_PREDICTORS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_SOC_0_30cm_z", "FS_TCC_t0_z", "FS_CBH_t0agg_z",
    "FS_EVT_t0agg_resistance_proxy", "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_post_z", "CLIM_hot_days_35C_post_z", "CLIM_tmmx_std_post_z",
]
RESISTANCE_RF_PREDICTORS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z", "TS_twi_z", "TS_roughness_z",
    "TS_SOC_0_30cm_z", "FS_TCC_t0_z", "FS_CBH_t0agg_z", "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z", "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z", "CLIM_pr_sum_pre_z", "CLIM_eto_sum_pre_z", "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z", "CLIM_aridity_pre_z", "CLIM_tmmx_std_pre_z", "x", "y",
]
RESISTANCE_GWR_PREDICTORS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_SOC_0_30cm_z", "FS_TCC_t0_z", "FS_CBH_t0agg_z",
    "FS_EVT_t0agg_resistance_proxy", "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z", "CLIM_hot_days_35C_pre_z", "CLIM_tmmx_std_pre_z",
]


def add_spatial_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    x_sq = out["x"] ** 2
    y_sq = out["y"] ** 2
    xy = out["x"] * out["y"]
    out["x_sq_z"] = (x_sq - x_sq.mean()) / x_sq.std(ddof=0)
    out["y_sq_z"] = (y_sq - y_sq.mean()) / y_sq.std(ddof=0)
    out["xy_z"] = (xy - xy.mean()) / xy.std(ddof=0)
    return out


def sample_df(df: pd.DataFrame, n: int, seed: int, stratify_col: str | None = None) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    if stratify_col is None:
        return df.sample(n=n, random_state=seed).copy()
    frac = n / len(df)
    parts = []
    for _, grp in df.groupby(stratify_col):
        take = max(1, int(round(len(grp) * frac)))
        parts.append(grp.sample(n=min(take, len(grp)), random_state=seed))
    out = pd.concat(parts, axis=0)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.copy()


def compute_moran(coords: np.ndarray, residuals: np.ndarray) -> float:
    w = KNN.from_array(coords, k=MORAN_K)
    w.transform = "R"
    return float(Moran(residuals.astype(float), w, permutations=0).I)


def probs_to_expected(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return probs @ classes.astype(float)


def fit_gwr_correction(train: pd.DataFrame, gwr_predictors: list[str], resid: np.ndarray, pred_df: pd.DataFrame) -> np.ndarray:
    fit = sample_df(train.assign(_resid=resid), GWR_FIT_N, RANDOM_STATE)
    gwr = GWR(
        fit[["x", "y"]].to_numpy(),
        fit["_resid"].to_numpy().reshape(-1, 1),
        fit[gwr_predictors].to_numpy(),
        bw=ADAPTIVE_BW,
        fixed=False,
        kernel="bisquare",
        n_jobs=1,
    )
    fit_res = gwr.fit()
    chunks = []
    for start in range(0, len(pred_df), GWR_FIT_N):
        chunk = pred_df.iloc[start : start + GWR_FIT_N]
        pred_res = gwr.predict(
            chunk[["x", "y"]].to_numpy(),
            chunk[gwr_predictors].to_numpy(),
            exog_scale=fit_res.scale,
            exog_resid=fit_res.resid_response,
        )
        chunks.append(pred_res.predy.flatten())
    return np.concatenate(chunks)


def run_regression(df: pd.DataFrame, response: str, rf_predictors: list[str], gwr_predictors: list[str]) -> dict:
    print("Running", response, flush=True)
    cols = [response] + list(dict.fromkeys(rf_predictors + gwr_predictors + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = sample_df(work, WORK_SAMPLE_N, RANDOM_STATE)
    train, test = train_test_split(work, test_size=0.2, random_state=RANDOM_STATE)

    rf = RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train[rf_predictors], train[response])
    rf_test = rf.predict(test[rf_predictors])
    train_resid = train[response].to_numpy() - rf.predict(train[rf_predictors])
    corr_test = fit_gwr_correction(train, gwr_predictors, train_resid, test)
    corrected_test = rf_test + corr_test

    moran_df = sample_df(work, min(GWR_MORAN_N, len(work)), RANDOM_STATE)
    rf_moran_pred = rf.predict(moran_df[rf_predictors])
    corr_moran = fit_gwr_correction(train, gwr_predictors, train_resid, moran_df)
    corrected_moran_pred = rf_moran_pred + corr_moran

    row = {
        "y": response,
        "family_or_scheme": "rf_plus_gwr_residual_screen",
        "n_rows_used": int(len(work)),
        "rf_test_r2": float(r2_score(test[response], rf_test)),
        "corrected_test_r2": float(r2_score(test[response], corrected_test)),
        "rf_test_rmse": float(np.sqrt(mean_squared_error(test[response], rf_test))),
        "corrected_test_rmse": float(np.sqrt(mean_squared_error(test[response], corrected_test))),
        "rf_moran_i": compute_moran(moran_df[["x", "y"]].to_numpy(), moran_df[response].to_numpy() - rf_moran_pred),
        "corrected_moran_i": compute_moran(moran_df[["x", "y"]].to_numpy(), moran_df[response].to_numpy() - corrected_moran_pred),
        "gwr_bw": ADAPTIVE_BW,
    }
    pd.DataFrame([row]).to_csv(OUT / f"{response}_row.csv", index=False)
    print("Finished", response, flush=True)
    return row


def run_t80(df: pd.DataFrame) -> dict:
    response = "T80_revised"
    print("Running", response, flush=True)
    cols = [response] + list(dict.fromkeys(RECOVERY_RF_PREDICTORS + RECOVERY_GWR_PREDICTORS + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = sample_df(work, WORK_SAMPLE_N, RANDOM_STATE, stratify_col=response)
    train, test = train_test_split(work, test_size=0.2, random_state=RANDOM_STATE, stratify=work[response])

    rf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train[RECOVERY_RF_PREDICTORS], train[response])
    rf_test_prob = rf.predict_proba(test[RECOVERY_RF_PREDICTORS])
    rf_test = probs_to_expected(rf_test_prob, rf.classes_)
    train_resid = train[response].to_numpy() - probs_to_expected(rf.predict_proba(train[RECOVERY_RF_PREDICTORS]), rf.classes_)
    corr_test = fit_gwr_correction(train, RECOVERY_GWR_PREDICTORS, train_resid, test)
    corrected_test = rf_test + corr_test

    moran_df = sample_df(work, min(GWR_MORAN_N, len(work)), RANDOM_STATE, stratify_col=response)
    rf_moran_pred = probs_to_expected(rf.predict_proba(moran_df[RECOVERY_RF_PREDICTORS]), rf.classes_)
    corr_moran = fit_gwr_correction(train, RECOVERY_GWR_PREDICTORS, train_resid, moran_df)
    corrected_moran_pred = rf_moran_pred + corr_moran

    row = {
        "y": response,
        "family_or_scheme": "ordinal_rf_plus_gwr_residual_screen",
        "n_rows_used": int(len(work)),
        "rf_test_r2": float(r2_score(test[response], rf_test)),
        "corrected_test_r2": float(r2_score(test[response], corrected_test)),
        "rf_test_rmse": float(np.sqrt(mean_squared_error(test[response], rf_test))),
        "corrected_test_rmse": float(np.sqrt(mean_squared_error(test[response], corrected_test))),
        "rf_moran_i": compute_moran(moran_df[["x", "y"]].to_numpy(), moran_df[response].to_numpy() - rf_moran_pred),
        "corrected_moran_i": compute_moran(moran_df[["x", "y"]].to_numpy(), moran_df[response].to_numpy() - corrected_moran_pred),
        "gwr_bw": ADAPTIVE_BW,
        "rf_accuracy": float(accuracy_score(test[response], rf.predict(test[RECOVERY_RF_PREDICTORS]))),
        "rf_macro_f1": float(f1_score(test[response], rf.predict(test[RECOVERY_RF_PREDICTORS]), average="macro")),
    }
    pd.DataFrame([row]).to_csv(OUT / f"{response}_row.csv", index=False)
    print("Finished", response, flush=True)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recovery_df = add_spatial_terms(pd.read_parquet(RECOVERY_TABLE))
    resistance_df = add_spatial_terms(pd.read_parquet(RESISTANCE_TABLE))
    rows = [
        run_regression(resistance_df, "Resistance", RESISTANCE_RF_PREDICTORS, RESISTANCE_GWR_PREDICTORS),
        run_t80(recovery_df),
        run_regression(recovery_df, "IRI_good_10yr", RECOVERY_RF_PREDICTORS, RECOVERY_GWR_PREDICTORS),
        run_regression(recovery_df, "STAB_10yr", RECOVERY_RF_PREDICTORS, RECOVERY_GWR_PREDICTORS),
        run_regression(recovery_df, "INC_end_rel_10obs", RECOVERY_RF_PREDICTORS, RECOVERY_GWR_PREDICTORS),
        run_regression(recovery_df, "INC_cum_rel_10obs", RECOVERY_RF_PREDICTORS, RECOVERY_GWR_PREDICTORS),
    ]
    summary = pd.DataFrame(rows)
    summary["delta_test_r2"] = summary["corrected_test_r2"] - summary["rf_test_r2"]
    summary["delta_test_rmse"] = summary["corrected_test_rmse"] - summary["rf_test_rmse"]
    summary["delta_moran_i"] = summary["corrected_moran_i"] - summary["rf_moran_i"]
    summary.to_csv(OUT / "gwr_residual_correction_all_y_screen_summary.csv", index=False)

    lines = ["GWR residual correction screening across all y", ""]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['y']}: rf_r2={row['rf_test_r2']:.4f}, corrected_r2={row['corrected_test_r2']:.4f}, "
            f"rf_moran={row['rf_moran_i']:.4f}, corrected_moran={row['corrected_moran_i']:.4f}"
        )
    (OUT / "gwr_residual_correction_all_y_screen_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
