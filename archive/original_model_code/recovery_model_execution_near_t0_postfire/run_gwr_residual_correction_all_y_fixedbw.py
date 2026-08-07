from __future__ import annotations

import json
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
OUT = BASE / "gwr_residual_correction_all_y_fixedbw_fast_2026-03-31"

RANDOM_STATE = 42
MORAN_K = 8
FIT_SAMPLE_N = 500
FULL_FIT_SAMPLE_N = 500
MORAN_SAMPLE_N = 5000
FIXED_BW = 80

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


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def sample_arrays(coords: np.ndarray, residuals: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) <= n:
        return coords, residuals
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(coords), size=n, replace=False)
    return coords[idx], residuals[idx]


def compute_moran(coords: np.ndarray, residuals: np.ndarray) -> dict:
    coords, residuals = sample_arrays(coords, residuals, MORAN_SAMPLE_N, RANDOM_STATE)
    w = KNN.from_array(coords, k=MORAN_K)
    w.transform = "R"
    moran = Moran(residuals.astype(float), w, permutations=0)
    return {"moran_i": float(moran.I), "z_norm": float(moran.z_norm), "p_norm": float(moran.p_norm)}


def probs_to_expected(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return probs @ classes.astype(float)


def gwr_predict(train_coords, train_X, train_resid, pred_coords, pred_X, bw: int) -> np.ndarray:
    fit_df = sample_df(
        pd.DataFrame(train_X).assign(resid=train_resid, xcoord=train_coords[:, 0], ycoord=train_coords[:, 1]),
        FIT_SAMPLE_N,
        RANDOM_STATE,
    )
    gwr = GWR(
        fit_df[["xcoord", "ycoord"]].to_numpy(),
        fit_df["resid"].to_numpy().reshape(-1, 1),
        fit_df.drop(columns=["resid", "xcoord", "ycoord"]).to_numpy(),
        bw=bw,
        fixed=False,
        kernel="bisquare",
        n_jobs=1,
    )
    return gwr.predict(pred_coords, pred_X).predy.flatten()


def full_corrected_pred(work: pd.DataFrame, response: str, rf_model, rf_predictors, gwr_predictors, bw: int) -> np.ndarray:
    full_pred = rf_model.predict_proba(work[rf_predictors]) if hasattr(rf_model, "predict_proba") else rf_model.predict(work[rf_predictors])
    if hasattr(rf_model, "predict_proba"):
        full_pred = probs_to_expected(full_pred, rf_model.classes_)
    full_resid = work[response].to_numpy() - full_pred
    fit_full = sample_df(work.assign(_resid=full_resid), FULL_FIT_SAMPLE_N, RANDOM_STATE)
    gwr = GWR(
        fit_full[["x", "y"]].to_numpy(),
        fit_full["_resid"].to_numpy().reshape(-1, 1),
        fit_full[gwr_predictors].to_numpy(),
        bw=bw,
        fixed=False,
        kernel="bisquare",
        n_jobs=1,
    )
    corr = gwr.predict(work[["x", "y"]].to_numpy(), work[gwr_predictors].to_numpy()).predy.flatten()
    return full_pred + corr


def corrected_moran_on_sample(
    work: pd.DataFrame,
    response: str,
    rf_model,
    rf_predictors: list[str],
    gwr_predictors: list[str],
    bw: int,
) -> float:
    eval_df = sample_df(work, MORAN_SAMPLE_N, RANDOM_STATE)
    rf_pred = rf_model.predict_proba(eval_df[rf_predictors]) if hasattr(rf_model, "predict_proba") else rf_model.predict(eval_df[rf_predictors])
    if hasattr(rf_model, "predict_proba"):
        rf_pred = probs_to_expected(rf_pred, rf_model.classes_)
        all_rf_pred = probs_to_expected(rf_model.predict_proba(work[rf_predictors]), rf_model.classes_)
    else:
        all_rf_pred = rf_model.predict(work[rf_predictors])
    full_base = sample_df(work.assign(_rf_pred=all_rf_pred), FULL_FIT_SAMPLE_N, RANDOM_STATE)
    full_base["_resid"] = full_base[response] - full_base["_rf_pred"]
    gwr = GWR(
        full_base[["x", "y"]].to_numpy(),
        full_base["_resid"].to_numpy().reshape(-1, 1),
        full_base[gwr_predictors].to_numpy(),
        bw=bw,
        fixed=False,
        kernel="bisquare",
        n_jobs=1,
    )
    corr = gwr.predict(eval_df[["x", "y"]].to_numpy(), eval_df[gwr_predictors].to_numpy()).predy.flatten()
    corrected = rf_pred + corr
    return compute_moran(eval_df[["x", "y"]].to_numpy(), eval_df[response].to_numpy() - corrected)["moran_i"]


def run_regression(df: pd.DataFrame, response: str, rf_predictors: list[str], gwr_predictors: list[str]) -> dict:
    print("Running", response, flush=True)
    cols = [response] + list(dict.fromkeys(rf_predictors + gwr_predictors + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, test_idx = train_test_split(work.index, test_size=0.2, random_state=RANDOM_STATE)
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train[rf_predictors], train[response])
    rf_test = rf.predict(test[rf_predictors])
    train_resid = train[response].to_numpy() - rf.predict(train[rf_predictors])
    gwr_corr_test = gwr_predict(train[["x", "y"]].to_numpy(), train[gwr_predictors].to_numpy(), train_resid, test[["x", "y"]].to_numpy(), test[gwr_predictors].to_numpy(), FIXED_BW)
    corrected_test = rf_test + gwr_corr_test

    rf_full = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf_full.fit(work[rf_predictors], work[response])
    rf_full_pred = rf_full.predict(work[rf_predictors])
    corrected_moran_i = corrected_moran_on_sample(work, response, rf_full, rf_predictors, gwr_predictors, FIXED_BW)
    row = {
        "y": response,
        "family_or_scheme": "rf_plus_gwr_residual_fixedbw",
        "n_rows_used": int(len(work)),
        "rf_test_r2": float(r2_score(test[response], rf_test)),
        "corrected_test_r2": float(r2_score(test[response], corrected_test)),
        "rf_test_rmse": float(np.sqrt(mean_squared_error(test[response], rf_test))),
        "corrected_test_rmse": float(np.sqrt(mean_squared_error(test[response], corrected_test))),
        "rf_moran_i": compute_moran(work[["x", "y"]].to_numpy(), work[response].to_numpy() - rf_full_pred)["moran_i"],
        "corrected_moran_i": corrected_moran_i,
        "gwr_bw": FIXED_BW,
    }
    pd.DataFrame([row]).to_csv(OUT / f"{response}_row.csv", index=False)
    print(f"Finished {response}", flush=True)
    return row


def run_t80(df: pd.DataFrame) -> dict:
    print("Running T80_revised", flush=True)
    response = "T80_revised"
    rf_predictors = RECOVERY_RF_PREDICTORS
    gwr_predictors = RECOVERY_GWR_PREDICTORS
    cols = [response] + list(dict.fromkeys(rf_predictors + gwr_predictors + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, test_idx = train_test_split(work.index, test_size=0.2, random_state=RANDOM_STATE, stratify=work[response])
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train[rf_predictors], train[response])
    rf_test_prob = rf.predict_proba(test[rf_predictors])
    rf_test = probs_to_expected(rf_test_prob, rf.classes_)
    train_resid = train[response].to_numpy() - probs_to_expected(rf.predict_proba(train[rf_predictors]), rf.classes_)
    gwr_corr_test = gwr_predict(train[["x", "y"]].to_numpy(), train[gwr_predictors].to_numpy(), train_resid, test[["x", "y"]].to_numpy(), test[gwr_predictors].to_numpy(), FIXED_BW)
    corrected_test = rf_test + gwr_corr_test

    rf_full = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf_full.fit(work[rf_predictors], work[response])
    rf_full_pred = probs_to_expected(rf_full.predict_proba(work[rf_predictors]), rf_full.classes_)
    corrected_moran_i = corrected_moran_on_sample(work, response, rf_full, rf_predictors, gwr_predictors, FIXED_BW)
    row = {
        "y": response,
        "family_or_scheme": "ordinal_rf_plus_gwr_residual_fixedbw",
        "n_rows_used": int(len(work)),
        "rf_test_r2": float(r2_score(test[response], rf_test)),
        "corrected_test_r2": float(r2_score(test[response], corrected_test)),
        "rf_test_rmse": float(np.sqrt(mean_squared_error(test[response], rf_test))),
        "corrected_test_rmse": float(np.sqrt(mean_squared_error(test[response], corrected_test))),
        "rf_moran_i": compute_moran(work[["x", "y"]].to_numpy(), work[response].to_numpy() - rf_full_pred)["moran_i"],
        "corrected_moran_i": corrected_moran_i,
        "gwr_bw": FIXED_BW,
        "rf_accuracy": float(accuracy_score(test[response], rf.predict(test[rf_predictors]))),
        "rf_macro_f1": float(f1_score(test[response], rf.predict(test[rf_predictors]), average="macro")),
    }
    pd.DataFrame([row]).to_csv(OUT / f"{response}_row.csv", index=False)
    print("Finished T80_revised", flush=True)
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
    summary.to_csv(OUT / "gwr_residual_correction_all_y_fixedbw_summary.csv", index=False)
    lines = [f"GWR residual correction across all y (adaptive bw={FIXED_BW}; fast pass)", ""]
    for _, row in summary.iterrows():
        extra = ""
        if row["y"] == "T80_revised":
            extra = f", acc={row['rf_accuracy']:.4f}, macro_f1={row['rf_macro_f1']:.4f}"
        lines.append(
            f"{row['y']}: RF_r2={row['rf_test_r2']:.4f} -> corrected_r2={row['corrected_test_r2']:.4f}, "
            f"RF_rmse={row['rf_test_rmse']:.4f} -> corrected_rmse={row['corrected_test_rmse']:.4f}, "
            f"RF_Moran={row['rf_moran_i']:.4f} -> corrected_Moran={row['corrected_moran_i']:.4f}{extra}"
        )
    (OUT / "gwr_residual_correction_all_y_fixedbw_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
