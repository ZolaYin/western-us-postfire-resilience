from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
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
OUT = BASE / "gwr_residual_correction_all_y_2026-03-31"

RANDOM_STATE = 42
MORAN_K = 8
SEARCH_SAMPLE_N = 1200
FIT_SAMPLE_N = 2500
FULL_FIT_SAMPLE_N = 3000
CLASSES = np.array(list(range(2, 11)))


RECOVERY_RF_PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z",
    "HUM_roaddens_r5km_z",
    "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_post_z",
    "CLIM_eto_sum_post_z",
    "CLIM_tmmn_mean_post_z",
    "CLIM_hot_days_35C_post_z",
    "CLIM_aridity_post_z",
    "CLIM_tmmx_std_post_z",
    "x",
    "y",
    "x_sq_z",
    "y_sq_z",
    "xy_z",
]

RECOVERY_GWR_PREDICTORS = [
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

RESISTANCE_RF_PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z",
    "HUM_roaddens_r5km_z",
    "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_eto_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_tmmx_std_pre_z",
    "x",
    "y",
]

RESISTANCE_GWR_PREDICTORS = [
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


def add_spatial_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    x_sq = out["x"] ** 2
    y_sq = out["y"] ** 2
    xy = out["x"] * out["y"]
    out["x_sq_z"] = (x_sq - x_sq.mean()) / x_sq.std(ddof=0)
    out["y_sq_z"] = (y_sq - y_sq.mean()) / y_sq.std(ddof=0)
    out["xy_z"] = (xy - xy.mean()) / xy.std(ddof=0)
    return out


def compute_moran(coords: np.ndarray, residuals: np.ndarray) -> dict:
    w = KNN.from_array(coords, k=MORAN_K)
    w.transform = "R"
    moran = Moran(residuals.astype(float), w, permutations=0)
    return {
        "k": MORAN_K,
        "n_obs": int(len(coords)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def probs_to_expected(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return probs @ classes.astype(float)


def fit_gwr_and_predict(
    train_coords: np.ndarray,
    train_X: np.ndarray,
    train_resid: np.ndarray,
    pred_coords: np.ndarray,
    pred_X: np.ndarray,
    seed: int,
) -> tuple[float, np.ndarray]:
    train_df = pd.DataFrame(train_X)
    train_df["resid"] = train_resid
    train_df["xcoord"] = train_coords[:, 0]
    train_df["ycoord"] = train_coords[:, 1]

    search_df = sample_df(train_df, SEARCH_SAMPLE_N, seed)
    fit_df = sample_df(train_df, FIT_SAMPLE_N, seed)

    coords_search = search_df[["xcoord", "ycoord"]].to_numpy()
    X_search = search_df.drop(columns=["resid", "xcoord", "ycoord"]).to_numpy()
    y_search = search_df["resid"].to_numpy().reshape(-1, 1)

    print(f"  GWR bw search on n={len(search_df)}; fit sample n={len(fit_df)}; predict n={len(pred_coords)}", flush=True)
    bw_selector = Sel_BW(coords_search, y_search, X_search, fixed=False, kernel="bisquare", n_jobs=1)
    bw = bw_selector.search(bw_min=40)
    print(f"  selected bw={bw}", flush=True)

    coords_fit = fit_df[["xcoord", "ycoord"]].to_numpy()
    X_fit = fit_df.drop(columns=["resid", "xcoord", "ycoord"]).to_numpy()
    y_fit = fit_df["resid"].to_numpy().reshape(-1, 1)
    gwr = GWR(coords_fit, y_fit, X_fit, bw=bw, fixed=False, kernel="bisquare", n_jobs=1)
    pred = gwr.predict(pred_coords, pred_X)
    return float(bw), pred.predy.flatten()


def run_regression_response(
    df: pd.DataFrame,
    response: str,
    rf_predictors: list[str],
    gwr_predictors: list[str],
    out_dir: Path,
) -> dict:
    print(f"Running regression response: {response}", flush=True)
    cols = [response] + list(dict.fromkeys(rf_predictors + gwr_predictors + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, test_idx = train_test_split(work.index, test_size=0.2, random_state=RANDOM_STATE)
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train[rf_predictors], train[response])
    rf_test = rf.predict(test[rf_predictors])
    rf_train = rf.predict(train[rf_predictors])
    train_resid = train[response].to_numpy() - rf_train

    bw, gwr_corr_test = fit_gwr_and_predict(
        train[["x", "y"]].to_numpy(),
        train[gwr_predictors].to_numpy(),
        train_resid,
        test[["x", "y"]].to_numpy(),
        test[gwr_predictors].to_numpy(),
        seed=RANDOM_STATE,
    )
    corrected_test = rf_test + gwr_corr_test

    full_rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_rf.fit(work[rf_predictors], work[response])
    full_pred = full_rf.predict(work[rf_predictors])
    full_resid = work[response].to_numpy() - full_pred

    fit_full = sample_df(work.assign(_resid=full_resid), FULL_FIT_SAMPLE_N, RANDOM_STATE)
    gwr_full = GWR(
        fit_full[["x", "y"]].to_numpy(),
        fit_full["_resid"].to_numpy().reshape(-1, 1),
        fit_full[gwr_predictors].to_numpy(),
        bw=bw,
        fixed=False,
        kernel="bisquare",
        n_jobs=1,
    )
    corrected_full = full_pred + gwr_full.predict(work[["x", "y"]].to_numpy(), work[gwr_predictors].to_numpy()).predy.flatten()
    corrected_resid_full = work[response].to_numpy() - corrected_full

    metrics = {
        "response": response,
        "scheme": "rf_plus_gwr_residual",
        "n_rows_used": int(len(work)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "rf_predictors": rf_predictors,
        "gwr_predictors": gwr_predictors,
        "gwr_bw": bw,
        "rf_test_r2": float(r2_score(test[response], rf_test)),
        "rf_test_rmse": float(np.sqrt(mean_squared_error(test[response], rf_test))),
        "corrected_test_r2": float(r2_score(test[response], corrected_test)),
        "corrected_test_rmse": float(np.sqrt(mean_squared_error(test[response], corrected_test))),
        "rf_full_residual_moran": compute_moran(work[["x", "y"]].to_numpy(), full_resid),
        "corrected_full_residual_moran": compute_moran(work[["x", "y"]].to_numpy(), corrected_resid_full),
    }
    (out_dir / f"{response}_gwr_residual_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {
        "y": response,
        "family_or_scheme": "rf_plus_gwr_residual",
        "n_rows_used": int(len(work)),
        "rf_test_r2": metrics["rf_test_r2"],
        "corrected_test_r2": metrics["corrected_test_r2"],
        "rf_test_rmse": metrics["rf_test_rmse"],
        "corrected_test_rmse": metrics["corrected_test_rmse"],
        "rf_moran_i": metrics["rf_full_residual_moran"]["moran_i"],
        "corrected_moran_i": metrics["corrected_full_residual_moran"]["moran_i"],
        "gwr_bw": bw,
    }


def run_t80_response(df: pd.DataFrame, out_dir: Path) -> dict:
    response = "T80_revised"
    print("Running ordinal response: T80_revised", flush=True)
    rf_predictors = RECOVERY_RF_PREDICTORS
    gwr_predictors = RECOVERY_GWR_PREDICTORS
    cols = [response] + list(dict.fromkeys(rf_predictors + gwr_predictors + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, test_idx = train_test_split(
        work.index, test_size=0.2, random_state=RANDOM_STATE, stratify=work[response]
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(train[rf_predictors], train[response])
    probs_test = rf.predict_proba(test[rf_predictors])
    rf_test = probs_to_expected(probs_test, rf.classes_)
    pred_labels = rf.predict(test[rf_predictors])
    rf_train = probs_to_expected(rf.predict_proba(train[rf_predictors]), rf.classes_)
    train_resid = train[response].to_numpy() - rf_train

    bw, gwr_corr_test = fit_gwr_and_predict(
        train[["x", "y"]].to_numpy(),
        train[gwr_predictors].to_numpy(),
        train_resid,
        test[["x", "y"]].to_numpy(),
        test[gwr_predictors].to_numpy(),
        seed=RANDOM_STATE,
    )
    corrected_test = rf_test + gwr_corr_test

    full_rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_rf.fit(work[rf_predictors], work[response])
    full_pred = probs_to_expected(full_rf.predict_proba(work[rf_predictors]), full_rf.classes_)
    full_resid = work[response].to_numpy() - full_pred

    fit_full = sample_df(work.assign(_resid=full_resid), FULL_FIT_SAMPLE_N, RANDOM_STATE)
    gwr_full = GWR(
        fit_full[["x", "y"]].to_numpy(),
        fit_full["_resid"].to_numpy().reshape(-1, 1),
        fit_full[gwr_predictors].to_numpy(),
        bw=bw,
        fixed=False,
        kernel="bisquare",
        n_jobs=1,
    )
    corrected_full = full_pred + gwr_full.predict(work[["x", "y"]].to_numpy(), work[gwr_predictors].to_numpy()).predy.flatten()
    corrected_resid_full = work[response].to_numpy() - corrected_full

    metrics = {
        "response": response,
        "scheme": "ordinal_rf_plus_gwr_residual",
        "n_rows_used": int(len(work)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "rf_predictors": rf_predictors,
        "gwr_predictors": gwr_predictors,
        "gwr_bw": bw,
        "rf_test_accuracy": float(accuracy_score(test[response], pred_labels)),
        "rf_test_macro_f1": float(f1_score(test[response], pred_labels, average="macro")),
        "rf_test_expected_r2": float(r2_score(test[response], rf_test)),
        "rf_test_expected_rmse": float(np.sqrt(mean_squared_error(test[response], rf_test))),
        "corrected_test_expected_r2": float(r2_score(test[response], corrected_test)),
        "corrected_test_expected_rmse": float(np.sqrt(mean_squared_error(test[response], corrected_test))),
        "rf_full_residual_moran": compute_moran(work[["x", "y"]].to_numpy(), full_resid),
        "corrected_full_residual_moran": compute_moran(work[["x", "y"]].to_numpy(), corrected_resid_full),
    }
    (out_dir / f"{response}_gwr_residual_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {
        "y": response,
        "family_or_scheme": "ordinal_rf_plus_gwr_residual",
        "n_rows_used": int(len(work)),
        "rf_test_r2": metrics["rf_test_expected_r2"],
        "corrected_test_r2": metrics["corrected_test_expected_r2"],
        "rf_test_rmse": metrics["rf_test_expected_rmse"],
        "corrected_test_rmse": metrics["corrected_test_expected_rmse"],
        "rf_accuracy": metrics["rf_test_accuracy"],
        "rf_macro_f1": metrics["rf_test_macro_f1"],
        "rf_moran_i": metrics["rf_full_residual_moran"]["moran_i"],
        "corrected_moran_i": metrics["corrected_full_residual_moran"]["moran_i"],
        "gwr_bw": bw,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    recovery_df = add_spatial_terms(pd.read_parquet(RECOVERY_TABLE))
    resistance_df = add_spatial_terms(pd.read_parquet(RESISTANCE_TABLE))

    summary_rows = []
    summary_rows.append(run_regression_response(resistance_df, "Resistance", RESISTANCE_RF_PREDICTORS, RESISTANCE_GWR_PREDICTORS, OUT))
    summary_rows.append(run_t80_response(recovery_df, OUT))
    for response in ["IRI_good_10yr", "STAB_10yr", "INC_end_rel_10obs", "INC_cum_rel_10obs"]:
        summary_rows.append(run_regression_response(recovery_df, response, RECOVERY_RF_PREDICTORS, RECOVERY_GWR_PREDICTORS, OUT))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "gwr_residual_correction_all_y_summary.csv", index=False)

    lines = ["GWR residual correction across all y", ""]
    for _, row in summary.iterrows():
        extra = ""
        if row["y"] == "T80_revised":
            extra = f", acc={row['rf_accuracy']:.4f}, macro_f1={row['rf_macro_f1']:.4f}"
        lines.append(
            f"{row['y']}: RF_r2={row['rf_test_r2']:.4f} -> corrected_r2={row['corrected_test_r2']:.4f}, "
            f"RF_rmse={row['rf_test_rmse']:.4f} -> corrected_rmse={row['corrected_test_rmse']:.4f}, "
            f"RF_Moran={row['rf_moran_i']:.4f} -> corrected_Moran={row['corrected_moran_i']:.4f}, "
            f"bw={row['gwr_bw']:.1f}{extra}"
        )
    (OUT / "gwr_residual_correction_all_y_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
