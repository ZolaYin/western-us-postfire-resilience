from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


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
OUT_DIR = RECOVERY_BASE / "spatial_rf_all_y_screen_2026-03-31"

RANDOM_STATE = 42
MORAN_K = 8
NEIGHBOR_K = 8
WORK_SAMPLE_N = 12000
N_TREES = 300

RECOVERY_RF_PREDICTORS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z", "TS_twi_z", "TS_roughness_z",
    "TS_SOC_0_30cm_z", "FS_TCC_t0_z", "FS_CBH_t0agg_z", "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z", "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z", "CLIM_pr_sum_post_z", "CLIM_eto_sum_post_z", "CLIM_tmmn_mean_post_z",
    "CLIM_hot_days_35C_post_z", "CLIM_aridity_post_z", "CLIM_tmmx_std_post_z", "x", "y", "x_sq_z", "y_sq_z", "xy_z",
]
RESISTANCE_RF_PREDICTORS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z", "TS_twi_z", "TS_roughness_z",
    "TS_SOC_0_30cm_z", "FS_TCC_t0_z", "FS_CBH_t0agg_z", "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z", "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z", "CLIM_pr_sum_pre_z", "CLIM_eto_sum_pre_z", "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z", "CLIM_aridity_pre_z", "CLIM_tmmx_std_pre_z", "x", "y",
]
SPACE_COLS = {"x", "y", "x_sq_z", "y_sq_z", "xy_z"}
T80_CLASSES = np.arange(2, 11, dtype=float)


def add_spatial_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    x_sq = out["x"] ** 2
    y_sq = out["y"] ** 2
    xy = out["x"] * out["y"]
    out["x_sq_z"] = (x_sq - x_sq.mean()) / x_sq.std(ddof=0)
    out["y_sq_z"] = (y_sq - y_sq.mean()) / y_sq.std(ddof=0)
    out["xy_z"] = (xy - xy.mean()) / xy.std(ddof=0)
    return out


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


def compute_moran(df: pd.DataFrame, residuals: np.ndarray) -> float:
    weights = KNN.from_array(df[["x", "y"]].to_numpy(), k=MORAN_K)
    weights.transform = "R"
    return float(Moran(residuals.astype(float), weights, permutations=0).I)


def probs_to_expected(probs: np.ndarray) -> np.ndarray:
    return probs @ T80_CLASSES


def neighbor_mean_features(df: pd.DataFrame, feature_cols: list[str], prefix: str) -> pd.DataFrame:
    knn = NearestNeighbors(n_neighbors=NEIGHBOR_K + 1, algorithm="ball_tree")
    knn.fit(df[["x", "y"]].to_numpy())
    idx = knn.kneighbors(return_distance=False)[:, 1:]
    vals = df[feature_cols].to_numpy(dtype=float)
    out = {}
    for j, col in enumerate(feature_cols):
        out[f"{prefix}{col}"] = vals[idx, j].mean(axis=1)
    return pd.DataFrame(out, index=df.index)


def eval_regression(work: pd.DataFrame, response: str, predictors: list[str]) -> tuple[float, float, float, RandomForestRegressor]:
    train, test = train_test_split(work, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestRegressor(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train[predictors], train[response])
    pred_test = model.predict(test[predictors])

    full_model = RandomForestRegressor(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[predictors], work[response])
    full_pred = full_model.predict(work[predictors])
    moran_i = compute_moran(work, work[response].to_numpy() - full_pred)
    return float(r2_score(test[response], pred_test)), float(np.sqrt(mean_squared_error(test[response], pred_test))), moran_i, full_model


def eval_t80(work: pd.DataFrame, predictors: list[str]) -> tuple[float, float, float, float, float, RandomForestClassifier]:
    train, test = train_test_split(work, test_size=0.2, random_state=RANDOM_STATE, stratify=work["T80_revised"])
    model = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train[predictors], train["T80_revised"])
    pred_cls = model.predict(test[predictors])
    pred_exp = probs_to_expected(model.predict_proba(test[predictors]))

    full_model = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[predictors], work["T80_revised"])
    full_exp = probs_to_expected(full_model.predict_proba(work[predictors]))
    moran_i = compute_moran(work, work["T80_revised"].to_numpy() - full_exp)
    return (
        float(r2_score(test["T80_revised"], pred_exp)),
        float(np.sqrt(mean_squared_error(test["T80_revised"], pred_exp))),
        float(accuracy_score(test["T80_revised"], pred_cls)),
        float(f1_score(test["T80_revised"], pred_cls, average="macro")),
        moran_i,
        full_model,
    )


def save_importance(model, predictors: list[str], out_path: Path) -> None:
    pd.DataFrame({"predictor": predictors, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(out_path, index=False)


def run_one_regression(df: pd.DataFrame, response: str, base_predictors: list[str]) -> dict:
    non_space = [c for c in base_predictors if c not in SPACE_COLS]
    cols = list(dict.fromkeys([response] + base_predictors + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = sample_df(work, WORK_SAMPLE_N)
    nn_df = neighbor_mean_features(work, non_space, prefix=f"nn{NEIGHBOR_K}_mean_")
    work = pd.concat([work, nn_df], axis=1)
    spatial_predictors = list(dict.fromkeys(base_predictors + nn_df.columns.tolist()))

    base_r2, base_rmse, base_moran, _ = eval_regression(work, response, base_predictors)
    sp_r2, sp_rmse, sp_moran, sp_model = eval_regression(work, response, spatial_predictors)

    run_dir = OUT_DIR / response
    run_dir.mkdir(parents=True, exist_ok=True)
    save_importance(sp_model, spatial_predictors, run_dir / "spatial_rf_importance.csv")

    return {
        "y": response,
        "baseline_r2": base_r2,
        "spatial_rf_r2": sp_r2,
        "baseline_rmse": base_rmse,
        "spatial_rf_rmse": sp_rmse,
        "baseline_moran_i": base_moran,
        "spatial_rf_moran_i": sp_moran,
        "delta_r2": sp_r2 - base_r2,
        "delta_rmse": sp_rmse - base_rmse,
        "delta_moran_i": sp_moran - base_moran,
        "n_rows_used": int(len(work)),
    }


def run_t80(df: pd.DataFrame) -> dict:
    cols = list(dict.fromkeys(["T80_revised"] + RECOVERY_RF_PREDICTORS + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    work = sample_df(work, WORK_SAMPLE_N, stratify_col="T80_revised")
    non_space = [c for c in RECOVERY_RF_PREDICTORS if c not in SPACE_COLS]
    nn_df = neighbor_mean_features(work, non_space, prefix=f"nn{NEIGHBOR_K}_mean_")
    work = pd.concat([work, nn_df], axis=1)
    spatial_predictors = list(dict.fromkeys(RECOVERY_RF_PREDICTORS + nn_df.columns.tolist()))

    base_r2, base_rmse, base_acc, base_f1, base_moran, _ = eval_t80(work, RECOVERY_RF_PREDICTORS)
    sp_r2, sp_rmse, sp_acc, sp_f1, sp_moran, sp_model = eval_t80(work, spatial_predictors)

    run_dir = OUT_DIR / "T80_revised"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_importance(sp_model, spatial_predictors, run_dir / "spatial_rf_importance.csv")

    return {
        "y": "T80_revised",
        "baseline_r2": base_r2,
        "spatial_rf_r2": sp_r2,
        "baseline_rmse": base_rmse,
        "spatial_rf_rmse": sp_rmse,
        "baseline_moran_i": base_moran,
        "spatial_rf_moran_i": sp_moran,
        "baseline_accuracy": base_acc,
        "spatial_rf_accuracy": sp_acc,
        "baseline_macro_f1": base_f1,
        "spatial_rf_macro_f1": sp_f1,
        "delta_r2": sp_r2 - base_r2,
        "delta_rmse": sp_rmse - base_rmse,
        "delta_moran_i": sp_moran - base_moran,
        "n_rows_used": int(len(work)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resistance_df = add_spatial_terms(pd.read_parquet(RESISTANCE_TABLE))
    recovery_df = add_spatial_terms(pd.read_parquet(RECOVERY_TABLE))

    rows = [
        run_one_regression(resistance_df, "Resistance", RESISTANCE_RF_PREDICTORS),
        run_t80(recovery_df),
        run_one_regression(recovery_df, "IRI_good_10yr", RECOVERY_RF_PREDICTORS),
        run_one_regression(recovery_df, "STAB_10yr", RECOVERY_RF_PREDICTORS),
        run_one_regression(recovery_df, "INC_end_rel_10obs", RECOVERY_RF_PREDICTORS),
        run_one_regression(recovery_df, "INC_cum_rel_10obs", RECOVERY_RF_PREDICTORS),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "spatial_rf_all_y_screen_summary.csv", index=False)
    lines = [f"Spatial RF screening across all y (KNN mean features, k={NEIGHBOR_K}, n={WORK_SAMPLE_N})", ""]
    for _, row in summary.iterrows():
        line = (
            f"{row['y']}: baseline_r2={row['baseline_r2']:.4f}, spatial_rf_r2={row['spatial_rf_r2']:.4f}, "
            f"baseline_moran={row['baseline_moran_i']:.4f}, spatial_rf_moran={row['spatial_rf_moran_i']:.4f}"
        )
        if row["y"] == "T80_revised":
            line += f", baseline_acc={row['baseline_accuracy']:.4f}, spatial_rf_acc={row['spatial_rf_accuracy']:.4f}"
        lines.append(line)
    (OUT_DIR / "spatial_rf_all_y_screen_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "spatial_rf_all_y_screen_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
