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
OUT_DIR = RECOVERY_BASE / "spatial_rf_all_y_2026-03-31"

RANDOM_STATE = 42
MORAN_K = 8
NEIGHBOR_K = 8
MORAN_MAX_N = 15000

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


def sample_for_moran(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) <= MORAN_MAX_N:
        return df
    return df.sample(n=MORAN_MAX_N, random_state=RANDOM_STATE).copy()


def compute_moran(work: pd.DataFrame, residuals: np.ndarray) -> float:
    eval_df = sample_for_moran(work)
    if len(eval_df) != len(residuals):
        residuals = residuals[work.index.get_indexer(eval_df.index)]
    weights = KNN.from_array(eval_df[["x", "y"]].to_numpy(), k=MORAN_K)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return float(moran.I)


def neighbor_mean_features(df: pd.DataFrame, feature_cols: list[str], prefix: str) -> pd.DataFrame:
    coords = df[["x", "y"]].to_numpy()
    knn = NearestNeighbors(n_neighbors=NEIGHBOR_K + 1, algorithm="ball_tree")
    knn.fit(coords)
    indices = knn.kneighbors(return_distance=False)[:, 1:]
    values = df[feature_cols].to_numpy(dtype=float)
    out = {}
    for j, col in enumerate(feature_cols):
        out[f"{prefix}{col}"] = values[indices, j].mean(axis=1)
    return pd.DataFrame(out, index=df.index)


def probs_to_expected(probs: np.ndarray) -> np.ndarray:
    return probs @ T80_CLASSES


def evaluate_regression(work: pd.DataFrame, response: str, base_predictors: list[str], variant_name: str) -> dict:
    train_idx, test_idx = train_test_split(work.index, test_size=0.2, random_state=RANDOM_STATE)
    train = work.loc[train_idx]
    test = work.loc[test_idx]

    model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train[base_predictors], train[response])
    pred_test = model.predict(test[base_predictors])

    full_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[base_predictors], work[response])
    full_pred = full_model.predict(work[base_predictors])
    residuals = work[response].to_numpy() - full_pred

    return {
        "variant": variant_name,
        "model": full_model,
        "test_r2": float(r2_score(test[response], pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(test[response], pred_test))),
        "moran_i": compute_moran(work, residuals),
    }


def evaluate_t80(work: pd.DataFrame, predictors: list[str], variant_name: str) -> dict:
    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_revised"],
    )
    train = work.loc[train_idx]
    test = work.loc[test_idx]

    model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train[predictors], train["T80_revised"])
    pred_cls = model.predict(test[predictors])
    pred_exp = probs_to_expected(model.predict_proba(test[predictors]))

    full_model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[predictors], work["T80_revised"])
    full_exp = probs_to_expected(full_model.predict_proba(work[predictors]))
    residuals = work["T80_revised"].to_numpy() - full_exp

    return {
        "variant": variant_name,
        "model": full_model,
        "test_accuracy": float(accuracy_score(test["T80_revised"], pred_cls)),
        "test_macro_f1": float(f1_score(test["T80_revised"], pred_cls, average="macro")),
        "test_expected_r2": float(r2_score(test["T80_revised"], pred_exp)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], pred_exp))),
        "moran_i": compute_moran(work, residuals),
    }


def save_importance(model, predictors: list[str], out_path: Path) -> None:
    pd.DataFrame({"predictor": predictors, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(out_path, index=False)


def prepare_recovery_df() -> pd.DataFrame:
    df = add_spatial_terms(pd.read_parquet(RECOVERY_TABLE))
    nn_features = neighbor_mean_features(
        df,
        [c for c in RECOVERY_RF_PREDICTORS if c not in SPACE_COLS],
        prefix=f"nn{NEIGHBOR_K}_mean_",
    )
    return pd.concat([df, nn_features], axis=1)


def prepare_resistance_df() -> pd.DataFrame:
    df = add_spatial_terms(pd.read_parquet(RESISTANCE_TABLE))
    nn_features = neighbor_mean_features(
        df,
        [c for c in RESISTANCE_RF_PREDICTORS if c not in SPACE_COLS],
        prefix=f"nn{NEIGHBOR_K}_mean_",
    )
    return pd.concat([df, nn_features], axis=1)


def run_resistance(df: pd.DataFrame) -> dict:
    base = RESISTANCE_RF_PREDICTORS
    extra = [f"nn{NEIGHBOR_K}_mean_{c}" for c in base if c not in SPACE_COLS]
    cols = list(dict.fromkeys(["Resistance"] + base + extra + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    baseline = evaluate_regression(work, "Resistance", base, "rf_plusxy")
    spatial = evaluate_regression(work, "Resistance", base + extra, f"spatial_rf_knn{NEIGHBOR_K}")

    run_dir = OUT_DIR / "Resistance"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_importance(spatial["model"], base + extra, run_dir / "spatial_rf_importance.csv")
    return {
        "y": "Resistance",
        "baseline_variant": baseline["variant"],
        "baseline_r2": baseline["test_r2"],
        "spatial_rf_r2": spatial["test_r2"],
        "baseline_rmse": baseline["test_rmse"],
        "spatial_rf_rmse": spatial["test_rmse"],
        "baseline_moran_i": baseline["moran_i"],
        "spatial_rf_moran_i": spatial["moran_i"],
        "delta_r2": spatial["test_r2"] - baseline["test_r2"],
        "delta_rmse": spatial["test_rmse"] - baseline["test_rmse"],
        "delta_moran_i": spatial["moran_i"] - baseline["moran_i"],
        "n_rows_used": int(len(work)),
    }


def run_continuous_recovery(df: pd.DataFrame, response: str) -> dict:
    base = RECOVERY_RF_PREDICTORS
    extra = [f"nn{NEIGHBOR_K}_mean_{c}" for c in base if c not in SPACE_COLS]
    cols = list(dict.fromkeys([response] + base + extra + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    baseline = evaluate_regression(work, response, base, "rf_plusxy_poly")
    spatial = evaluate_regression(work, response, base + extra, f"spatial_rf_knn{NEIGHBOR_K}")

    run_dir = OUT_DIR / response
    run_dir.mkdir(parents=True, exist_ok=True)
    save_importance(spatial["model"], base + extra, run_dir / "spatial_rf_importance.csv")
    return {
        "y": response,
        "baseline_variant": baseline["variant"],
        "baseline_r2": baseline["test_r2"],
        "spatial_rf_r2": spatial["test_r2"],
        "baseline_rmse": baseline["test_rmse"],
        "spatial_rf_rmse": spatial["test_rmse"],
        "baseline_moran_i": baseline["moran_i"],
        "spatial_rf_moran_i": spatial["moran_i"],
        "delta_r2": spatial["test_r2"] - baseline["test_r2"],
        "delta_rmse": spatial["test_rmse"] - baseline["test_rmse"],
        "delta_moran_i": spatial["moran_i"] - baseline["moran_i"],
        "n_rows_used": int(len(work)),
    }


def run_t80(df: pd.DataFrame) -> dict:
    base = RECOVERY_RF_PREDICTORS
    extra = [f"nn{NEIGHBOR_K}_mean_{c}" for c in base if c not in SPACE_COLS]
    cols = list(dict.fromkeys(["T80_revised"] + base + extra + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    baseline = evaluate_t80(work, base, "ordinal_rf_plusxy_poly")
    spatial = evaluate_t80(work, base + extra, f"ordinal_spatial_rf_knn{NEIGHBOR_K}")

    run_dir = OUT_DIR / "T80_revised"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_importance(spatial["model"], base + extra, run_dir / "spatial_rf_importance.csv")
    return {
        "y": "T80_revised",
        "baseline_variant": baseline["variant"],
        "baseline_r2": baseline["test_expected_r2"],
        "spatial_rf_r2": spatial["test_expected_r2"],
        "baseline_rmse": baseline["test_expected_rmse"],
        "spatial_rf_rmse": spatial["test_expected_rmse"],
        "baseline_moran_i": baseline["moran_i"],
        "spatial_rf_moran_i": spatial["moran_i"],
        "baseline_accuracy": baseline["test_accuracy"],
        "spatial_rf_accuracy": spatial["test_accuracy"],
        "baseline_macro_f1": baseline["test_macro_f1"],
        "spatial_rf_macro_f1": spatial["test_macro_f1"],
        "delta_r2": spatial["test_expected_r2"] - baseline["test_expected_r2"],
        "delta_rmse": spatial["test_expected_rmse"] - baseline["test_expected_rmse"],
        "delta_moran_i": spatial["moran_i"] - baseline["moran_i"],
        "n_rows_used": int(len(work)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resistance_df = prepare_resistance_df()
    recovery_df = prepare_recovery_df()

    rows = [
        run_resistance(resistance_df),
        run_t80(recovery_df),
        run_continuous_recovery(recovery_df, "IRI_good_10yr"),
        run_continuous_recovery(recovery_df, "STAB_10yr"),
        run_continuous_recovery(recovery_df, "INC_end_rel_10obs"),
        run_continuous_recovery(recovery_df, "INC_cum_rel_10obs"),
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "spatial_rf_all_y_summary.csv", index=False)
    lines = [f"Spatial RF across all y (KNN mean features, k={NEIGHBOR_K})", ""]
    for _, row in summary.iterrows():
        line = (
            f"{row['y']}: baseline_r2={row['baseline_r2']:.4f}, spatial_rf_r2={row['spatial_rf_r2']:.4f}, "
            f"baseline_moran={row['baseline_moran_i']:.4f}, spatial_rf_moran={row['spatial_rf_moran_i']:.4f}"
        )
        if row["y"] == "T80_revised":
            line += f", baseline_acc={row['baseline_accuracy']:.4f}, spatial_rf_acc={row['spatial_rf_accuracy']:.4f}"
        lines.append(line)
    (OUT_DIR / "spatial_rf_all_y_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "spatial_rf_all_y_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
