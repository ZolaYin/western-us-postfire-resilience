from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_increment_recovery_near_t0_postfire.parquet"
OUT_DIR = BASE_DIR / "t80_spatial_ordinal_rf_refined_2026-03-31"
RANDOM_STATE = 42
MORAN_K = 8
NEIGHBOR_K = 8
CLASSES = np.arange(2, 11, dtype=float)

BASE_PREDICTORS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z", "TS_twi_z", "TS_roughness_z",
    "TS_SOC_0_30cm_z", "FS_TCC_t0_z", "FS_CBH_t0agg_z", "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z", "HUM_imperv_near_t0_z",
    "HUM_viirs_near_t0_log_z", "CLIM_pr_sum_post_z", "CLIM_eto_sum_post_z", "CLIM_tmmn_mean_post_z",
    "CLIM_hot_days_35C_post_z", "CLIM_aridity_post_z", "CLIM_tmmx_std_post_z", "x", "y", "x_sq_z", "y_sq_z", "xy_z",
]
NEIGHBOR_SOURCE = [
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "FS_EVT_t0agg_resistance_proxy",
    "HUM_popdens_win10km_log_z",
    "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_post_z",
    "CLIM_aridity_post_z",
    "CLIM_tmmx_std_post_z",
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


def probs_to_expected_year(probs: np.ndarray) -> np.ndarray:
    return probs @ CLASSES


def compute_moran(coords_df: pd.DataFrame, residuals: np.ndarray) -> float:
    weights = KNN.from_array(coords_df[["x", "y"]].to_numpy(), k=MORAN_K)
    weights.transform = "R"
    return float(Moran(residuals.astype(float), weights, permutations=0).I)


def add_neighbor_features(df: pd.DataFrame, source_cols: list[str]) -> pd.DataFrame:
    knn = NearestNeighbors(n_neighbors=NEIGHBOR_K + 1, algorithm="ball_tree")
    knn.fit(df[["x", "y"]].to_numpy())
    idx = knn.kneighbors(return_distance=False)[:, 1:]
    vals = df[source_cols].to_numpy(dtype=float)
    out = {}
    for j, col in enumerate(source_cols):
        out[f"nn{NEIGHBOR_K}_mean_{col}"] = vals[idx, j].mean(axis=1)
    return pd.concat([df, pd.DataFrame(out, index=df.index)], axis=1)


def fit_and_score(work: pd.DataFrame, predictors: list[str], variant: str, class_weight=None) -> tuple[dict, RandomForestClassifier]:
    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_revised"],
    )
    train = work.loc[train_idx]
    test = work.loc[test_idx]

    model = RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
        min_samples_leaf=3,
    )
    model.fit(train[predictors], train["T80_revised"])
    pred_cls = model.predict(test[predictors])
    pred_exp = probs_to_expected_year(model.predict_proba(test[predictors]))

    full_model = RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=class_weight,
        min_samples_leaf=3,
    )
    full_model.fit(work[predictors], work["T80_revised"])
    full_exp = probs_to_expected_year(full_model.predict_proba(work[predictors]))
    residuals = work["T80_revised"].to_numpy() - full_exp

    metrics = {
        "variant": variant,
        "n_rows_used": int(len(work)),
        "test_accuracy": float(accuracy_score(test["T80_revised"], pred_cls)),
        "test_macro_f1": float(f1_score(test["T80_revised"], pred_cls, average="macro")),
        "test_expected_r2": float(r2_score(test["T80_revised"], pred_exp)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], pred_exp))),
        "moran_i": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
        "class_weight": class_weight,
    }
    return metrics, full_model


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = add_spatial_terms(pd.read_parquet(INPUT_PATH))
    df = add_neighbor_features(df, NEIGHBOR_SOURCE)
    neighbor_cols = [f"nn{NEIGHBOR_K}_mean_{c}" for c in NEIGHBOR_SOURCE]

    cols = list(dict.fromkeys(["T80_revised"] + BASE_PREDICTORS + neighbor_cols + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    rows = []
    baseline_metrics, baseline_model = fit_and_score(work, BASE_PREDICTORS, "ordinal_rf_plusxy_poly_refit")
    rows.append(baseline_metrics)
    refined_predictors = list(dict.fromkeys(BASE_PREDICTORS + neighbor_cols))
    refined_metrics, refined_model = fit_and_score(
        work,
        refined_predictors,
        f"ordinal_spatial_rf_knn{NEIGHBOR_K}_balanced",
        class_weight="balanced_subsample",
    )
    rows.append(refined_metrics)

    summary = pd.DataFrame(rows)
    summary["delta_expected_r2_vs_baseline"] = summary["test_expected_r2"] - baseline_metrics["test_expected_r2"]
    summary["delta_expected_rmse_vs_baseline"] = summary["test_expected_rmse"] - baseline_metrics["test_expected_rmse"]
    summary["delta_moran_vs_baseline"] = summary["moran_i"] - baseline_metrics["moran_i"]
    summary.to_csv(OUT_DIR / "t80_spatial_ordinal_rf_refined_summary.csv", index=False)

    pd.DataFrame({"predictor": BASE_PREDICTORS, "importance": baseline_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(OUT_DIR / "t80_baseline_refit_importance.csv", index=False)
    pd.DataFrame({"predictor": refined_predictors, "importance": refined_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(OUT_DIR / "t80_spatial_refined_importance.csv", index=False)

    (OUT_DIR / "t80_spatial_ordinal_rf_refined_summary.txt").write_text(
        "\n".join(
            [
                "T80 refined ordinal spatial RF comparison",
                *[
                    (
                        f"{row['variant']}: expected_r2={row['test_expected_r2']:.4f}, "
                        f"expected_rmse={row['test_expected_rmse']:.4f}, "
                        f"accuracy={row['test_accuracy']:.4f}, macro_f1={row['test_macro_f1']:.4f}, "
                        f"moran_i={row['moran_i']:.4f}"
                    )
                    for _, row in summary.iterrows()
                ],
            ]
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "t80_spatial_ordinal_rf_refined_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
