#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
RANDOM_RF_METRICS = (
    ROOT
    / "westernus_rf_noevt_plusxy_complete_corrected_20260411"
    / "westernus_rf_noevt_plusxy_complete_corrected_metrics.json"
)
OUT_DIR = ROOT / "spatial_block_validation_complete_rf_20260411"
OUT_SUMMARY = OUT_DIR / "spatial_block_validation_summary.csv"
OUT_FOLDS = OUT_DIR / "spatial_block_validation_fold_metrics.csv"
OUT_REPORT = OUT_DIR / "spatial_block_validation_report.md"
OUT_JSON = OUT_DIR / "spatial_block_validation_best.json"

RESPONSE = "Resistance"
RANDOM_STATE = 42
N_ESTIMATORS = 500
BLOCK_SIZES_KM = [50, 100, 200]
TEST_SIZE = 0.2
N_SPLITS = 5

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

BASE_TO_Z = {
    "TS_elev_m_z": "TS_elev_m",
    "TS_slope_deg_z": "TS_slope_deg",
    "TS_northness_z": "TS_northness",
    "TS_eastness_z": "TS_eastness",
    "TS_twi_z": "TS_twi",
    "TS_roughness_z": "TS_roughness",
    "TS_SOC_0_30cm_z": "TS_SOC_0_30cm",
    "FS_TCC_t0_z": "FS_TCC_t0",
    "FS_CBH_t0agg_z": "FS_CBH_t0agg",
    "HUM_roaddens_r5km_z": "HUM_roaddens_r5km",
    "HUM_traildens_r10km_z": "HUM_traildens_r10km",
    "HUM_imperv_near_t0_z": "HUM_imperv_near_t0",
    "CLIM_pr_sum_pre_z": "CLIM_pr_sum_pre",
    "CLIM_eto_sum_pre_z": "CLIM_eto_sum_pre",
    "CLIM_tmmn_mean_pre_z": "CLIM_tmmn_mean_pre",
    "CLIM_hot_days_35C_pre_z": "CLIM_hot_days_35C_pre",
    "CLIM_aridity_pre_z": "CLIM_aridity_pre",
    "CLIM_tmmx_std_pre_z": "CLIM_tmmx_std_pre",
}


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def log1p_z(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float).clip(lower=0)
    return zscore(np.log1p(clean))


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for z_col, base_col in BASE_TO_Z.items():
        if z_col not in out.columns:
            out[z_col] = zscore(out[base_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    return out


def fit_and_predict(X_train, y_train, X_test, y_test) -> tuple[dict[str, float], np.ndarray]:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return (
        {
            "r2": float(r2_score(y_test, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        },
        pred,
    )


def make_groups(df: pd.DataFrame, block_km: int) -> pd.Series:
    block_m = block_km * 1000.0
    gx = np.floor(df["x"] / block_m).astype(int)
    gy = np.floor(df["y"] / block_m).astype(int)
    return gx.astype(str) + "_" + gy.astype(str)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT).copy()
    df = ensure_columns(df)
    work = (
        df[[RESPONSE] + PREDICTORS]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
        .copy()
    )
    X = work[PREDICTORS]
    y = work[RESPONSE]

    random_metrics = json.loads(RANDOM_RF_METRICS.read_text())

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []

    for block_km in BLOCK_SIZES_KM:
        groups = make_groups(work, block_km)
        n_groups = int(groups.nunique())

        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        hold, _ = fit_and_predict(X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx], y.iloc[test_idx])
        test_groups = int(pd.Series(groups.iloc[test_idx]).nunique())

        summary_rows.append(
            {
                "block_km": block_km,
                "scheme": "group_holdout_20pct",
                "rows_used": int(len(work)),
                "n_groups": n_groups,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "test_groups": test_groups,
                "r2": hold["r2"],
                "rmse": hold["rmse"],
                "random_split_rf_r2": float(random_metrics["test_r2"]),
                "delta_vs_random": hold["r2"] - float(random_metrics["test_r2"]),
            }
        )

        gkf = GroupKFold(n_splits=min(N_SPLITS, n_groups))
        pooled_true = np.full(len(work), np.nan, dtype=np.float64)
        pooled_pred = np.full(len(work), np.nan, dtype=np.float64)

        for fold_id, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
            scores, pred = fit_and_predict(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[te_idx], y.iloc[te_idx])
            pooled_true[te_idx] = y.iloc[te_idx].to_numpy()
            pooled_pred[te_idx] = pred
            fold_rows.append(
                {
                    "block_km": block_km,
                    "scheme": "group_kfold",
                    "fold": fold_id,
                    "train_rows": int(len(tr_idx)),
                    "test_rows": int(len(te_idx)),
                    "train_groups": int(pd.Series(groups.iloc[tr_idx]).nunique()),
                    "test_groups": int(pd.Series(groups.iloc[te_idx]).nunique()),
                    "r2": scores["r2"],
                    "rmse": scores["rmse"],
                }
            )

        mask = np.isfinite(pooled_true) & np.isfinite(pooled_pred)
        pooled_r2 = float(r2_score(pooled_true[mask], pooled_pred[mask]))
        pooled_rmse = float(np.sqrt(mean_squared_error(pooled_true[mask], pooled_pred[mask])))
        summary_rows.append(
            {
                "block_km": block_km,
                "scheme": "group_kfold_pooled",
                "rows_used": int(mask.sum()),
                "n_groups": n_groups,
                "train_rows": np.nan,
                "test_rows": np.nan,
                "test_groups": np.nan,
                "r2": pooled_r2,
                "rmse": pooled_rmse,
                "random_split_rf_r2": float(random_metrics["test_r2"]),
                "delta_vs_random": pooled_r2 - float(random_metrics["test_r2"]),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["block_km", "scheme"]).reset_index(drop=True)
    folds = pd.DataFrame(fold_rows).sort_values(["block_km", "fold"]).reset_index(drop=True)
    summary.to_csv(OUT_SUMMARY, index=False)
    folds.to_csv(OUT_FOLDS, index=False)

    best = summary.loc[summary["scheme"] == "group_holdout_20pct"].sort_values("r2", ascending=False).iloc[0].to_dict()
    OUT_JSON.write_text(json.dumps(best, indent=2), encoding="utf-8")

    lines = [
        "# Spatial Block Validation For Complete Corrected RF",
        "",
        f"- Input table: `{INPUT}`",
        f"- Rows used after NA filtering: `{len(work)}`",
        f"- Random-split RF reference R2: `{float(random_metrics['test_r2']):.6f}`",
        f"- RF predictor count: `{len(PREDICTORS)}`",
        f"- Block sizes tested (km): `{', '.join(str(x) for x in BLOCK_SIZES_KM)}`",
        "",
        "Summary:",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{int(row['block_km'])} km` `{row['scheme']}`: R2=`{row['r2']:.6f}`, RMSE=`{row['rmse']:.6f}`, delta_vs_random=`{row['delta_vs_random']:.6f}`"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- The predictor set is unchanged from the complete corrected RF_noEVT_plusXY run.",
            "- The only change is the split rule: groups are fixed EPSG:5070 grid blocks.",
            "- `x` and `y` remain in the predictor set, so this is stricter than random split but still not a coordinate-free test.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"summary_csv": str(OUT_SUMMARY), "fold_csv": str(OUT_FOLDS), "best": best}, ensure_ascii=False))


if __name__ == "__main__":
    main()
