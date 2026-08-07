#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_tcc_build_20260410" / "westernus_current_candidate_table_plus_cbh_tcc.parquet"
BASE_MODEL_METRICS = ROOT / "reduced_noevt_models_20260410" / "reduced_model_metrics.csv"
OUT_DIR = ROOT / "bi_threshold_sensitivity_20260410"

RESPONSE = "Resistance"
PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_1km_z",
    "FS_EVT_resistance_proxy",
    "HUM_popdens_win10km_log_z",
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_eto_sum_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_tmmx_std_pre_z",
]
BASE_TO_Z = {
    "TS_elev_m_z": "TS_elev_m",
    "TS_slope_deg_z": "TS_slope_deg",
    "TS_twi_z": "TS_twi",
    "TS_roughness_z": "TS_roughness",
    "TS_SOC_0_30cm_z": "TS_SOC_0_30cm",
    "FS_TCC_t0_z": "FS_TCC_t0",
    "FS_CBH_1km_z": "FS_CBH_1km",
    "HUM_imperv_near_t0_z": "HUM_imperv_near_t0",
    "CLIM_pr_sum_pre_z": "CLIM_pr_sum_pre",
    "CLIM_eto_sum_pre_z": "CLIM_eto_sum_pre",
    "CLIM_aridity_pre_z": "CLIM_aridity_pre",
    "CLIM_hot_days_35C_pre_z": "CLIM_hot_days_35C_pre",
    "CLIM_tmmx_std_pre_z": "CLIM_tmmx_std_pre",
}
THRESHOLDS = [0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2]
RERUN_THRESHOLDS = [1e-5, 1e-4, 1e-3]
RANDOM_STATE = 42


def zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series), dtype=np.float32), index=series.index)
    return ((series - mean) / std).astype(np.float32)


def log1p_z(series: pd.Series) -> pd.Series:
    clean = series.astype(float).clip(lower=0)
    return zscore(np.log1p(clean))


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for z_col, base_col in BASE_TO_Z.items():
        if z_col not in out.columns:
            out[z_col] = zscore(out[base_col].astype(float))
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    return out


def dist_stats(s: pd.Series) -> dict:
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "skew": float(s.skew()),
        "kurt": float(s.kurt()),
        "gt_10_frac": float((s > 10).mean()),
        "gt_100_frac": float((s > 100).mean()),
        "gt_1000_frac": float((s > 1000).mean()),
    }


def run_models(df: pd.DataFrame) -> list[dict]:
    rows = []
    work = df[[RESPONSE] + PREDICTORS].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 50:
        return rows

    X = work[PREDICTORS]
    y = work[RESPONSE]

    X_sm = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y, X_sm).fit()
    ols_pred = ols.predict(X_sm)
    rows.append(
        {
            "model": "OLS",
            "rows_used": int(len(work)),
            "r2": float(ols.rsquared),
            "rmse": float(np.sqrt(mean_squared_error(y, ols_pred))),
        }
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    rf_eval = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf_eval.fit(X_train, y_train)
    rf_test = rf_eval.predict(X_test)
    rf_full = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf_full.fit(X, y)
    rf_pred = rf_full.predict(X)
    rows.append(
        {
            "model": "RF",
            "rows_used": int(len(work)),
            "test_r2": float(r2_score(y_test, rf_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, rf_test))),
            "full_r2": float(r2_score(y, rf_pred)),
            "full_rmse": float(np.sqrt(mean_squared_error(y, rf_pred))),
        }
    )

    xgb_eval = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    xgb_eval.fit(X_train, y_train)
    xgb_test = xgb_eval.predict(X_test)
    xgb_full = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    xgb_full.fit(X, y)
    xgb_pred = xgb_full.predict(X)
    rows.append(
        {
            "model": "XGBoost",
            "rows_used": int(len(work)),
            "test_r2": float(r2_score(y_test, xgb_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, xgb_test))),
            "full_r2": float(r2_score(y, xgb_pred)),
            "full_rmse": float(np.sqrt(mean_squared_error(y, xgb_pred))),
        }
    )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = ensure_columns(pd.read_parquet(INPUT))

    dist_rows = []
    model_rows = []

    base_metrics = pd.read_csv(BASE_MODEL_METRICS)

    for thr in THRESHOLDS:
        sub = df[df["Bi"] >= thr].copy() if thr > 0 else df.copy()
        label = f"Bi_ge_{thr:g}"
        if len(sub) == 0:
            dist_rows.append({"threshold": thr, "label": label, "n": 0})
            continue
        stats = dist_stats(sub[RESPONSE])
        stats.update({"threshold": thr, "label": label})
        dist_rows.append(stats)

        if thr == 0.0:
            for _, row in base_metrics.iterrows():
                out = row.to_dict()
                out["threshold"] = thr
                out["label"] = label
                model_rows.append(out)
        elif thr in RERUN_THRESHOLDS:
            for row in run_models(sub):
                row["threshold"] = thr
                row["label"] = label
                model_rows.append(row)

    dist_df = pd.DataFrame(dist_rows)
    model_df = pd.DataFrame(model_rows)
    dist_df.to_csv(OUT_DIR / "bi_threshold_distribution_summary.csv", index=False)
    model_df.to_csv(OUT_DIR / "bi_threshold_model_summary.csv", index=False)

    report_lines = [
        "# Bi Threshold Sensitivity",
        "",
        f"- Input: `{INPUT}`",
        f"- Baseline models (`Bi >= 0`) reused from: `{BASE_MODEL_METRICS}`",
        "",
        "## Distribution summary",
        dist_df.to_csv(index=False),
        "",
        "## Model summary",
        model_df.to_csv(index=False),
        "",
        "## Notes",
        "- Thresholds `0.005` and `0.01` produce zero rows in the current WesternUS table.",
        "- Threshold `0.001` leaves very few rows, so its model metrics are high-variance and should be treated cautiously.",
    ]
    (OUT_DIR / "bi_threshold_sensitivity_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
