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
OUT_DIR = ROOT / "bi_threshold_sensitivity_fast_20260410"
BASE_MODEL_METRICS = ROOT / "reduced_noevt_models_20260410" / "reduced_model_metrics.csv"

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


def run_models(work: pd.DataFrame) -> list[dict]:
    rows = []
    if len(work) < 50:
        return rows

    X = work[PREDICTORS]
    y = work[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train_sm = sm.add_constant(X_train, has_constant="add")
    X_test_sm = sm.add_constant(X_test, has_constant="add")
    ols = sm.OLS(y_train, X_train_sm).fit()
    ols_test = ols.predict(X_test_sm)
    rows.append(
        {
            "model": "OLS",
            "rows_used": int(len(work)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_r2": float(r2_score(y_test, ols_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, ols_test))),
        }
    )

    rf = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_test = rf.predict(X_test)
    rows.append(
        {
            "model": "RF",
            "rows_used": int(len(work)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_r2": float(r2_score(y_test, rf_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, rf_test))),
        }
    )

    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    xgb.fit(X_train, y_train)
    xgb_test = xgb.predict(X_test)
    rows.append(
        {
            "model": "XGBoost",
            "rows_used": int(len(work)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_r2": float(r2_score(y_test, xgb_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, xgb_test))),
        }
    )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = ensure_columns(pd.read_parquet(INPUT))
    base_metrics = pd.read_csv(BASE_MODEL_METRICS)

    dist_rows = []
    model_rows = []

    for thr in THRESHOLDS:
        label = f"Bi_ge_{thr:g}"
        sub = df[df["Bi"] >= thr].copy() if thr > 0 else df.copy()
        if len(sub) == 0:
            dist_rows.append({"threshold": thr, "label": label, "n": 0})
            print(f"{label}: empty")
            continue
        work = sub[[RESPONSE] + PREDICTORS].replace([np.inf, -np.inf], np.nan).dropna().copy()
        stats = dist_stats(work[RESPONSE])
        stats.update({"threshold": thr, "label": label, "rows_used": int(len(work))})
        dist_rows.append(stats)
        print(f"{label}: rows={len(work)}")

        if thr == 0.0:
            for _, row in base_metrics.iterrows():
                out = row.to_dict()
                out["threshold"] = thr
                out["label"] = label
                model_rows.append(out)
        else:
            for row in run_models(work):
                row["threshold"] = thr
                row["label"] = label
                model_rows.append(row)

        pd.DataFrame(dist_rows).to_csv(OUT_DIR / "bi_threshold_distribution_summary.csv", index=False)
        pd.DataFrame(model_rows).to_csv(OUT_DIR / "bi_threshold_model_summary.csv", index=False)

    dist_df = pd.DataFrame(dist_rows)
    model_df = pd.DataFrame(model_rows)
    report_lines = [
        "# Bi Threshold Sensitivity Fast",
        "",
        f"- Input: `{INPUT}`",
        "- Models use the same random 80/20 split style as the prior reduced-model run.",
        "- This sensitivity run focuses on distribution change and test-set behavior after filtering low-Bi pixels.",
        "",
        "## Distribution summary",
        dist_df.to_csv(index=False),
        "",
        "## Model summary",
        model_df.to_csv(index=False),
    ]
    (OUT_DIR / "bi_threshold_sensitivity_fast_report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
