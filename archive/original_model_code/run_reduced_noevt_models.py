#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBRegressor


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "westernus_tcc_build_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc.parquet"
)
OUT_DIR = ROOT / "reduced_noevt_models_legacy_nlcd_20260411"

REDUCED_TABLE = OUT_DIR / "reduced_model_table.parquet"
REDUCED_VIF = OUT_DIR / "reduced_model_vif.csv"
MODEL_METRICS = OUT_DIR / "reduced_model_metrics.csv"
MORAN_METRICS = OUT_DIR / "reduced_model_residual_moran.csv"
RUN_REPORT = OUT_DIR / "reduced_model_run_report.md"

ID_COLS = ["pixel_id", "row", "col", "x", "y", "t0_year"]
RESPONSE = "Resistance"
K_MORAN = 8
RANDOM_STATE = 42

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


def compute_vif(work: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    X = work[predictors].astype(float)
    rows = []
    for i, col in enumerate(X.columns):
        rows.append({"variable": col, "vif": float(variance_inflation_factor(X.values, i))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def compute_moran(df: pd.DataFrame, residuals: np.ndarray, k: int = K_MORAN) -> dict:
    w = KNN.from_array(df[["x", "y"]].to_numpy(), k=k)
    w.transform = "R"
    moran = Moran(residuals.astype(float), w, permutations=0)
    return {
        "k": int(k),
        "n_obs": int(len(df)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def run_ols(work: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X = sm.add_constant(work[PREDICTORS], has_constant="add")
    y = work[RESPONSE]
    fit = sm.OLS(y, X).fit()
    pred = fit.predict(X)
    resid = y - pred
    metrics = {
        "model": "OLS",
        "rows_used": int(len(work)),
        "r2": float(fit.rsquared),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "adj_r2": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }
    coef = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "p_value": fit.pvalues.values})
    coef.to_csv(OUT_DIR / "reduced_model_ols_coefficients.csv", index=False)
    (OUT_DIR / "reduced_model_ols_summary.txt").write_text(fit.summary().as_text(), encoding="utf-8")
    residual_df = work[ID_COLS + [RESPONSE]].copy()
    residual_df["prediction"] = pred
    residual_df["residual"] = resid
    residual_df.to_parquet(OUT_DIR / "reduced_model_ols_residuals.parquet", index=False)
    return metrics, residual_df


def run_rf(work: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X = work[PREDICTORS]
    y = work[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    eval_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    eval_model.fit(X_train, y_train)
    pred_test = eval_model.predict(X_test)

    full_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(X, y)
    pred_full = full_model.predict(X)
    metrics = {
        "model": "RF",
        "rows_used": int(len(work)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_r2": float(r2_score(y, pred_full)),
        "full_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
    }
    pd.DataFrame({"variable": PREDICTORS, "importance": full_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(OUT_DIR / "reduced_model_rf_feature_importance.csv", index=False)
    residual_df = work[ID_COLS + [RESPONSE]].copy()
    residual_df["prediction"] = pred_full
    residual_df["residual"] = y - pred_full
    residual_df.to_parquet(OUT_DIR / "reduced_model_rf_residuals.parquet", index=False)
    return metrics, residual_df


def run_xgb(work: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X = work[PREDICTORS]
    y = work[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    eval_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    eval_model.fit(X_train, y_train)
    pred_test = eval_model.predict(X_test)

    full_model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=8,
    )
    full_model.fit(X, y)
    pred_full = full_model.predict(X)
    metrics = {
        "model": "XGBoost",
        "rows_used": int(len(work)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_r2": float(r2_score(y, pred_full)),
        "full_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
    }
    booster = full_model.get_booster()
    score = booster.get_score(importance_type="gain")
    pd.DataFrame(
        [{"variable": k, "importance_gain": float(v)} for k, v in score.items()]
    ).sort_values("importance_gain", ascending=False).to_csv(
        OUT_DIR / "reduced_model_xgb_feature_importance.csv", index=False
    )
    residual_df = work[ID_COLS + [RESPONSE]].copy()
    residual_df["prediction"] = pred_full
    residual_df["residual"] = y - pred_full
    residual_df.to_parquet(OUT_DIR / "reduced_model_xgb_residuals.parquet", index=False)
    return metrics, residual_df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT)
    df = ensure_columns(df)

    keep_cols = ID_COLS + [RESPONSE] + PREDICTORS
    reduced = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    reduced.to_parquet(REDUCED_TABLE, index=False)
    reduced.head(1000).to_csv(OUT_DIR / "reduced_model_table_sample.csv", index=False)

    vif_df = compute_vif(reduced, PREDICTORS)
    vif_df.to_csv(REDUCED_VIF, index=False)

    ols_metrics, ols_resid = run_ols(reduced)
    rf_metrics, rf_resid = run_rf(reduced)
    xgb_metrics, xgb_resid = run_xgb(reduced)

    moran_rows = []
    for model_name, resid_df in [("OLS", ols_resid), ("RF", rf_resid), ("XGBoost", xgb_resid)]:
        moran = compute_moran(resid_df, resid_df["residual"].to_numpy(), K_MORAN)
        moran["model"] = model_name
        moran_rows.append(moran)
    moran_df = pd.DataFrame(moran_rows)[["model", "k", "n_obs", "moran_i", "z_norm", "p_norm"]]
    moran_df.to_csv(MORAN_METRICS, index=False)

    metrics_df = pd.DataFrame([ols_metrics, rf_metrics, xgb_metrics])
    metrics_df.to_csv(MODEL_METRICS, index=False)

    def df_lines(df: pd.DataFrame) -> list[str]:
        lines = [",".join(df.columns.astype(str))]
        for row in df.itertuples(index=False, name=None):
            lines.append(",".join(str(x) for x in row))
        return lines

    report = "\n".join(
        [
            "# Reduced No-EVT Model Run",
            "",
            f"- Input: `{INPUT}`",
            f"- Output reduced table: `{REDUCED_TABLE}`",
            f"- Rows used after complete-case filtering: {len(reduced)}",
            f"- Predictors: {', '.join(PREDICTORS)}",
            "",
            "## VIF",
            *df_lines(vif_df),
            "",
            "## Model metrics",
            *df_lines(metrics_df),
            "",
            "## Residual Moran's I",
            *df_lines(moran_df),
            "",
            "## Uncertainty",
            "- `HUM_popdens_win10km_log_z` and `HUM_viirs_near_t0_log_z` were generated here with `log1p` then z-score because the exact transform implementation was not explicitly recovered from a prior table-building script; the variable names themselves are verified from the existing modeling scripts.",
        ]
    )
    RUN_REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(
        {
            "reduced_rows": int(len(reduced)),
            "reduced_table": str(REDUCED_TABLE),
            "vif_csv": str(REDUCED_VIF),
            "model_metrics_csv": str(MODEL_METRICS),
            "moran_csv": str(MORAN_METRICS),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
