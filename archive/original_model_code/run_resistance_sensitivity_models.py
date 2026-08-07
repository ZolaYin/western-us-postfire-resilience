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
from xgboost import XGBRegressor


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_tcc_build_20260410" / "westernus_current_candidate_table_plus_cbh_tcc.parquet"
OUT_DIR = ROOT / "resistance_sensitivity_models_20260410"

ID_COLS = ["pixel_id", "row", "col", "x", "y", "t0_year", "Bi"]
BASE_RESPONSE = "Resistance"
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

VARIANTS = [
    {"variant": "raw_all", "bi_threshold": 0.0, "transform": "raw"},
    {"variant": "log1p_all", "bi_threshold": 0.0, "transform": "log1p"},
    {"variant": "raw_Bi_ge_0.001", "bi_threshold": 0.001, "transform": "raw"},
    {"variant": "log1p_Bi_ge_0.001", "bi_threshold": 0.001, "transform": "log1p"},
]


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


def dist_stats(series: pd.Series) -> dict:
    return {
        "n": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
        "max": float(series.max()),
        "skew": float(series.skew()),
        "kurt": float(series.kurt()),
    }


def compute_moran(df: pd.DataFrame, residuals: np.ndarray) -> dict:
    w = KNN.from_array(df[["x", "y"]].to_numpy(), k=K_MORAN)
    w.transform = "R"
    moran = Moran(residuals.astype(float), w, permutations=0)
    return {
        "k": int(K_MORAN),
        "n_obs": int(len(df)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def prepare_variant(df: pd.DataFrame, bi_threshold: float, transform: str) -> pd.DataFrame:
    sub = df[df["Bi"] >= bi_threshold].copy() if bi_threshold > 0 else df.copy()
    keep_cols = ID_COLS + [BASE_RESPONSE] + PREDICTORS
    work = sub[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if transform == "raw":
        work["response_model"] = work[BASE_RESPONSE].astype(float)
    elif transform == "log1p":
        work["response_model"] = np.log1p(work[BASE_RESPONSE].astype(float))
    else:
        raise ValueError(transform)
    return work


def run_ols(work: pd.DataFrame) -> tuple[dict, np.ndarray]:
    X = work[PREDICTORS]
    y = work["response_model"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    fit = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()
    pred_test = fit.predict(sm.add_constant(X_test, has_constant="add"))
    pred_full = fit.predict(sm.add_constant(X, has_constant="add"))
    return (
        {
            "model": "OLS",
            "rows_used": int(len(work)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_r2": float(r2_score(y_test, pred_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "full_r2": float(r2_score(y, pred_full)),
            "full_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
            "aic_train": float(fit.aic),
            "bic_train": float(fit.bic),
        },
        (y - pred_full).to_numpy(),
    )


def run_rf(work: pd.DataFrame) -> tuple[dict, np.ndarray]:
    X = work[PREDICTORS]
    y = work["response_model"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    eval_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    eval_model.fit(X_train, y_train)
    pred_test = eval_model.predict(X_test)
    full_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(X, y)
    pred_full = full_model.predict(X)
    return (
        {
            "model": "RF",
            "rows_used": int(len(work)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_r2": float(r2_score(y_test, pred_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "full_r2": float(r2_score(y, pred_full)),
            "full_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
        },
        (y - pred_full).to_numpy(),
    )


def run_xgb(work: pd.DataFrame) -> tuple[dict, np.ndarray]:
    X = work[PREDICTORS]
    y = work["response_model"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
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
    return (
        {
            "model": "XGBoost",
            "rows_used": int(len(work)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_r2": float(r2_score(y_test, pred_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "full_r2": float(r2_score(y, pred_full)),
            "full_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
        },
        (y - pred_full).to_numpy(),
    )


def df_lines(df: pd.DataFrame) -> list[str]:
    lines = [",".join(df.columns.astype(str))]
    for row in df.itertuples(index=False, name=None):
        lines.append(",".join(str(x) for x in row))
    return lines


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = ensure_columns(pd.read_parquet(INPUT))

    dist_rows = []
    model_rows = []
    moran_rows = []

    for spec in VARIANTS:
        variant = spec["variant"]
        work = prepare_variant(df, spec["bi_threshold"], spec["transform"])
        stats = dist_stats(work["response_model"])
        stats.update(
            {
                "variant": variant,
                "transform": spec["transform"],
                "bi_threshold": spec["bi_threshold"],
                "rows_used": int(len(work)),
                "base_response_median": float(work[BASE_RESPONSE].median()),
                "base_response_p99": float(work[BASE_RESPONSE].quantile(0.99)),
                "base_response_max": float(work[BASE_RESPONSE].max()),
            }
        )
        dist_rows.append(stats)

        variant_path = OUT_DIR / f"{variant}_table.parquet"
        work[ID_COLS + [BASE_RESPONSE, "response_model"] + PREDICTORS].to_parquet(variant_path, index=False)

        for runner in (run_ols, run_rf, run_xgb):
            metrics, residuals = runner(work)
            metrics.update(
                {
                    "variant": variant,
                    "transform": spec["transform"],
                    "bi_threshold": spec["bi_threshold"],
                }
            )
            model_rows.append(metrics)

            moran = compute_moran(work, residuals)
            moran.update(
                {
                    "variant": variant,
                    "model": metrics["model"],
                    "transform": spec["transform"],
                    "bi_threshold": spec["bi_threshold"],
                }
            )
            moran_rows.append(moran)

    dist_df = pd.DataFrame(dist_rows)
    model_df = pd.DataFrame(model_rows)
    moran_df = pd.DataFrame(moran_rows)

    dist_df.to_csv(OUT_DIR / "resistance_sensitivity_distribution.csv", index=False)
    model_df.to_csv(OUT_DIR / "resistance_sensitivity_model_metrics.csv", index=False)
    moran_df.to_csv(OUT_DIR / "resistance_sensitivity_moran.csv", index=False)

    report = "\n".join(
        [
            "# Resistance Sensitivity Modeling",
            "",
            f"- Input: `{INPUT}`",
            f"- Predictors: {', '.join(PREDICTORS)}",
            "- Train/test split uses `train_test_split(test_size=0.2, random_state=42)` for every variant.",
            "- `log1p` variants model `log1p(Resistance)` directly.",
            "- `Bi >= 0.001` is the only verified non-empty threshold among the previously tested larger cutoffs; `0.005` and `0.01` are empty in the current table.",
            "",
            "## Distribution summary",
            *df_lines(dist_df),
            "",
            "## Model metrics",
            *df_lines(model_df),
            "",
            "## Residual Moran's I",
            *df_lines(moran_df),
            "",
            "## Uncertainty",
            "- Cross-variant RMSE values are directly comparable only within the same response scale. Raw-response RMSE and log-response RMSE are on different scales.",
            "- `HUM_popdens_win10km_log_z` and `HUM_viirs_near_t0_log_z` still use the locally reconstructed `log1p` then z-score transformation from the prior reduced-model script.",
        ]
    )
    (OUT_DIR / "resistance_sensitivity_report.md").write_text(report, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(OUT_DIR),
                "distribution_csv": str(OUT_DIR / "resistance_sensitivity_distribution.csv"),
                "metrics_csv": str(OUT_DIR / "resistance_sensitivity_model_metrics.csv"),
                "moran_csv": str(OUT_DIR / "resistance_sensitivity_moran.csv"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
