import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


ROOT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "resistance_model_execution_near_t0_aggregated"
    / "MGWR_ready_table_near_t0_aggregated.parquet"
)
OUT = ROOT / "resistance_full_saf_models_2026-03-30"

ID_COLS = ["pixel_id", "row", "col", "x", "y", "t0_year"]
RESPONSE = "Resistance"

BASE_PREDICTORS = [
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
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_eto_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_tmmx_std_pre_z",
]


def build_full_saf(df: pd.DataFrame):
    saf = pd.to_numeric(df["FS_EVT_t0agg_SAF_code"], errors="coerce").astype("Int64")
    codes = sorted([int(x) for x in saf.dropna().unique().tolist()])
    created = []
    out = df.copy()
    for code in codes:
        col = f"FS_EVT_t0agg_SAF_{code}"
        out[col] = (saf == code).fillna(False).astype(int)
        created.append(col)
    return out, codes, created


def run_ols(df: pd.DataFrame, predictors: list[str]):
    work = df[ID_COLS + [RESPONSE] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = sm.add_constant(work[predictors], has_constant="add")
    y = work[RESPONSE]
    fit = sm.OLS(y, X).fit()

    metrics = {
        "model_type": "OLS",
        "rows_used": int(len(work)),
        "predictor_count": int(len(predictors)),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
        "rmse_train": float(np.sqrt(np.mean((y - fit.predict(X)) ** 2))),
    }
    coef = pd.DataFrame({"term": fit.params.index, "coef": fit.params.values, "p_value": fit.pvalues.values})
    coef.to_csv(OUT / "ols_full_saf_coefficients.csv", index=False)
    (OUT / "ols_full_saf_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def run_rf(df: pd.DataFrame, predictors: list[str]):
    work = df[ID_COLS + [RESPONSE] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[predictors]
    y = work[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "model_type": "RF",
        "rows_used": int(len(work)),
        "predictor_count": int(len(predictors)),
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    pd.DataFrame({"predictor": predictors, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(OUT / "rf_full_saf_feature_importance.csv", index=False)
    (OUT / "rf_full_saf_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def run_xgb(df: pd.DataFrame, predictors: list[str]):
    work = df[ID_COLS + [RESPONSE] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[predictors]
    y = work[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = model.get_booster().get_score(importance_type="gain")
    pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in score.items()]).sort_values(
        "importance_gain", ascending=False
    ).to_csv(OUT / "xgb_full_saf_feature_importance.csv", index=False)
    metrics = {
        "model_type": "XGBoost",
        "rows_used": int(len(work)),
        "predictor_count": int(len(predictors)),
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    (OUT / "xgb_full_saf_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT)
    df, saf_codes, saf_cols = build_full_saf(df)
    predictors = BASE_PREDICTORS + saf_cols

    metadata = {
        "input_file": str(INPUT),
        "rows_total": int(len(df)),
        "response": RESPONSE,
        "base_predictors": BASE_PREDICTORS,
        "full_saf_codes": saf_codes,
        "full_saf_indicator_columns": saf_cols,
        "notes": [
            "All observed non-null SAF codes were expanded into indicator columns.",
            "Non-SAF and missing SAF pixels remain in the dataset and act as the reference condition in OLS.",
            "This is the 'full SAF' version requested by the user; it is fuller than saf_top5.",
        ],
    }
    (OUT / "full_saf_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    run_ols(df, predictors)
    run_rf(df, predictors)
    run_xgb(df, predictors)


if __name__ == "__main__":
    main()
