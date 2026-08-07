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
OUT = ROOT / "resistance_official_near_t0_system_2026-03-30"

ID_COLS = ["pixel_id", "row", "col", "x", "y", "t0_year"]
RESPONSE = "Resistance"

STATIC_TOPO_SOIL = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
]
NEAR_T0_NON_EVT = [
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z",
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
]
CLIMATE_PRE = [
    "CLIM_pr_sum_pre_z",
    "CLIM_eto_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_tmmx_std_pre_z",
]
BASE_PREDICTORS = STATIC_TOPO_SOIL + NEAR_T0_NON_EVT + CLIMATE_PRE

TOP_SAF_CODES = [211, 210, 243, 234, 230]
TOP_EVT_CODES = [7028, 7043, 7037, 7027, 3028]

EVT_VARIANTS = {
    "proxy": ["FS_EVT_t0agg_resistance_proxy"],
    "grouped": [
        "FS_EVT_t0agg_group_is_shrub",
        "FS_EVT_t0agg_group_is_deciduous",
        "FS_EVT_t0agg_group_is_mixed",
        "FS_EVT_t0agg_group_is_conifer",
    ],
    "saf_top5": [
        "FS_EVT_t0agg_is_SAF",
        *[f"FS_EVT_t0agg_SAF_{code}" for code in TOP_SAF_CODES],
    ],
    "evtcode_top5": [f"FS_EVT_t0agg_code_{code}" for code in TOP_EVT_CODES],
}


def ensure_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    saf = pd.to_numeric(df["FS_EVT_t0agg_SAF_code"], errors="coerce").astype("Int64")
    evt = pd.to_numeric(df["FS_EVT_t0agg_code"], errors="coerce").astype("Int64")
    for code in TOP_SAF_CODES:
        df[f"FS_EVT_t0agg_SAF_{code}"] = (saf == code).fillna(False).astype(int)
    for code in TOP_EVT_CODES:
        df[f"FS_EVT_t0agg_code_{code}"] = (evt == code).fillna(False).astype(int)
    return df


def official_variable_text() -> str:
    keep = {
        "static_topo_soil": STATIC_TOPO_SOIL,
        "near_t0_non_evt": NEAR_T0_NON_EVT,
        "climate_pre": CLIMATE_PRE,
        "evt_variants": EVT_VARIANTS,
    }
    drop = {
        "static_versions_formally_deprecated": [
            "FS_EVT_resistance_proxy",
            "FS_EVT_resistance_proxy_z",
            "FS_CBH_1km",
            "FS_CBH_1km_z",
            "FS_EVT_regeneration_proxy",
            "FS_EVT_regeneration_proxy_z",
            "HUM_roaddens_r5km_z",
            "HUM_traildens_r10km_z",
            "HUM_roaddens_r10km_z",
            "all static 2022-only EVT/SAF snapshots for official Resistance inference",
        ],
        "reason": [
            "topo/soil are the only variables retained as static",
            "climate remains pre/post-style and is not replaced by single-year t0 sampling",
            "EVT/SAF and CBH are retained only as temporally aligned near-t0 versions",
            "road/trail remain excluded because no near-t0 source stack is currently available",
        ],
    }
    lines = ["Official Resistance Variable System", "", "Retained variables:"]
    for group, cols in keep.items():
        lines.append(f"- {group}:")
        for col in cols:
            lines.append(f"  - {col}")
    lines.extend(["", "Formally deprecated variables:", ""])
    for item in drop["static_versions_formally_deprecated"]:
        lines.append(f"- {item}")
    lines.extend(["", "Rationale:"])
    for item in drop["reason"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def run_ols(work: pd.DataFrame, predictors: list[str], out_dir: Path, variant: str) -> dict:
    model_df = work[ID_COLS + [RESPONSE] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = sm.add_constant(model_df[predictors], has_constant="add")
    y = model_df[RESPONSE]
    fit = sm.OLS(y, X).fit()

    metrics = {
        "model_type": "OLS",
        "evt_variant": variant,
        "rows_used": int(len(model_df)),
        "predictor_count": int(len(predictors)),
        "r2": float(fit.rsquared),
        "adj_r2": float(fit.rsquared_adj),
        "rmse": float(np.sqrt(np.mean((y - fit.predict(X)) ** 2))),
        "aic": float(fit.aic),
        "bic": float(fit.bic),
    }

    coef = pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.values,
        "p_value": fit.pvalues.values,
    })
    coef.to_csv(out_dir / f"ols_{variant}_coefficients.csv", index=False)

    residual_df = model_df[ID_COLS + [RESPONSE]].copy()
    residual_df["prediction"] = fit.predict(X)
    residual_df["residual"] = residual_df[RESPONSE] - residual_df["prediction"]
    residual_df.to_parquet(out_dir / f"ols_{variant}_residuals.parquet", index=False)

    (out_dir / f"ols_{variant}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def run_rf(work: pd.DataFrame, predictors: list[str], out_dir: Path, variant: str) -> dict:
    model_df = work[ID_COLS + [RESPONSE] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = model_df[predictors]
    y = model_df[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    eval_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    eval_model.fit(X_train, y_train)
    pred_test = eval_model.predict(X_test)

    full_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    full_model.fit(X, y)
    pred_full = full_model.predict(X)

    metrics = {
        "model_type": "RF",
        "evt_variant": variant,
        "rows_used": int(len(model_df)),
        "predictor_count": int(len(predictors)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    }

    imp = pd.DataFrame({"predictor": predictors, "importance": full_model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    imp.to_csv(out_dir / f"rf_{variant}_feature_importance.csv", index=False)

    residual_df = model_df[ID_COLS + [RESPONSE]].copy()
    residual_df["prediction"] = pred_full
    residual_df["residual"] = residual_df[RESPONSE] - residual_df["prediction"]
    residual_df.to_parquet(out_dir / f"rf_{variant}_residuals.parquet", index=False)

    (out_dir / f"rf_{variant}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def run_xgb(work: pd.DataFrame, predictors: list[str], out_dir: Path, variant: str) -> dict:
    model_df = work[ID_COLS + [RESPONSE] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = model_df[predictors]
    y = model_df[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    eval_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    eval_model.fit(X_train, y_train)
    pred_test = eval_model.predict(X_test)

    full_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    full_model.fit(X, y)
    pred_full = full_model.predict(X)

    score = full_model.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame([{"predictor": k, "importance_gain": v} for k, v in score.items()]).sort_values(
        "importance_gain", ascending=False
    )
    imp.to_csv(out_dir / f"xgb_{variant}_feature_importance.csv", index=False)

    residual_df = model_df[ID_COLS + [RESPONSE]].copy()
    residual_df["prediction"] = pred_full
    residual_df["residual"] = residual_df[RESPONSE] - residual_df["prediction"]
    residual_df.to_parquet(out_dir / f"xgb_{variant}_residuals.parquet", index=False)

    metrics = {
        "model_type": "XGBoost",
        "evt_variant": variant,
        "rows_used": int(len(model_df)),
        "predictor_count": int(len(predictors)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
    }
    (out_dir / f"xgb_{variant}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT)
    df = ensure_variant_columns(df)

    (OUT / "official_retained_variable_system.txt").write_text(official_variable_text())

    rows = []
    for variant, evt_cols in EVT_VARIANTS.items():
        model_dir = OUT / variant
        model_dir.mkdir(exist_ok=True)
        predictors = BASE_PREDICTORS + evt_cols

        for model_type, runner in [("OLS", run_ols), ("RF", run_rf), ("XGBoost", run_xgb)]:
            metrics = runner(df, predictors, model_dir, variant)
            metrics["evt_columns"] = ",".join(evt_cols)
            metrics["base_predictors"] = ",".join(BASE_PREDICTORS)
            rows.append(metrics)

    pd.DataFrame(rows).to_csv(OUT / "official_model_comparison.csv", index=False)


if __name__ == "__main__":
    main()
