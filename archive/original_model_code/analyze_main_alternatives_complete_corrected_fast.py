#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
COMPLETE_RF_METRICS = (
    ROOT
    / "westernus_rf_noevt_plusxy_complete_corrected_20260411"
    / "westernus_rf_noevt_plusxy_complete_corrected_metrics.json"
)
SPATIAL_XY_SUMMARY = (
    ROOT
    / "spatial_block_validation_complete_rf_fast_20260411"
    / "spatial_block_validation_100km_summary.json"
)

OUT_DIR = ROOT / "main_alternatives_complete_corrected_fast_20260411"
OUT_NOXY_JSON = OUT_DIR / "spatial_block_validation_no_xy_100km_summary_fast.json"
OUT_DIST_CSV = OUT_DIR / "predictor_distribution_audit_fast.csv"
OUT_RESP_CSV = OUT_DIR / "main_other_response_rf_metrics_fast.csv"
OUT_REPORT = OUT_DIR / "main_alternatives_complete_corrected_fast_report.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
BLOCK_KM = 100
RESPONSE = "Resistance"

COMPLETE_PREDICTORS = [
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
NOXY_PREDICTORS = [c for c in COMPLETE_PREDICTORS if c not in {"x", "y"}]
MAIN_OTHER_RESPONSES = ["T80", "IRI_good_pow2", "STAB_good_pow2"]

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

PREDICTOR_INFO = [
    ("TS_elev_m_z", "TS_elev_m", "z"),
    ("TS_slope_deg_z", "TS_slope_deg", "z"),
    ("TS_northness_z", "TS_northness", "z"),
    ("TS_eastness_z", "TS_eastness", "z"),
    ("TS_twi_z", "TS_twi", "z"),
    ("TS_roughness_z", "TS_roughness", "z"),
    ("TS_SOC_0_30cm_z", "TS_SOC_0_30cm", "z"),
    ("FS_TCC_t0_z", "FS_TCC_t0", "z"),
    ("FS_CBH_t0agg_z", "FS_CBH_t0agg", "z"),
    ("HUM_popdens_win10km_log_z", "HUM_popdens_win10km", "log1p_z"),
    ("HUM_roaddens_r5km_z", "HUM_roaddens_r5km", "z"),
    ("HUM_traildens_r10km_z", "HUM_traildens_r10km", "z"),
    ("HUM_imperv_near_t0_z", "HUM_imperv_near_t0", "z"),
    ("HUM_viirs_near_t0_log_z", "HUM_viirs_near_t0", "log1p_z"),
    ("CLIM_pr_sum_pre_z", "CLIM_pr_sum_pre", "z"),
    ("CLIM_eto_sum_pre_z", "CLIM_eto_sum_pre", "z"),
    ("CLIM_tmmn_mean_pre_z", "CLIM_tmmn_mean_pre", "z"),
    ("CLIM_hot_days_35C_pre_z", "CLIM_hot_days_35C_pre", "z"),
    ("CLIM_aridity_pre_z", "CLIM_aridity_pre", "z"),
    ("CLIM_tmmx_std_pre_z", "CLIM_tmmx_std_pre", "z"),
    ("x", "x", "xy"),
    ("y", "y", "xy"),
]


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


def fit_rf_predict(X_train, y_train, X_test) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def summarize_series(series: pd.Series) -> dict[str, float]:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    valid = vals[np.isfinite(vals)]
    return {
        "raw_skew": float(pd.Series(valid).skew()),
        "raw_zero_frac": float(np.mean(valid == 0)),
        "raw_min": float(valid.min()),
        "raw_p50": float(np.percentile(valid, 50)),
        "raw_p99": float(np.percentile(valid, 99)),
        "raw_max": float(valid.max()),
    }


def recommend_transform(raw_stats: dict[str, float], transform_type: str) -> str:
    skew = raw_stats["raw_skew"]
    min_v = raw_stats["raw_min"]
    zero_frac = raw_stats["raw_zero_frac"]
    max_v = raw_stats["raw_max"]
    if transform_type == "log1p_z":
        return "current_log1p_z_looks_appropriate"
    if transform_type == "xy":
        return "keep_as_spatial_terms"
    if min_v >= 0 and skew >= 2:
        if max_v <= 100:
            return "consider_log1p_or_sqrt_sensitivity"
        if zero_frac >= 0.05:
            return "strong_log1p_candidate"
        return "consider_log_or_sqrt_sensitivity"
    if min_v >= 0 and 1 <= skew < 2:
        return "moderate_right_skew_z_only_may_be_ok_for_rf"
    if abs(skew) < 1:
        return "current_scale_looks_ok"
    return "current_scale_probably_ok_but_monitor"


def run_random_rf(df: pd.DataFrame, response: str, predictors: list[str]) -> dict:
    work = df[[response] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[predictors]
    y = work[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pred = fit_rf_predict(X_train, y_train, X_test)
    return {
        "response": response,
        "rows_used": int(len(work)),
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def run_group_holdout_rf(df: pd.DataFrame, response: str, predictors: list[str], block_km: int) -> dict:
    keep_cols = list(dict.fromkeys([response] + predictors + ["x", "y"]))
    work = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    block_m = block_km * 1000.0
    groups = (
        np.floor(work["x"] / block_m).astype(int).astype(str)
        + "_"
        + np.floor(work["y"] / block_m).astype(int).astype(str)
    )
    X = work[predictors]
    y = work[response]
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    pred = fit_rf_predict(X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx])
    return {
        "response": response,
        "rows_used": int(len(work)),
        "n_groups": int(pd.Series(groups).nunique()),
        "test_r2": float(r2_score(y.iloc[test_idx], pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y.iloc[test_idx], pred))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = ensure_columns(pd.read_parquet(INPUT).copy())

    complete_rf_metrics = json.loads(COMPLETE_RF_METRICS.read_text())
    spatial_xy_summary = json.loads(SPATIAL_XY_SUMMARY.read_text())

    noxy_random = run_random_rf(df, RESPONSE, NOXY_PREDICTORS)
    noxy_block = run_group_holdout_rf(df, RESPONSE, NOXY_PREDICTORS, BLOCK_KM)
    OUT_NOXY_JSON.write_text(
        json.dumps(
            {
                "n_estimators": N_ESTIMATORS,
                "random_split_no_xy": noxy_random,
                "group_holdout_100km_no_xy": noxy_block,
                "reference_random_with_xy_r2": float(complete_rf_metrics["test_r2"]),
                "reference_group_holdout_with_xy_r2": float(
                    spatial_xy_summary["group_holdout_20pct"]["r2"]
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    dist_rows = []
    for model_col, raw_col, transform_type in PREDICTOR_INFO:
        row = summarize_series(df[raw_col])
        row["model_predictor"] = model_col
        row["raw_source"] = raw_col
        row["current_transform"] = transform_type
        row["recommendation"] = recommend_transform(row, transform_type)
        dist_rows.append(row)
    dist_df = pd.DataFrame(dist_rows).sort_values("raw_skew", ascending=False).reset_index(drop=True)
    dist_df.to_csv(OUT_DIST_CSV, index=False)

    resp_rows = [run_random_rf(df, RESPONSE, COMPLETE_PREDICTORS)]
    pd.DataFrame(resp_rows).to_csv(OUT_RESP_CSV, index=False)
    for response in MAIN_OTHER_RESPONSES:
        resp_rows.append(run_random_rf(df, response, COMPLETE_PREDICTORS))
        pd.DataFrame(resp_rows).sort_values("test_r2", ascending=False).to_csv(OUT_RESP_CSV, index=False)
    resp_df = pd.DataFrame(resp_rows).sort_values("test_r2", ascending=False).reset_index(drop=True)

    lines = [
        "# Main Alternatives Fast Screen",
        "",
        f"- This is a fast screening run using `RF` with `n_estimators={N_ESTIMATORS}`.",
        "",
        "## No-XY Spatial Validation",
        f"- Resistance random split without `x/y`: R2=`{noxy_random['test_r2']:.6f}`",
        f"- Resistance 100 km block holdout without `x/y`: R2=`{noxy_block['test_r2']:.6f}`",
        f"- Reference random split with `x/y` (500 trees): R2=`{float(complete_rf_metrics['test_r2']):.6f}`",
        f"- Reference 100 km block holdout with `x/y` (500 trees): R2=`{float(spatial_xy_summary['group_holdout_20pct']['r2']):.6f}`",
        "",
        "## Most Skewed Predictors",
    ]
    for _, row in dist_df.head(10).iterrows():
        lines.append(
            f"- `{row['model_predictor']}`: raw_skew=`{row['raw_skew']:.3f}`, recommendation=`{row['recommendation']}`"
        )
    lines.extend(["", "## Main Response RF Screen"])
    for _, row in resp_df.iterrows():
        lines.append(
            f"- `{row['response']}`: test_R2=`{row['test_r2']:.6f}`, test_RMSE=`{row['test_rmse']:.6f}`, rows=`{int(row['rows_used'])}`"
        )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "noxy_json": str(OUT_NOXY_JSON),
                "distribution_csv": str(OUT_DIST_CSV),
                "response_csv": str(OUT_RESP_CSV),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
