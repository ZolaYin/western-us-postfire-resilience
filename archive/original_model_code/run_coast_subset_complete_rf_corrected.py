#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
COAST_BOUNDARY = ROOT.parent / "WUS_states_boundary.shp"
FULL_WUS_COMPLETE_METRICS = (
    ROOT
    / "westernus_rf_noevt_plusxy_complete_corrected_20260411"
    / "westernus_rf_noevt_plusxy_complete_corrected_metrics.json"
)
OUT_DIR = ROOT / "coast_subset_complete_rf_corrected_20260411"
OUT_METRICS = OUT_DIR / "coast_subset_complete_rf_corrected_metrics.json"
OUT_SAMPLE = OUT_DIR / "coast_subset_complete_rf_corrected_sample.csv"
OUT_REPORT = OUT_DIR / "coast_subset_complete_rf_corrected_report.md"

RESPONSE = "Resistance"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500
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


def load_coast_polygon() -> gpd.GeoSeries:
    coast = gpd.read_file(COAST_BOUNDARY).to_crs("EPSG:5070")
    return coast.geometry.union_all()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT).copy()
    df = ensure_columns(df)
    coast_poly = load_coast_polygon()

    geom = gpd.GeoSeries([Point(xy) for xy in zip(df["x"], df["y"])], crs="EPSG:5070")
    in_coast = geom.within(coast_poly)
    coast = df.loc[in_coast, ["pixel_id", "row", "col", "x", "y", RESPONSE] + PREDICTORS].copy()
    coast = coast.replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = coast[PREDICTORS]
    y = coast[RESPONSE]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    eval_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    eval_model.fit(X_train, y_train)
    pred_test = eval_model.predict(X_test)

    full_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    full_model.fit(X, y)
    full_pred = full_model.predict(X)

    full_wus = json.loads(FULL_WUS_COMPLETE_METRICS.read_text())
    metrics = {
        "model": "RF",
        "variant": "complete_corrected_coast_subset",
        "input_table": str(INPUT),
        "coast_boundary": str(COAST_BOUNDARY),
        "rows_in_full_table": int(len(df)),
        "rows_in_coast_subset": int(len(coast)),
        "predictors": PREDICTORS,
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_r2": float(r2_score(y, full_pred)),
        "full_rmse": float(np.sqrt(mean_squared_error(y, full_pred))),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "full_wus_complete_rf_test_r2": float(full_wus["test_r2"]),
        "full_wus_complete_rf_test_rmse": float(full_wus["test_rmse"]),
        "old_coast_rf_plusxy_doc_test_r2": 0.695254,
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    coast.head(1000).to_csv(OUT_SAMPLE, index=False)

    lines = [
        "# Coast Subset Complete RF Check",
        "",
        f"- Input table: `{INPUT}`",
        f"- Coast boundary: `{COAST_BOUNDARY}`",
        f"- Rows in corrected full WUS table: `{len(df)}`",
        f"- Rows in corrected Coast subset: `{len(coast)}`",
        f"- RF test R2 on corrected Coast subset: `{metrics['test_r2']:.6f}`",
        f"- RF test RMSE on corrected Coast subset: `{metrics['test_rmse']:.6f}`",
        f"- RF test R2 on corrected full WUS complete model: `{metrics['full_wus_complete_rf_test_r2']:.6f}`",
        f"- Old Coast doc RF_plusXY reference: `{metrics['old_coast_rf_plusxy_doc_test_r2']:.6f}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
