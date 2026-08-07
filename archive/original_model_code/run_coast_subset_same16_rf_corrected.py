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
INPUT = ROOT / "reduced_noevt_models_legacy_nlcd_20260411" / "reduced_model_table.parquet"
COAST_BOUNDARY = ROOT.parent / "WUS_states_boundary.shp"
FULL_WUS_METRICS = ROOT / "reduced_noevt_models_legacy_nlcd_20260411" / "reduced_model_metrics.csv"
OUT_DIR = ROOT / "coast_subset_same16_rf_corrected_20260411"
OUT_METRICS = OUT_DIR / "coast_subset_same16_rf_metrics.json"
OUT_SAMPLE = OUT_DIR / "coast_subset_same16_rf_sample.csv"
OUT_REPORT = OUT_DIR / "coast_subset_same16_rf_report.md"

RESPONSE = "Resistance"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500
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


def load_coast_polygon() -> gpd.GeoSeries:
    coast = gpd.read_file(COAST_BOUNDARY).to_crs("EPSG:5070")
    return coast.geometry.union_all()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT).copy()
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
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    eval_model.fit(X_train, y_train)
    pred_test = eval_model.predict(X_test)

    full_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    full_model.fit(X, y)
    full_pred = full_model.predict(X)

    full_wus = pd.read_csv(FULL_WUS_METRICS)
    full_wus_rf = full_wus.loc[full_wus["model"] == "RF"].iloc[0].to_dict()

    metrics = {
        "model": "RF",
        "variant": "same16_corrected_coast_subset",
        "input_table": str(INPUT),
        "coast_boundary": str(COAST_BOUNDARY),
        "rows_in_full_reduced_table": int(len(df)),
        "rows_in_coast_subset": int(len(coast)),
        "predictors": PREDICTORS,
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_r2": float(r2_score(y, full_pred)),
        "full_rmse": float(np.sqrt(mean_squared_error(y, full_pred))),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "full_wus_same16_rf_test_r2": float(full_wus_rf["test_r2"]),
        "full_wus_same16_rf_test_rmse": float(full_wus_rf["test_rmse"]),
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    sample = coast.head(1000).copy()
    sample.to_csv(OUT_SAMPLE, index=False)

    lines = [
        "# Coast Subset Same-16 RF Check",
        "",
        f"- Input reduced table: `{INPUT}`",
        f"- Coast boundary: `{COAST_BOUNDARY}`",
        f"- Rows in corrected full WUS reduced table: `{len(df)}`",
        f"- Rows in corrected Coast subset: `{len(coast)}`",
        f"- RF test R2 on Coast subset: `{metrics['test_r2']:.6f}`",
        f"- RF test RMSE on Coast subset: `{metrics['test_rmse']:.6f}`",
        f"- RF test R2 on corrected full WUS with same 16 predictors: `{metrics['full_wus_same16_rf_test_r2']:.6f}`",
        f"- RF test RMSE on corrected full WUS with same 16 predictors: `{metrics['full_wus_same16_rf_test_rmse']:.6f}`",
        "",
        "Predictors:",
    ]
    lines.extend([f"- `{p}`" for p in PREDICTORS])
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
