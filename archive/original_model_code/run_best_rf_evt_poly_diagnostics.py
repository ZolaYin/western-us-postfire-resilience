#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


ROOT = Path(
    "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
REFERENCE_METRICS = (
    ROOT
    / "westernus_rf_noevt_plusxy_complete_corrected_20260411"
    / "westernus_rf_noevt_plusxy_complete_corrected_metrics.json"
)
CLIM_TILE_TOP = Path(
    "/path/to/google-drive/我的云端硬盘/WesternUS_drivers_raw/GRIDMET_STACK_2000_2023_WesternUS_11states-0000000000-0000000000.tif"
)
CLIM_TILE_BOTTOM = Path(
    "/path/to/google-drive/我的云端硬盘/WesternUS_drivers_raw/GRIDMET_STACK_2000_2023_WesternUS_11states-0000002048-0000000000.tif"
)

OUT_DIR = ROOT / "best_rf_evt_poly_diagnostics_20260412"
OUT_METRICS = OUT_DIR / "best_rf_evt_poly_metrics.json"
OUT_IMPORTANCE = OUT_DIR / "best_rf_evt_poly_feature_importance.csv"
OUT_RESID = OUT_DIR / "best_rf_evt_poly_residuals.parquet"
OUT_MORAN = OUT_DIR / "best_rf_evt_poly_moran.csv"
OUT_SPATIAL = OUT_DIR / "best_rf_evt_poly_spatial_100km.json"
OUT_REPORT = OUT_DIR / "best_rf_evt_poly_report.md"

RESPONSE = "Resistance"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500
BLOCK_KM = 100
K_MORAN = 8


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def build_climate_year_band_map(descs: tuple[str | None, ...], variable_name: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for i, desc in enumerate(descs, start=1):
        label = desc or ""
        m = re.match(r"(\d+)_(.+)", label)
        if not m:
            continue
        year = 2000 + int(m.group(1))
        var = m.group(2)
        if var == variable_name:
            out[year] = i
    return out


def sample_bands_from_tile(path: Path, band_indices: list[int], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.full((xs.shape[0], len(band_indices)), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        inside = (
            (xs >= src.bounds.left)
            & (xs <= src.bounds.right)
            & (ys >= src.bounds.bottom)
            & (ys <= src.bounds.top)
        )
        coords = list(zip(xs[inside], ys[inside]))
        if coords:
            vals = np.array(list(src.sample(coords, indexes=band_indices)), dtype=np.float32)
            vals[~np.isfinite(vals)] = np.nan
            out[inside, :] = vals
    return out


def sample_two_tile_bands(paths: list[Path], band_indices: list[int], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    out = np.full((xs.shape[0], len(band_indices)), np.nan, dtype=np.float32)
    filled = np.zeros(xs.shape[0], dtype=bool)
    for path in paths:
        vals = sample_bands_from_tile(path, band_indices, xs, ys)
        valid = np.isfinite(vals).any(axis=1)
        take = valid & (~filled)
        out[take, :] = vals[take, :]
        filled[take] = True
    return out


def repair_aridity_pre(df: pd.DataFrame) -> pd.Series:
    with rasterio.open(CLIM_TILE_TOP) as src:
        descs = src.descriptions
    year_band_map = build_climate_year_band_map(descs, "aridity")

    affected = df["CLIM_aridity_pre"] < -1000
    xs = df.loc[affected, "x"].to_numpy(dtype=float)
    ys = df.loc[affected, "y"].to_numpy(dtype=float)
    t0 = df.loc[affected, "t0_year"].to_numpy(dtype=int)
    years = sorted(year_band_map)
    bands = [year_band_map[y] for y in years]
    stack = sample_two_tile_bands([CLIM_TILE_TOP, CLIM_TILE_BOTTOM], bands, xs, ys)
    stack[stack == -9999] = np.nan
    year_to_pos = {y: i for i, y in enumerate(years)}

    repaired = df["CLIM_aridity_pre"].astype(float).copy()
    repaired_vals = np.full(xs.shape[0], np.nan, dtype=np.float32)
    for i, year in enumerate(t0):
        pre_years = [year - 3, year - 2, year - 1]
        idxs = [year_to_pos[y] for y in pre_years if y in year_to_pos]
        vals = stack[i, idxs]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            repaired_vals[i] = float(np.mean(vals))
    repaired.loc[affected] = repaired_vals
    return repaired.astype(np.float32)


def build_work_table() -> pd.DataFrame:
    df = pd.read_parquet(INPUT).copy()

    df["TS_SOC_0_30cm_clean"] = pd.to_numeric(df["TS_SOC_0_30cm"], errors="coerce").astype(float)
    df.loc[df["TS_SOC_0_30cm_clean"] == -9999, "TS_SOC_0_30cm_clean"] = np.nan
    df["CLIM_aridity_pre_clean"] = repair_aridity_pre(df)

    df["TS_elev_m_z"] = zscore(df["TS_elev_m"])
    df["TS_slope_deg_z"] = zscore(df["TS_slope_deg"])
    df["TS_northness_z"] = zscore(df["TS_northness"])
    df["TS_eastness_z"] = zscore(df["TS_eastness"])
    df["TS_twi_z"] = zscore(df["TS_twi"])
    df["TS_roughness_z"] = zscore(df["TS_roughness"])
    df["TS_SOC_0_30cm_clean_z"] = zscore(df["TS_SOC_0_30cm_clean"])
    df["FS_TCC_t0_z"] = zscore(df["FS_TCC_t0"])
    df["FS_CBH_t0agg_z"] = zscore(df["FS_CBH_t0agg"])
    df["HUM_popdens_win10km_log_z"] = zscore(np.log1p(pd.to_numeric(df["HUM_popdens_win10km"], errors="coerce").clip(lower=0)))
    df["HUM_roaddens_r5km_z"] = zscore(df["HUM_roaddens_r5km"])
    df["HUM_traildens_r10km_z"] = zscore(df["HUM_traildens_r10km"])
    df["HUM_imperv_near_t0_z"] = zscore(df["HUM_imperv_near_t0"])
    df["HUM_viirs_near_t0_log_z"] = zscore(np.log1p(pd.to_numeric(df["HUM_viirs_near_t0"], errors="coerce").clip(lower=0)))
    df["CLIM_pr_sum_pre_z"] = zscore(df["CLIM_pr_sum_pre"])
    df["CLIM_eto_sum_pre_z"] = zscore(df["CLIM_eto_sum_pre"])
    df["CLIM_tmmn_mean_pre_z"] = zscore(df["CLIM_tmmn_mean_pre"])
    df["CLIM_hot_days_35C_pre_z"] = zscore(df["CLIM_hot_days_35C_pre"])
    df["CLIM_aridity_pre_clean_z"] = zscore(df["CLIM_aridity_pre_clean"])
    df["CLIM_tmmx_std_pre_z"] = zscore(df["CLIM_tmmx_std_pre"])
    df["x_sq_z"] = zscore(pd.to_numeric(df["x"], errors="coerce").astype(float) ** 2)
    df["y_sq_z"] = zscore(pd.to_numeric(df["y"], errors="coerce").astype(float) ** 2)
    df["xy_z"] = zscore(
        pd.to_numeric(df["x"], errors="coerce").astype(float)
        * pd.to_numeric(df["y"], errors="coerce").astype(float)
    )

    predictors = [
        "TS_elev_m_z",
        "TS_slope_deg_z",
        "TS_northness_z",
        "TS_eastness_z",
        "TS_twi_z",
        "TS_roughness_z",
        "TS_SOC_0_30cm_clean_z",
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
        "CLIM_aridity_pre_clean_z",
        "CLIM_tmmx_std_pre_z",
        "FS_EVT_resistance_proxy",
        "FS_EVT_regeneration_proxy",
        "x",
        "y",
        "x_sq_z",
        "y_sq_z",
        "xy_z",
    ]
    cols = list(dict.fromkeys(["pixel_id", "row", "col", "x", "y", "t0_year", RESPONSE] + predictors))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True).copy()
    return work, predictors


def build_rf() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1,
        max_depth=None,
    )


def compute_moran(work_xy: pd.DataFrame, residuals: np.ndarray) -> dict:
    weights = KNN.from_array(work_xy[["x", "y"]].to_numpy(), k=K_MORAN)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return {
        "k": int(K_MORAN),
        "n_obs": int(len(work_xy)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    work, predictors = build_work_table()
    X = work[predictors]
    y = work[RESPONSE]

    # random split metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    eval_model = build_rf()
    eval_model.fit(X_train, y_train)
    pred_test = eval_model.predict(X_test)

    # full fit for residuals and importance
    full_model = build_rf()
    full_model.fit(X, y)
    full_pred = full_model.predict(X)
    resid = y.to_numpy() - full_pred

    importance = (
        pd.DataFrame({"predictor": predictors, "importance": full_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(OUT_IMPORTANCE, index=False)

    resid_df = work[["pixel_id", "row", "col", "x", "y", "t0_year", RESPONSE]].copy()
    resid_df["prediction"] = full_pred.astype(np.float32)
    resid_df["residual"] = resid.astype(np.float32)
    resid_df.to_parquet(OUT_RESID, index=False)

    moran = compute_moran(work[["x", "y"]], resid)
    pd.DataFrame([{"model": "RF_best_evt_poly", **moran}]).to_csv(OUT_MORAN, index=False)

    # spatial validation 100 km
    block_m = BLOCK_KM * 1000.0
    groups = (
        np.floor(work["x"] / block_m).astype(int).astype(str)
        + "_"
        + np.floor(work["y"] / block_m).astype(int).astype(str)
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(X, y, groups=groups))
    block_model = build_rf()
    block_model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    block_pred = block_model.predict(X.iloc[te_idx])

    gkf = GroupKFold(n_splits=5)
    pooled_true = np.full(len(work), np.nan, dtype=np.float64)
    pooled_pred = np.full(len(work), np.nan, dtype=np.float64)
    fold_rows = []
    for fold_id, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        m = build_rf()
        m.fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        pooled_true[te] = y.iloc[te].to_numpy()
        pooled_pred[te] = pred
        fold_rows.append(
            {
                "fold": fold_id,
                "train_rows": int(len(tr)),
                "test_rows": int(len(te)),
                "train_groups": int(pd.Series(groups.iloc[tr]).nunique()),
                "test_groups": int(pd.Series(groups.iloc[te]).nunique()),
                "r2": float(r2_score(y.iloc[te], pred)),
                "rmse": float(np.sqrt(mean_squared_error(y.iloc[te], pred))),
            }
        )
    mask = np.isfinite(pooled_true) & np.isfinite(pooled_pred)
    spatial = {
        "block_km": BLOCK_KM,
        "group_holdout_20pct": {
            "rows_used": int(len(work)),
            "n_groups": int(pd.Series(groups).nunique()),
            "train_rows": int(len(tr_idx)),
            "test_rows": int(len(te_idx)),
            "train_groups": int(pd.Series(groups.iloc[tr_idx]).nunique()),
            "test_groups": int(pd.Series(groups.iloc[te_idx]).nunique()),
            "r2": float(r2_score(y.iloc[te_idx], block_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y.iloc[te_idx], block_pred))),
        },
        "group_kfold_pooled": {
            "rows_used": int(mask.sum()),
            "n_groups": int(pd.Series(groups).nunique()),
            "r2": float(r2_score(pooled_true[mask], pooled_pred[mask])),
            "rmse": float(np.sqrt(mean_squared_error(pooled_true[mask], pooled_pred[mask]))),
        },
        "folds": fold_rows,
    }
    OUT_SPATIAL.write_text(json.dumps(spatial, indent=2), encoding="utf-8")

    ref = json.loads(REFERENCE_METRICS.read_text())
    metrics = {
        "model": "RF_best_evt_poly",
        "input_table": str(INPUT),
        "rows_used": int(len(work)),
        "predictor_count": int(len(predictors)),
        "predictors": predictors,
        "params": {
            "n_estimators": N_ESTIMATORS,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "max_depth": None,
        },
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_r2": float(r2_score(y, full_pred)),
        "full_rmse": float(np.sqrt(mean_squared_error(y, full_pred))),
        "reference_noevt_plusxy_test_r2": float(ref["test_r2"]),
        "reference_noevt_plusxy_rows": int(ref["rows_used"]),
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# Best RF EVT Poly Diagnostics",
        "",
        f"- Input table: `{INPUT}`",
        f"- Rows used: `{len(work)}`",
        f"- Predictor count: `{len(predictors)}`",
        f"- Random split test R2: `{metrics['test_r2']:.6f}`",
        f"- Random split test RMSE: `{metrics['test_rmse']:.6f}`",
        f"- Reference noEVT_plusXY random split R2: `{metrics['reference_noevt_plusxy_test_r2']:.6f}`",
        "",
        "Residual Moran:",
        f"- Moran's I: `{moran['moran_i']:.6f}`",
        f"- z_norm: `{moran['z_norm']:.6f}`",
        "",
        "100 km spatial validation:",
        f"- Group holdout 20% blocks R2: `{spatial['group_holdout_20pct']['r2']:.6f}`",
        f"- Group holdout 20% blocks RMSE: `{spatial['group_holdout_20pct']['rmse']:.6f}`",
        f"- Group 5-fold pooled R2: `{spatial['group_kfold_pooled']['r2']:.6f}`",
        f"- Group 5-fold pooled RMSE: `{spatial['group_kfold_pooled']['rmse']:.6f}`",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_json": str(OUT_METRICS), "spatial_json": str(OUT_SPATIAL)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
