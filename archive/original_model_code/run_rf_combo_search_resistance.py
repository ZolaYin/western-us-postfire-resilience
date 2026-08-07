#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
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
CURRENT_BEST_METRICS = (
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

OUT_DIR = ROOT / "rf_combo_search_resistance_20260411"
OUT_RESULTS = OUT_DIR / "rf_combo_search_results.csv"
OUT_BEST = OUT_DIR / "rf_combo_search_best.json"
OUT_REPORT = OUT_DIR / "rf_combo_search_report.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2
RESPONSE = "Resistance"


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def log1p_z(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float).clip(lower=0)
    return zscore(np.log1p(clean))


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


def fit_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    n_estimators: int,
    max_features,
    min_samples_leaf: int,
    max_depth,
) -> dict[str, float]:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT).copy()

    # Clean data issues already verified
    df["TS_SOC_0_30cm_clean"] = pd.to_numeric(df["TS_SOC_0_30cm"], errors="coerce").astype(float)
    df.loc[df["TS_SOC_0_30cm_clean"] == -9999, "TS_SOC_0_30cm_clean"] = np.nan
    df["CLIM_aridity_pre_clean"] = repair_aridity_pre(df)

    # Base transforms
    df["TS_elev_m_z"] = zscore(df["TS_elev_m"])
    df["TS_slope_deg_z"] = zscore(df["TS_slope_deg"])
    df["TS_northness_z"] = zscore(df["TS_northness"])
    df["TS_eastness_z"] = zscore(df["TS_eastness"])
    df["TS_twi_z"] = zscore(df["TS_twi"])
    df["TS_roughness_z"] = zscore(df["TS_roughness"])
    df["TS_SOC_0_30cm_clean_z"] = zscore(df["TS_SOC_0_30cm_clean"])
    df["FS_TCC_t0_z"] = zscore(df["FS_TCC_t0"])
    df["FS_CBH_t0agg_z"] = zscore(df["FS_CBH_t0agg"])
    df["HUM_popdens_win10km_log_z"] = log1p_z(df["HUM_popdens_win10km"])
    df["HUM_roaddens_r5km_z"] = zscore(df["HUM_roaddens_r5km"])
    df["HUM_traildens_r10km_z"] = zscore(df["HUM_traildens_r10km"])
    df["HUM_imperv_near_t0_z"] = zscore(df["HUM_imperv_near_t0"])
    df["HUM_viirs_near_t0_log_z"] = log1p_z(df["HUM_viirs_near_t0"])
    df["CLIM_pr_sum_pre_z"] = zscore(df["CLIM_pr_sum_pre"])
    df["CLIM_eto_sum_pre_z"] = zscore(df["CLIM_eto_sum_pre"])
    df["CLIM_tmmn_mean_pre_z"] = zscore(df["CLIM_tmmn_mean_pre"])
    df["CLIM_hot_days_35C_pre_z"] = zscore(df["CLIM_hot_days_35C_pre"])
    df["CLIM_aridity_pre_clean_z"] = zscore(df["CLIM_aridity_pre_clean"])
    df["CLIM_tmmx_std_pre_z"] = zscore(df["CLIM_tmmx_std_pre"])

    # Spatial polynomial terms
    df["x_sq_z"] = zscore(pd.to_numeric(df["x"], errors="coerce").astype(float) ** 2)
    df["y_sq_z"] = zscore(pd.to_numeric(df["y"], errors="coerce").astype(float) ** 2)
    df["xy_z"] = zscore(
        pd.to_numeric(df["x"], errors="coerce").astype(float)
        * pd.to_numeric(df["y"], errors="coerce").astype(float)
    )

    base_core = [
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
    ]

    combos = {
        "base_xy": base_core + ["x", "y"],
        "base_xy_poly": base_core + ["x", "y", "x_sq_z", "y_sq_z", "xy_z"],
        "evtres_xy": base_core + ["FS_EVT_resistance_proxy", "x", "y"],
        "evtres_xy_poly": base_core + ["FS_EVT_resistance_proxy", "x", "y", "x_sq_z", "y_sq_z", "xy_z"],
        "evtboth_xy": base_core + ["FS_EVT_resistance_proxy", "FS_EVT_regeneration_proxy", "x", "y"],
        "evtboth_xy_poly": base_core
        + ["FS_EVT_resistance_proxy", "FS_EVT_regeneration_proxy", "x", "y", "x_sq_z", "y_sq_z", "xy_z"],
    }

    union_cols = list(dict.fromkeys([RESPONSE] + [c for cols in combos.values() for c in cols]))
    work = df[union_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True).copy()

    X_idx = np.arange(len(work))
    tr_idx, te_idx = train_test_split(X_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_train = work.iloc[tr_idx][RESPONSE]
    y_test = work.iloc[te_idx][RESPONSE]

    current_best = json.loads(CURRENT_BEST_METRICS.read_text())
    rows: list[dict] = []

    # Stage 1: structure search
    for combo_name, predictors in combos.items():
        scores = fit_eval(
            work.iloc[tr_idx][predictors],
            y_train,
            work.iloc[te_idx][predictors],
            y_test,
            n_estimators=200,
            max_features=1.0,
            min_samples_leaf=1,
            max_depth=None,
        )
        row = {
            "stage": "structure_search",
            "combo": combo_name,
            "predictor_count": len(predictors),
            "rows_used_common_subset": int(len(work)),
            "n_estimators": 200,
            "max_features": 1.0,
            "min_samples_leaf": 1,
            "max_depth": "None",
            **scores,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_RESULTS, index=False)

    stage1 = pd.DataFrame(rows)
    best_combo = stage1.sort_values("test_r2", ascending=False).iloc[0]["combo"]

    # Stage 2: parameter tuning on best structure, plus anchor base_xy at 500 on same subset
    tune_runs = [
        ("base_xy_anchor_500", "base_xy", 500, 1.0, 1, None),
        ("best_default_500", best_combo, 500, 1.0, 1, None),
        ("best_sqrt_500", best_combo, 500, "sqrt", 1, None),
        ("best_half_leaf2_500", best_combo, 500, 0.5, 2, None),
        ("best_sqrt_leaf2_depth20_500", best_combo, 500, "sqrt", 2, 20),
    ]
    seen = set()
    for label, combo_name, n_estimators, max_features, min_samples_leaf, max_depth in tune_runs:
        key = (combo_name, n_estimators, str(max_features), min_samples_leaf, str(max_depth))
        if key in seen:
            continue
        seen.add(key)
        predictors = combos[combo_name]
        scores = fit_eval(
            work.iloc[tr_idx][predictors],
            y_train,
            work.iloc[te_idx][predictors],
            y_test,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
        )
        row = {
            "stage": "parameter_tuning",
            "combo": combo_name,
            "label": label,
            "predictor_count": len(predictors),
            "rows_used_common_subset": int(len(work)),
            "n_estimators": n_estimators,
            "max_features": str(max_features),
            "min_samples_leaf": min_samples_leaf,
            "max_depth": str(max_depth),
            **scores,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_RESULTS, index=False)

    res = pd.DataFrame(rows).sort_values(["stage", "test_r2"], ascending=[True, False]).reset_index(drop=True)
    best_overall = res.sort_values("test_r2", ascending=False).iloc[0].to_dict()
    best_overall["reference_current_best_r2"] = float(current_best["test_r2"])
    best_overall["reference_current_best_rows"] = int(current_best["rows_used"])
    best_overall["common_subset_rows"] = int(len(work))
    OUT_BEST.write_text(json.dumps(best_overall, indent=2), encoding="utf-8")

    lines = [
        "# RF Combo Search For Resistance",
        "",
        f"- Input table: `{INPUT}`",
        f"- Response: `{RESPONSE}`",
        f"- Current reference complete corrected RF R2: `{float(current_best['test_r2']):.6f}` on `{int(current_best['rows_used'])}` rows",
        f"- Common subset rows used in this combo search: `{len(work)}`",
        "",
        "Top structure-search runs (200 trees):",
    ]
    for _, row in res[res["stage"] == "structure_search"].sort_values("test_r2", ascending=False).iterrows():
        lines.append(
            f"- `{row['combo']}`: R2=`{row['test_r2']:.6f}`, RMSE=`{row['test_rmse']:.6f}`, predictors=`{int(row['predictor_count'])}`"
        )
    lines.extend(["", "Parameter tuning runs:"])
    for _, row in res[res["stage"] == "parameter_tuning"].sort_values("test_r2", ascending=False).iterrows():
        lines.append(
            f"- `{row['label']}` on `{row['combo']}`: R2=`{row['test_r2']:.6f}`, RMSE=`{row['test_rmse']:.6f}`, max_features=`{row['max_features']}`, min_samples_leaf=`{row['min_samples_leaf']}`, max_depth=`{row['max_depth']}`"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- `TS_SOC_0_30cm == -9999` was treated as missing before z-scoring.",
            "- `CLIM_aridity_pre < -1000` rows were repaired from valid yearly aridity values before z-scoring.",
            "- This combo search uses one common complete-case subset across all tested structures, so structure comparisons are apples-to-apples.",
        ]
    )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"results_csv": str(OUT_RESULTS), "best_json": str(OUT_BEST)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
