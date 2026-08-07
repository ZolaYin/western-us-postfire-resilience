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
CLIM_TILE_TOP = Path(
    "/path/to/google-drive/我的云端硬盘/WesternUS_drivers_raw/GRIDMET_STACK_2000_2023_WesternUS_11states-0000000000-0000000000.tif"
)
CLIM_TILE_BOTTOM = Path(
    "/path/to/google-drive/我的云端硬盘/WesternUS_drivers_raw/GRIDMET_STACK_2000_2023_WesternUS_11states-0000002048-0000000000.tif"
)

OUT_DIR = ROOT / "formal_transform_sensitivity_models_20260411"
OUT_METRICS = OUT_DIR / "formal_transform_sensitivity_metrics.csv"
OUT_AUDIT = OUT_DIR / "formal_transform_sensitivity_audit.json"
OUT_REPORT = OUT_DIR / "formal_transform_sensitivity_report.md"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500

RESPONSES = ["Resistance", "IRI_good_pow2", "STAB_good_pow2"]

BASELINE_PREDICTORS = [
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

TRANSFORMED_PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_log1p_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_clean_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_log1p_z",
    "HUM_popdens_win10km_log_z",
    "HUM_roaddens_r5km_log1p_z",
    "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_log1p_z",
    "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_eto_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_log1p_z",
    "CLIM_aridity_pre_clean_z",
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


def ensure_current_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for z_col, base_col in BASE_TO_Z.items():
        if z_col not in out.columns:
            out[z_col] = zscore(out[base_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    return out


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


def repair_aridity_pre(df: pd.DataFrame) -> tuple[pd.Series, dict]:
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
    valid_year_counts = np.zeros(xs.shape[0], dtype=np.int16)
    for i, year in enumerate(t0):
        pre_years = [year - 3, year - 2, year - 1]
        idxs = [year_to_pos[y] for y in pre_years if y in year_to_pos]
        vals = stack[i, idxs]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            repaired_vals[i] = float(np.mean(vals))
            valid_year_counts[i] = len(vals)
    repaired.loc[affected] = repaired_vals
    audit = {
        "affected_rows_original_lt_minus1000": int(affected.sum()),
        "repaired_nonmissing": int(np.isfinite(repaired_vals).sum()),
        "still_missing_after_repair": int(np.isnan(repaired_vals).sum()),
        "rows_with_1_valid_pre_year": int((valid_year_counts == 1).sum()),
        "rows_with_2_valid_pre_years": int((valid_year_counts == 2).sum()),
        "rows_with_3_valid_pre_years": int((valid_year_counts == 3).sum()),
        "remaining_lt_minus1000_after_repair": int((pd.to_numeric(repaired, errors='coerce') < -1000).sum()),
    }
    return repaired.astype(np.float32), audit


def fit_rf(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT).copy()
    df = ensure_current_columns(df)

    # Verified input cleaning
    ts_soc_invalid = pd.to_numeric(df["TS_SOC_0_30cm"], errors="coerce").astype(float) == -9999
    df["TS_SOC_0_30cm_clean"] = pd.to_numeric(df["TS_SOC_0_30cm"], errors="coerce").astype(float)
    df.loc[ts_soc_invalid, "TS_SOC_0_30cm_clean"] = np.nan
    df["TS_SOC_0_30cm_clean_z"] = zscore(df["TS_SOC_0_30cm_clean"])

    aridity_clean, aridity_audit = repair_aridity_pre(df)
    df["CLIM_aridity_pre_clean"] = aridity_clean
    df["CLIM_aridity_pre_clean_z"] = zscore(df["CLIM_aridity_pre_clean"])

    # Requested transform sensitivity
    df["TS_twi_log1p_z"] = log1p_z(df["TS_twi"])
    df["FS_CBH_t0agg_log1p_z"] = log1p_z(df["FS_CBH_t0agg"])
    df["HUM_roaddens_r5km_log1p_z"] = log1p_z(df["HUM_roaddens_r5km"])
    df["HUM_imperv_near_t0_log1p_z"] = log1p_z(df["HUM_imperv_near_t0"])
    df["CLIM_hot_days_35C_pre_log1p_z"] = log1p_z(df["CLIM_hot_days_35C_pre"])

    audit = {
        "input_table": str(INPUT),
        "n_estimators": N_ESTIMATORS,
        "ts_soc_invalid_eq_minus9999": int(ts_soc_invalid.sum()),
        "ts_soc_remaining_negative_after_clean": int(
            (pd.to_numeric(df["TS_SOC_0_30cm_clean"], errors="coerce") < 0).sum()
        ),
        "aridity_repair": aridity_audit,
    }

    results: list[dict] = []
    all_needed = list(dict.fromkeys(BASELINE_PREDICTORS + TRANSFORMED_PREDICTORS))

    for response in RESPONSES:
        work = (
            df[[response] + all_needed]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .reset_index(drop=True)
            .copy()
        )
        X_base = work[BASELINE_PREDICTORS]
        X_tx = work[TRANSFORMED_PREDICTORS]
        y = work[response]

        indices = np.arange(len(work))
        tr_idx, te_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        pred_base = fit_rf(X_base.iloc[tr_idx], y.iloc[tr_idx], X_base.iloc[te_idx])
        pred_tx = fit_rf(X_tx.iloc[tr_idx], y.iloc[tr_idx], X_tx.iloc[te_idx])

        base_row = {
            "response": response,
            "variant": "baseline_current_transforms",
            "rows_used": int(len(work)),
            "train_rows": int(len(tr_idx)),
            "test_rows": int(len(te_idx)),
            "test_r2": float(r2_score(y.iloc[te_idx], pred_base)),
            "test_rmse": float(np.sqrt(mean_squared_error(y.iloc[te_idx], pred_base))),
        }
        tx_row = {
            "response": response,
            "variant": "cleaned_plus_requested_transforms",
            "rows_used": int(len(work)),
            "train_rows": int(len(tr_idx)),
            "test_rows": int(len(te_idx)),
            "test_r2": float(r2_score(y.iloc[te_idx], pred_tx)),
            "test_rmse": float(np.sqrt(mean_squared_error(y.iloc[te_idx], pred_tx))),
        }
        tx_row["delta_r2_vs_baseline"] = tx_row["test_r2"] - base_row["test_r2"]
        tx_row["delta_rmse_vs_baseline"] = tx_row["test_rmse"] - base_row["test_rmse"]

        results.extend([base_row, tx_row])
        pd.DataFrame(results).to_csv(OUT_METRICS, index=False)

    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    res_df = pd.DataFrame(results)
    lines = [
        "# Formal Transform Sensitivity Models",
        "",
        f"- Input table: `{INPUT}`",
        f"- RF trees: `{N_ESTIMATORS}`",
        f"- Shared split: random `{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}` with random_state `{RANDOM_STATE}`",
        "",
        "Data cleaning applied before transformed reruns:",
        f"- `TS_SOC_0_30cm == -9999` set to missing: `{audit['ts_soc_invalid_eq_minus9999']}` rows",
        f"- `CLIM_aridity_pre < -1000` rows detected: `{audit['aridity_repair']['affected_rows_original_lt_minus1000']}`",
        f"- `CLIM_aridity_pre` rows successfully repaired from valid yearly values: `{audit['aridity_repair']['repaired_nonmissing']}`",
        f"- `CLIM_aridity_pre` rows still missing after repair: `{audit['aridity_repair']['still_missing_after_repair']}`",
        "",
        "Requested transformed predictors:",
        "- `TS_twi -> log1p`",
        "- `FS_CBH_t0agg -> log1p`",
        "- `HUM_roaddens_r5km -> log1p`",
        "- `HUM_imperv_near_t0 -> log1p`",
        "- `CLIM_hot_days_35C_pre -> log1p`",
        "- `TS_SOC_0_30cm` cleaned before z-scoring",
        "- `CLIM_aridity_pre` repaired before z-scoring",
        "",
        "Model results:",
    ]
    for response in RESPONSES:
        sub = res_df[res_df["response"] == response].copy()
        base = sub[sub["variant"] == "baseline_current_transforms"].iloc[0]
        tx = sub[sub["variant"] == "cleaned_plus_requested_transforms"].iloc[0]
        lines.append(
            f"- `{response}` baseline: R2=`{base['test_r2']:.6f}`, RMSE=`{base['test_rmse']:.6f}`"
        )
        lines.append(
            f"- `{response}` transformed: R2=`{tx['test_r2']:.6f}`, RMSE=`{tx['test_rmse']:.6f}`, delta_R2=`{tx['delta_r2_vs_baseline']:.6f}`"
        )
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"metrics_csv": str(OUT_METRICS), "audit_json": str(OUT_AUDIT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
