#!/usr/bin/env python3
"""Rebuild corrected WesternUS response metrics from legacy RESI + MTBS + NLCD-by-t0.

This script copies the verified coast production metric logic and only changes
the input paths/grid to the WesternUS11 domain. It intentionally writes to a
new output location so the earlier incorrect WesternUS response table is not
silently overwritten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path("/path/to/google-drive/我的云端硬盘")
OUT_DIR = ROOT / "US_Fire_and_Ecology_Data" / "WUS_1km" / "westernus_response_legacy_nlcd_20260410"
DIR_RESI = ROOT / "WesternUS_RESI_u16_1km"
DIR_MTBS = ROOT / "US_Fire_and_Ecology_Data" / "WesternUS_MTBS_1km"
DIR_NLCD = ROOT / "US_Fire_and_Ecology_Data" / "WesternUS_NLCD_ForestMasks"
COAST_TEMPLATE_SCRIPT = ROOT / "US_Fire_and_Ecology_Data" / "WUS_1km" / "rebuild_coast_response_metrics.py"

POSTFIRE_WINDOW = 10
MIN_VALID_YEARS = 6
RESI_SCALE = 10000.0
RECOVERY_NOT_REACHED_VALUE = float(POSTFIRE_WINDOW)


def read_float_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, profile


def read_stack() -> tuple[np.ndarray, np.ndarray, dict]:
    resi_files = sorted(DIR_RESI.glob("WesternUS_RESI_u16_*.tif"))
    years = []
    stack = []
    ref_profile = None

    for path in resi_files:
        year = int(path.stem[-4:])
        if not (2000 <= year <= 2023):
            continue
        arr, profile = read_float_raster(path)
        arr[arr <= 0] = np.nan
        arr = arr / RESI_SCALE
        years.append(year)
        stack.append(arr)
        if ref_profile is None:
            ref_profile = profile

    if not stack:
        raise RuntimeError("No legacy WesternUS RESI yearly rasters were loaded.")

    return np.stack(stack, axis=0), np.array(years, dtype=np.int16), ref_profile


def read_mask_uint8(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.uint8, copy=False)


def read_int16_raster(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.int16, copy=False)


def interp_1d_nan(vec: np.ndarray) -> np.ndarray:
    x = np.arange(vec.size)
    m = np.isfinite(vec)
    if m.sum() == 0:
        return vec
    if m.sum() == 1:
        return vec
    return np.interp(x, x[m], vec[m]).astype(np.float32)


def flatten_valid_table(
    ref_profile: dict,
    valid_mask: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows, cols = np.where(valid_mask)
    transform = ref_profile["transform"]
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

    data = {
        "pixel_id": np.arange(len(rows), dtype=np.int64),
        "row": rows.astype(np.int32),
        "col": cols.astype(np.int32),
        "x": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.float64),
    }
    for key, arr in arrays.items():
        data[key] = arr[rows, cols]
    return pd.DataFrame(data)


def write_float_tif(path: Path, ref_profile: dict, arr: np.ndarray, nodata: float = -9999.0) -> None:
    profile = ref_profile.copy()
    profile.update(dtype="float32", count=1, nodata=nodata, compress="lzw")
    out = arr.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def write_int_tif(path: Path, ref_profile: dict, arr: np.ndarray, dtype: str, nodata: int) -> None:
    profile = ref_profile.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    out = arr.copy()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out.astype(dtype), 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    RESI, years, ref_profile = read_stack()
    t0 = read_int16_raster(DIR_MTBS / "WesternUS_MTBS_t0_1km.tif")
    sev = read_mask_uint8(DIR_MTBS / "WesternUS_MTBS_SEV_at_t0_1km.tif")

    nlcd_masks = {
        2006: read_mask_uint8(DIR_NLCD / "WesternUS_NLCD_Forest_2006.tif"),
        2011: read_mask_uint8(DIR_NLCD / "WesternUS_NLCD_Forest_2011.tif"),
        2016: read_mask_uint8(DIR_NLCD / "WesternUS_NLCD_Forest_2016.tif"),
        2019: read_mask_uint8(DIR_NLCD / "WesternUS_NLCD_Forest_2019.tif"),
    }

    H, W = t0.shape
    year_to_index = {int(y): i for i, y in enumerate(years)}

    forest_at_t0 = np.zeros_like(t0, dtype=np.uint8)
    cond_2006 = (t0 <= 2010) & (t0 > 0)
    cond_2011 = (t0 >= 2011) & (t0 <= 2015)
    cond_2016 = (t0 >= 2016) & (t0 <= 2018)
    cond_2019 = t0 >= 2019
    forest_at_t0[cond_2006] = nlcd_masks[2006][cond_2006]
    forest_at_t0[cond_2011] = nlcd_masks[2011][cond_2011]
    forest_at_t0[cond_2016] = nlcd_masks[2016][cond_2016]
    forest_at_t0[cond_2019] = nlcd_masks[2019][cond_2019]
    forest_at_t0[t0 == 0] = nlcd_masks[2019][t0 == 0]

    resi_year_min = int(years.min())
    resi_year_max = int(years.max())
    t0_min_res = max(2000, resi_year_min + 5)
    t0_max_res = min(resi_year_max - 1, 2023)
    t0_min_long = max(2000, resi_year_min + 5)
    t0_max_long = min(resi_year_max - (POSTFIRE_WINDOW - 1), 2023)

    Bi = np.full((H, W), np.nan, dtype=np.float32)
    Rmin = np.full((H, W), np.nan, dtype=np.float32)
    T50 = np.full((H, W), np.nan, dtype=np.float32)
    T80 = np.full((H, W), np.nan, dtype=np.float32)
    T90 = np.full((H, W), np.nan, dtype=np.float32)
    T95 = np.full((H, W), np.nan, dtype=np.float32)
    T50_reached = np.full((H, W), np.nan, dtype=np.float32)
    T80_reached = np.full((H, W), np.nan, dtype=np.float32)
    T90_reached = np.full((H, W), np.nan, dtype=np.float32)
    T95_reached = np.full((H, W), np.nan, dtype=np.float32)
    IRI_gap = np.full((H, W), np.nan, dtype=np.float32)
    RESI_post_CV = np.full((H, W), np.nan, dtype=np.float32)
    STAB_CV = np.full((H, W), np.nan, dtype=np.float32)
    AUC_deficit = np.full((H, W), np.nan, dtype=np.float32)

    candidate_mask = (
        (t0 >= t0_min_res)
        & (t0 <= t0_max_res)
        & (forest_at_t0 == 1)
    )
    candidate_rows, candidate_cols = np.where(candidate_mask)

    n_resistance_valid = 0
    n_long_valid = 0

    recovery_thresholds = {
        "T50": 0.50,
        "T80": 0.80,
        "T90": 0.90,
        "T95": 0.95,
    }
    recovery_year_arrays = {
        "T50": T50,
        "T80": T80,
        "T90": T90,
        "T95": T95,
    }
    recovery_status_arrays = {
        "T50": T50_reached,
        "T80": T80_reached,
        "T90": T90_reached,
        "T95": T95_reached,
    }

    for i, j in zip(candidate_rows, candidate_cols):
        yr0 = int(t0[i, j])
        idx_pre_start = year_to_index.get(yr0 - 5)
        idx_pre_end = year_to_index.get(yr0 - 1)
        idx0 = year_to_index.get(yr0)
        if idx_pre_start is None or idx_pre_end is None or idx0 is None:
            continue

        pre_slice = RESI[idx_pre_start:idx_pre_end + 1, i, j]
        if np.all(np.isnan(pre_slice)):
            continue

        Bi_ij = np.nanmean(pre_slice)
        if not np.isfinite(Bi_ij) or Bi_ij <= 0:
            continue
        Bi[i, j] = Bi_ij

        idx_post_short_end = year_to_index.get(yr0 + 1)
        if idx_post_short_end is None:
            continue
        post_short = RESI[idx0:idx_post_short_end + 1, i, j]
        if np.all(np.isnan(post_short)):
            continue
        Rmin_ij = np.nanmin(post_short)
        Rmin[i, j] = Rmin_ij
        n_resistance_valid += 1

        if yr0 < t0_min_long or yr0 > t0_max_long:
            continue

        post_vec = np.full(POSTFIRE_WINDOW, np.nan, dtype=np.float32)
        for k in range(POSTFIRE_WINDOW):
            idx_k = year_to_index.get(yr0 + k)
            if idx_k is not None:
                post_vec[k] = RESI[idx_k, i, j]

        valid_count = np.isfinite(post_vec).sum()
        if valid_count < MIN_VALID_YEARS:
            continue
        n_long_valid += 1

        post_vec_filled = interp_1d_nan(post_vec)
        rel = post_vec_filled / Bi_ij
        rel = np.clip(rel, 0, None)
        rel_recovery = rel[2:]

        for metric_name, threshold in recovery_thresholds.items():
            reached = rel_recovery.size > 0 and (rel_recovery >= threshold).any()
            if reached:
                recovery_status_arrays[metric_name][i, j] = 1.0
                recovery_year_arrays[metric_name][i, j] = float(2 + np.argmax(rel_recovery >= threshold))
            else:
                recovery_status_arrays[metric_name][i, j] = 0.0
                recovery_year_arrays[metric_name][i, j] = RECOVERY_NOT_REACHED_VALUE

        seg = np.clip(post_vec_filled, 0, None)
        mean_seg = np.nanmean(seg)
        auc_obs = mean_seg * POSTFIRE_WINDOW
        auc_base = Bi_ij * POSTFIRE_WINDOW
        iri_gap = 1.0 - (auc_obs / auc_base)
        iri_gap = np.clip(iri_gap, 0, 1)
        IRI_gap[i, j] = iri_gap
        deficit = (auc_base - auc_obs) / auc_base
        AUC_deficit[i, j] = np.clip(deficit, 0, 1)
        if mean_seg > 0:
            cv = np.nanstd(seg) / mean_seg
            RESI_post_CV[i, j] = cv
            STAB_CV[i, j] = 1.0 / (1.0 + cv)

    Resistance = np.where(np.isfinite(Bi) & (Bi > 0), Rmin / Bi, np.nan).astype(np.float32)
    IRI_good = np.clip(1.0 - IRI_gap, 0, 1).astype(np.float32)
    STAB_good = np.clip(STAB_CV.copy(), 0, 1).astype(np.float32)
    IRI_good_pow2 = (IRI_good ** 2).astype(np.float32)
    STAB_good_pow2 = (STAB_good ** 2).astype(np.float32)

    table_valid = np.isfinite(Resistance)
    arrays_for_table = {
        "t0_year": t0,
        "sev": sev.astype(np.float32),
        "Forest_at_t0": forest_at_t0.astype(np.float32),
        "Bi": Bi,
        "Rmin": Rmin,
        "Resistance": Resistance,
        "T50": T50,
        "T50_reached": T50_reached,
        "T80": T80,
        "T80_reached": T80_reached,
        "T90": T90,
        "T90_reached": T90_reached,
        "T95": T95,
        "T95_reached": T95_reached,
        "IRI_gap": IRI_gap,
        "IRI_good": IRI_good,
        "IRI_good_pow2": IRI_good_pow2,
        "RESI_post_CV": RESI_post_CV,
        "STAB_CV": STAB_CV,
        "STAB_good": STAB_good,
        "STAB_good_pow2": STAB_good_pow2,
        "AUC_deficit": AUC_deficit,
    }
    df = flatten_valid_table(ref_profile, table_valid, arrays_for_table)

    parquet_path = OUT_DIR / "westernus_response_metrics_legacy_nlcd.parquet"
    csv_path = OUT_DIR / "westernus_response_metrics_legacy_nlcd.csv"
    audit_path = OUT_DIR / "westernus_response_metrics_legacy_nlcd_audit.md"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    t0_out = t0.astype(np.int16, copy=True)
    t0_out[t0_out <= 0] = -9999
    write_int_tif(OUT_DIR / "WesternUS_t0_year_legacy_nlcd.tif", ref_profile, t0_out, "int16", -9999)
    write_int_tif(OUT_DIR / "WesternUS_SEV_at_t0_legacy_nlcd.tif", ref_profile, sev.astype(np.uint8), "uint8", 0)
    write_int_tif(OUT_DIR / "WesternUS_Forest_at_t0_legacy_nlcd.tif", ref_profile, forest_at_t0.astype(np.uint8), "uint8", 0)
    write_float_tif(OUT_DIR / "WesternUS_Bi_legacy_nlcd.tif", ref_profile, Bi)
    write_float_tif(OUT_DIR / "WesternUS_Rmin_legacy_nlcd.tif", ref_profile, Rmin)
    write_float_tif(OUT_DIR / "WesternUS_Resistance_legacy_nlcd.tif", ref_profile, Resistance)
    write_float_tif(OUT_DIR / "WesternUS_T80_legacy_nlcd.tif", ref_profile, T80)
    write_float_tif(OUT_DIR / "WesternUS_IRI_good_pow2_legacy_nlcd.tif", ref_profile, IRI_good_pow2)
    write_float_tif(OUT_DIR / "WesternUS_STAB_good_pow2_legacy_nlcd.tif", ref_profile, STAB_good_pow2)

    audit_text = f"""# WesternUS Response Metrics Rebuild Audit

## Purpose
- Rebuild corrected WesternUS response metrics using the verified coast production logic.
- Keep the coast metric definitions unchanged.
- Replace the earlier incorrect WesternUS response chain that used `RESI_fast_WesternUS_*`.

## Source Of Truth
- Coast production template script: `{COAST_TEMPLATE_SCRIPT}`

## Inputs
- Legacy WesternUS RESI: `{DIR_RESI}`
- WesternUS MTBS: `{DIR_MTBS}`
- WesternUS NLCD forest masks: `{DIR_NLCD}`

## Forest Mask Rule
- `2006` for `t0 <= 2010`
- `2011` for `2011..2015`
- `2016` for `2016..2018`
- `2019` for `t0 >= 2019`
- `2019` for `t0 == 0`

## Metric Rules
- `Bi = mean(t0-5 .. t0-1)`
- `Rmin = min(t0 .. t0+1)`
- `Resistance = Rmin / Bi`
- postfire window = `10`
- minimum valid postfire years = `6`
- interpolation = `1-D linear`
- recovery search starts at `t0+2`
- `IRI_gap = 1 - auc_obs / auc_base`
- `IRI_good_pow2 = (clip(1 - IRI_gap, 0, 1)) ** 2`
- `STAB_CV = 1 / (1 + CV)`
- `STAB_good_pow2 = (clip(STAB_CV, 0, 1)) ** 2`

## Outputs
- Parquet: `{parquet_path}`
- CSV: `{csv_path}`
- Audit: `{audit_path}`

## Counts
- resistance_valid = `{n_resistance_valid}`
- long_window_valid = `{n_long_valid}`
- table_rows = `{len(df)}`

## Remaining Uncertainty
- Whether the recovered WesternUS NLCD forest-mask rule is exact is determined upstream by `verify_and_build_westernus_nlcd_forest_masks_via_ee.py`.
- This script assumes the upstream mask files already passed that validation step.
"""
    audit_path.write_text(audit_text, encoding="utf-8")

    print(f"Corrected WesternUS response rebuild complete.")
    print(f"Rows={len(df)} resistance_valid={n_resistance_valid} long_valid={n_long_valid}")
    print(f"Outputs: {parquet_path} | {csv_path} | {audit_path}")


if __name__ == "__main__":
    main()
