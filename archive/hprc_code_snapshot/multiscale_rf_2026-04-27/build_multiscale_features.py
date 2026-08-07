#!/usr/bin/env python3
"""
Build multi-scale Gaussian-smoothed features for RF.

For each predictor, creates spatially smoothed versions at multiple radii.
Scales are chosen to span the MGWR bandwidth hierarchy from stage5b:

  CBH bw=122 → ~140 km   →  captured by 150 km scale
  roaddens bw=252 → ~200 km  →  captured by 150–300 km
  SOC/pr bw=503–513 → ~280 km  →  captured by 300 km
  elev/TCC bw=652–1018 → ~320–400 km  →  captured by 300–600 km
  slope/tmmn bw=1856–3372 → ~540–730 km  →  captured by 600 km
  viirs/imperv bw=11588–11995 → ~1350 km  →  captured by 1200 km

Usage:
  python build_multiscale_features.py \
    --input  mgwr_multi_y_samples/sample_n12000_seed42.parquet \
    --output mgwr_multi_y_samples/sample_n12000_seed42_multiscale.parquet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

SCALES_KM = [50, 150, 300, 600, 1200]

PREDICTORS = [
    "FS_CBH_t0agg_z",
    "HUM_roaddens_r5km_z",
    "TS_SOC_0_30cm_z",
    "CLIM_pr_sum_pre_z",
    "TS_elev_m_z",
    "FS_TCC_t0_z",
    "TS_slope_deg_z",
    "CLIM_tmmn_mean_pre_z",
    "HUM_traildens_r10km_z",
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
]


def adaptive_k(n: int, r_km: float, coords: np.ndarray, max_k: int = 5000) -> int:
    """Estimate how many KNN to query for radius r_km based on point density."""
    x_range = (coords[:, 0].max() - coords[:, 0].min()) / 1000  # km
    y_range = (coords[:, 1].max() - coords[:, 1].min()) / 1000
    area_km2 = x_range * y_range
    density = n / area_km2  # pts/km²
    expected = density * np.pi * r_km ** 2
    # 2× buffer; hard cap to avoid memory explosion on large n
    return int(min(n - 1, max_k, max(50, expected * 2)))


def gaussian_smooth(
    coords: np.ndarray,  # (n, 2) in meters
    values: np.ndarray,  # (n,)
    radius_m: float,
    k: int,
) -> np.ndarray:
    """Gaussian kernel smooth: each point gets weighted avg of k nearest neighbors."""
    tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=k)      # (n, k)
    weights = np.exp(-0.5 * (dists / radius_m) ** 2)
    weights[dists > 2.5 * radius_m] = 0.0      # hard cutoff at 2.5σ
    w_sum = weights.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum == 0, 1.0, w_sum)   # avoid div-by-zero
    smoothed = (weights * values[idxs]).sum(axis=1) / w_sum.squeeze()
    return smoothed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--scales-km",   nargs="+", type=int, default=SCALES_KM)
    parser.add_argument("--predictors",  nargs="+",           default=PREDICTORS)
    parser.add_argument("--col-map",     nargs="*",           default=[],
                        help="Column renames before processing: old=new (e.g. TS_SOC_0_30cm_clean_z=TS_SOC_0_30cm_z)")
    parser.add_argument("--max-k",       type=int,            default=5000,
                        help="Hard cap on KNN neighbors (avoids memory explosion on large n)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    for pair in args.col_map:
        old, new = pair.split("=")
        if old in df.columns:
            df = df.rename(columns={old: new})
            print(f"Renamed column: {old} → {new}", flush=True)
    coords = df[["x", "y"]].to_numpy(dtype=float)
    n = len(df)
    print(f"n={n}  building KDTree ...", flush=True)

    new_cols: dict[str, np.ndarray] = {}

    for r_km in args.scales_km:
        r_m = r_km * 1000.0
        k = adaptive_k(n, r_km, coords, max_k=args.max_k)
        print(f"\n── {r_km} km  (k={k}) ──", flush=True)
        t0 = time.time()

        for pred in args.predictors:
            if pred not in df.columns:
                print(f"  [skip] {pred} not in data")
                continue
            values = df[pred].fillna(0.0).to_numpy(dtype=float)
            col = f"{pred}_s{r_km}km"
            new_cols[col] = gaussian_smooth(coords, values, r_m, k)
            print(f"  {col}", flush=True)

        print(f"  {r_km} km done in {time.time()-t0:.1f}s", flush=True)

    out = pd.concat(
        [df.reset_index(drop=True),
         pd.DataFrame(new_cols)],
        axis=1,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"\nSaved {len(new_cols)} new columns → {args.output}")
    print("Column summary:")
    for r_km in args.scales_km:
        cols = [c for c in new_cols if f"_s{r_km}km" in c]
        print(f"  {r_km} km: {len(cols)} columns")


if __name__ == "__main__":
    main()
