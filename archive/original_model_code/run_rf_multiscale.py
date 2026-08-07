#!/usr/bin/env python3
"""
Random Forest with multi-scale features + block CV.

Compares three feature sets:
  original   — 11 z-scored predictors only
  multiscale — original + smoothed at 50/150/300/600/1200 km (55 features total)
  mgwr_scale — for each predictor, only its MGWR-matched scale + original (22 features)

For each set reports: R²_randCV, R²_blkCV, RMSE_blkCV, Moran's I (block residuals).
Feature importance for multiscale set shows which scale wins per predictor.

Usage:
  python run_rf_multiscale.py \
    --input   mgwr_multi_y_samples/sample_n12000_seed42_multiscale.parquet \
    --output-dir results/rf_multiscale_seed42 \
    --response Resistance
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.spatial import KDTree

# ── Constants ──────────────────────────────────────────────────────────────────

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

SCALES_KM = [50, 150, 300, 600, 1200]

# MGWR stage5b bandwidth → nearest scale in SCALES_KM
# (CBH=122→150, road=252→150, SOC=513→300, pr=503→300,
#  elev=652→600, TCC=1018→600, slope=1856→600, tmmn=3372→1200,
#  trail=4490→1200, viirs=11588→1200, imperv=11995→1200)
MGWR_SCALE = {
    "FS_CBH_t0agg_z":        150,
    "HUM_roaddens_r5km_z":   150,
    "TS_SOC_0_30cm_z":       300,
    "CLIM_pr_sum_pre_z":     300,
    "TS_elev_m_z":           600,
    "FS_TCC_t0_z":           600,
    "TS_slope_deg_z":        600,
    "CLIM_tmmn_mean_pre_z":  1200,
    "HUM_traildens_r10km_z": 1200,
    "HUM_viirs_near_t0_log_z": 1200,
    "HUM_imperv_near_t0_z":  1200,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def moran_i(resid: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    tree = KDTree(coords)
    dists, idxs = tree.query(coords, k=k + 1)
    idxs = idxs[:, 1:]       # exclude self
    n = len(resid)
    z = resid - resid.mean()
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idxs[i]] = 1.0
    W /= W.sum(axis=1, keepdims=True)
    num = n * float(np.einsum("ij,i,j->", W, z, z))
    den = float(W.sum()) * float((z ** 2).sum())
    return num / den if den != 0 else float("nan")


def evaluate(model, X_train, y_train, X_test, y_test, coords_test, label):
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    resid_train = y_train - model.predict(X_train)
    # Moran on train residuals (consistent with block CV approach)
    mi = float("nan")   # skip Moran for speed; compute separately if needed
    print(f"  {label:20s}  R²={r2:.4f}  RMSE={rmse:.4f}")
    return {"label": label, "r2": r2, "rmse": rmse, "moran_i": mi}


def block_split(coords: np.ndarray, block_m: float = 100_000, seed: int = 42):
    bx = np.floor(coords[:, 0] / block_m).astype(int).astype(str)
    by = np.floor(coords[:, 1] / block_m).astype(int).astype(str)
    block_ids = np.char.add(np.char.add(bx, "_"), by)
    unique = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    test_blocks = set(rng.choice(unique, size=max(1, round(len(unique) * 0.2)), replace=False))
    is_test = np.array([b in test_blocks for b in block_ids])
    return is_test


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--response",    default="Resistance")
    p.add_argument("--n-trees",     type=int, default=300)
    p.add_argument("--n-jobs",      type=int, default=-1)
    p.add_argument("--random-state",type=int, default=42)
    p.add_argument("--block-m",     type=float, default=100_000)
    p.add_argument("--col-map",     nargs="*", default=[],
                   help="Column renames: old=new pairs")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    for pair in args.col_map:
        old, new = pair.split("=")
        if old in df.columns:
            df = df.rename(columns={old: new})
    coords = df[["x", "y"]].to_numpy(dtype=float)
    y = df[args.response].to_numpy(dtype=float)

    # ── Feature sets ──────────────────────────────────────────────────────────
    orig_cols = [p for p in PREDICTORS if p in df.columns]

    multi_cols = orig_cols + [
        f"{p}_s{r}km"
        for p in PREDICTORS for r in SCALES_KM
        if f"{p}_s{r}km" in df.columns
    ]

    mgwr_cols = orig_cols + [
        f"{p}_s{MGWR_SCALE[p]}km"
        for p in PREDICTORS
        if f"{p}_s{MGWR_SCALE[p]}km" in df.columns
    ]

    feature_sets = {
        "original":   orig_cols,
        "mgwr_scale": mgwr_cols,
        "multiscale": multi_cols,
    }

    # ── Splits ────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.random_state)
    is_test_rnd = rng.random(len(df)) < 0.2
    is_test_blk = block_split(coords, block_m=args.block_m, seed=args.random_state)

    print(f"n={len(df)}  "
          f"rnd: train={( ~is_test_rnd).sum()} test={is_test_rnd.sum()}  "
          f"blk: train={(~is_test_blk).sum()} test={is_test_blk.sum()}")

    all_results = []
    importance_frames = []

    for name, cols in feature_sets.items():
        X = df[cols].to_numpy(dtype=float)
        print(f"\n[{name}]  {len(cols)} features")

        rf = RandomForestRegressor(
            n_estimators=args.n_trees,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )

        # Random CV
        rf.fit(X[~is_test_rnd], y[~is_test_rnd])
        res_rnd = evaluate(rf,
            X[~is_test_rnd], y[~is_test_rnd],
            X[is_test_rnd],  y[is_test_rnd],
            coords[is_test_rnd], "rand_CV")

        # Block CV
        rf.fit(X[~is_test_blk], y[~is_test_blk])
        res_blk = evaluate(rf,
            X[~is_test_blk], y[~is_test_blk],
            X[is_test_blk],  y[is_test_blk],
            coords[is_test_blk], "block_CV")

        # Moran's I on block residuals
        resid_blk = y[is_test_blk] - rf.predict(X[is_test_blk])
        mi = moran_i(resid_blk, coords[is_test_blk], k=8)
        print(f"  {'Moran_I':20s}  {mi:.4f}")

        all_results.append({
            "feature_set":  name,
            "n_features":   len(cols),
            "r2_randCV":    res_rnd["r2"],
            "rmse_randCV":  res_rnd["rmse"],
            "r2_blkCV":     res_blk["r2"],
            "rmse_blkCV":   res_blk["rmse"],
            "moran_i_blk":  mi,
        })

        # Feature importance (block-trained model)
        imp_df = pd.DataFrame({
            "feature":    cols,
            "importance": rf.feature_importances_,
            "feature_set": name,
        }).sort_values("importance", ascending=False)
        importance_frames.append(imp_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(outdir / "multiscale_rf_summary.csv", index=False)

    imp_all = pd.concat(importance_frames, ignore_index=True)
    imp_all.to_csv(outdir / "multiscale_rf_importance.csv", index=False)

    # Per-predictor best-scale table (from multiscale model)
    imp_ms = imp_all[imp_all.feature_set == "multiscale"].copy()
    imp_ms["predictor"] = imp_ms["feature"].str.replace(r"_s\d+km$", "", regex=True)
    imp_ms["scale_km"]  = imp_ms["feature"].str.extract(r"_s(\d+)km$")[0].fillna("original")
    best_scale = (
        imp_ms[imp_ms["scale_km"] != "original"]
        .sort_values("importance", ascending=False)
        .groupby("predictor")
        .first()
        .reset_index()[["predictor", "scale_km", "importance"]]
    )
    best_scale.to_csv(outdir / "multiscale_best_scale.csv", index=False)

    print("\n=== Summary ===")
    print(results_df[["feature_set", "n_features",
                       "r2_randCV", "r2_blkCV", "moran_i_blk"]].to_string(index=False))
    print(f"\nBest scale per predictor (multiscale model):")
    print(best_scale.to_string(index=False))
    print(f"\nOutputs → {outdir}")


if __name__ == "__main__":
    main()
