#!/usr/bin/env python3
"""
Multiscale RF with leak-free CV.

Previous version (v1) computed smooth features from ALL data, causing
spatial leakage: training points' smoothed values included test-block
neighbors → blkCV deflated.

Fix: compute smooth features INSIDE each CV fold:
  - Train smooth: Gaussian-weighted avg of training-only neighbors
  - Test  smooth: IDW from training KDTree → same info a deployed
                  model would actually have

Feature sets compared:
  original   — 11 raw z-scored predictors
  mgwr_scale — original + 1 MGWR-matched smooth per predictor (22 total)
  multiscale — original + 4 smooths per predictor at 150/300/600/1200 km (55 total)

Usage:
  python run_rf_multiscale_v2.py \
    --input    GWR_MGWR_ready_table_corrected_noevt15.parquet \
    --output-dir results/rf_multiscale_v2 \
    --response Resistance
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ── Config ─────────────────────────────────────────────────────────────────────

PREDICTORS = [
    "FS_CBH_t0agg_z", "HUM_roaddens_r5km_z", "TS_SOC_0_30cm_z",
    "CLIM_pr_sum_pre_z", "TS_elev_m_z", "FS_TCC_t0_z",
    "TS_slope_deg_z", "CLIM_tmmn_mean_pre_z", "HUM_traildens_r10km_z",
    "HUM_viirs_near_t0_log_z", "HUM_imperv_near_t0_z",
]

# MGWR stage5b bandwidth → nearest smoothing radius (km)
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

ALL_SCALES_KM = [150, 300, 600, 1200]


# ── Smoothing (leak-free) ──────────────────────────────────────────────────────

def smooth_train_test(
    train_coords: np.ndarray,   # (n_tr, 2)
    train_X:      np.ndarray,   # (n_tr, p)
    test_coords:  np.ndarray,   # (n_te, 2)
    radius_m:     float,
    max_k:        int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leak-free Gaussian smooth.
    - Train points: smooth from training neighbors only (exclude self)
    - Test  points: smooth from training neighbors
    Returns (smooth_train, smooth_test) each shape (n, p).
    """
    tree = KDTree(train_coords)
    p = train_X.shape[1]

    def _weighted(dists, idxs, X_src):
        w = np.exp(-0.5 * (dists / radius_m) ** 2)
        w[dists > 2.5 * radius_m] = 0.0
        w_sum = w.sum(axis=1)
        # points with no neighbors in range → fallback to nearest
        no_nb = w_sum == 0
        w_sum = np.where(no_nb, 1.0, w_sum)
        # vectorised: (n, k, p) → (n, p)
        result = np.einsum("ij,ijp->ip", w, X_src[idxs]) / w_sum[:, None]
        if no_nb.any():
            result[no_nb] = X_src[idxs[no_nb, 0]]
        return result

    # training: query k+1, drop self (idx 0)
    k_tr = min(len(train_coords) - 1, max_k)
    d_tr, i_tr = tree.query(train_coords, k=k_tr + 1)
    s_train = _weighted(d_tr[:, 1:], i_tr[:, 1:], train_X)

    # test: query k from training tree
    k_te = min(len(train_coords), max_k)
    d_te, i_te = tree.query(test_coords, k=k_te)
    s_test = _weighted(d_te, i_te, train_X)

    return s_train, s_test


# ── Helpers ────────────────────────────────────────────────────────────────────

def moran_i(resid: np.ndarray, coords: np.ndarray, k: int = 8) -> float:
    tree = KDTree(coords)
    _, idxs = tree.query(coords, k=k + 1)
    idxs = idxs[:, 1:]
    n = len(resid)
    z = resid - resid.mean()
    W = np.zeros((n, n))
    for i in range(n):
        W[i, idxs[i]] = 1.0
    W /= W.sum(axis=1, keepdims=True)
    num = n * float(np.einsum("ij,i,j->", W, z, z))
    den = float(W.sum()) * float((z ** 2).sum())
    return num / den if den != 0 else float("nan")


def block_ids(coords: np.ndarray, block_m: float) -> np.ndarray:
    bx = np.floor(coords[:, 0] / block_m).astype(int).astype(str)
    by = np.floor(coords[:, 1] / block_m).astype(int).astype(str)
    return np.char.add(np.char.add(bx, "_"), by)


def run_cv(
    feature_set: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    coords: np.ndarray,
    X_raw: np.ndarray,        # (n, p) original predictors
    y: np.ndarray,
    preds: list[str],
    rf: RandomForestRegressor,
    max_k: int,
) -> dict:
    tr, te = train_mask, test_mask
    n_tr, n_te = tr.sum(), te.sum()
    coords_tr, coords_te = coords[tr], coords[te]
    X_raw_tr, X_raw_te = X_raw[tr], X_raw[te]
    y_tr, y_te = y[tr], y[te]

    if feature_set == "original":
        X_tr, X_te = X_raw_tr, X_raw_te

    elif feature_set == "mgwr_scale":
        # Group predictors by their MGWR scale
        scale_groups: dict[int, list[int]] = {}
        for j, p in enumerate(preds):
            r = MGWR_SCALE.get(p, 150)
            scale_groups.setdefault(r, []).append(j)

        parts_tr, parts_te = [X_raw_tr], [X_raw_te]
        for r_km, js in sorted(scale_groups.items()):
            r_m = r_km * 1000.0
            s_tr, s_te = smooth_train_test(
                coords_tr, X_raw_tr[:, js],
                coords_te, r_m, max_k=max_k,
            )
            parts_tr.append(s_tr)
            parts_te.append(s_te)
        X_tr = np.hstack(parts_tr)
        X_te = np.hstack(parts_te)

    else:  # multiscale
        parts_tr, parts_te = [X_raw_tr], [X_raw_te]
        for r_km in ALL_SCALES_KM:
            r_m = r_km * 1000.0
            s_tr, s_te = smooth_train_test(
                coords_tr, X_raw_tr,
                coords_te, r_m, max_k=max_k,
            )
            parts_tr.append(s_tr)
            parts_te.append(s_te)
        X_tr = np.hstack(parts_tr)
        X_te = np.hstack(parts_te)

    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    r2 = float(r2_score(y_te, y_pred))
    rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
    return {"r2": r2, "rmse": rmse, "n_features": X_tr.shape[1]}


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",        required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--response",     default="Resistance")
    p.add_argument("--n-trees",      type=int,   default=300)
    p.add_argument("--n-jobs",       type=int,   default=-1)
    p.add_argument("--random-state", type=int,   default=42)
    p.add_argument("--block-m",      type=float, default=100_000)
    p.add_argument("--max-k",        type=int,   default=5000)
    p.add_argument("--col-map",      nargs="*",  default=[])
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

    preds = [p for p in PREDICTORS if p in df.columns]
    missing = [p for p in PREDICTORS if p not in df.columns]
    if missing:
        print(f"Warning: missing predictors: {missing}")

    coords = df[["x", "y"]].to_numpy(dtype=float)
    X_raw  = df[preds].fillna(0).to_numpy(dtype=float)
    y      = df[args.response].to_numpy(dtype=float)
    n      = len(df)

    # ── Splits ────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.random_state)
    is_test_rnd = rng.random(n) < 0.2

    bids   = block_ids(coords, args.block_m)
    unique = np.unique(bids)
    test_b = set(np.random.default_rng(args.random_state).choice(
        unique, size=max(1, round(len(unique) * 0.2)), replace=False))
    is_test_blk = np.array([b in test_b for b in bids])

    print(f"n={n}  rnd: train={( ~is_test_rnd).sum()} test={is_test_rnd.sum()}"
          f"  blk: train={(~is_test_blk).sum()} test={is_test_blk.sum()}\n")

    rf = RandomForestRegressor(
        n_estimators=args.n_trees,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )

    rows = []
    for fs in ["original", "mgwr_scale", "multiscale"]:
        print(f"[{fs}]", flush=True)
        for cv_name, tr_mask, te_mask in [
            ("randCV", ~is_test_rnd, is_test_rnd),
            ("blkCV",  ~is_test_blk, is_test_blk),
        ]:
            print(f"  {cv_name} ...", flush=True)
            res = run_cv(fs, tr_mask, te_mask, coords, X_raw, y,
                         preds, rf, args.max_k)
            print(f"  {cv_name}: R²={res['r2']:.4f}  RMSE={res['rmse']:.4f}"
                  f"  ({res['n_features']} features)", flush=True)
            rows.append({"feature_set": fs, "cv": cv_name, **res})

        # Moran's I on block residuals (block-trained model)
        res_blk = run_cv(fs, ~is_test_blk, is_test_blk, coords,
                         X_raw, y, preds, rf, args.max_k)
        # re-predict to get residuals
        print(f"  Moran's I ...", flush=True)

    out = pd.DataFrame(rows)
    # Pivot to wide format
    wide = out.pivot(index="feature_set", columns="cv",
                     values=["r2", "rmse", "n_features"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    print("\n=== Summary ===")
    print(wide[["feature_set", "n_features_blkCV",
                "r2_randCV", "r2_blkCV", "rmse_blkCV"]].to_string(index=False))

    out.to_csv(outdir / "multiscale_rf_v2_results.csv", index=False)
    wide.to_csv(outdir / "multiscale_rf_v2_summary.csv", index=False)
    print(f"\nOutputs → {outdir}")


if __name__ == "__main__":
    main()
