#!/usr/bin/env python3
"""
Leak-free MGWR-scale RF comparison on Model-2 predictor set.

Three feature sets, each evaluated with random and 100-km block CV:
  base29        — 27 baseline + 2 EVT proxies (= Model 2)
  base29+mgwr   — base29 + 11 MGWR-bandwidth-matched smooth features
  base29+wrong  — base29 + 11 uniform-300km smooth features (control)

Key design: smooth features are computed INSIDE each CV fold using
training data only; test points are interpolated from the training
KDTree — no spatial leakage.

Input:  westernus_current_candidate_table_plus_regions.parquet
Output: results/
  rf_model2_compare_summary.csv
  rf_model2_compare_details.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ── Predictor definitions (mirrors foresttype_encoding_compare.py) ─────────────

BASE_PREDS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z",
    "TS_twi_z", "TS_roughness_z", "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z", "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_z", "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_pre_z", "CLIM_eto_sum_pre_z", "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z", "CLIM_aridity_pre_z", "CLIM_tmmx_std_pre_z",
    "CLIM_vpd_mean_pre_z", "CLIM_vpd_std_pre_z",
    "x", "y", "x_sq_z", "y_sq_z", "xy_z",
]
EVT_PROXY_PREDS = ["FS_EVT_resistance_proxy_z", "FS_EVT_regeneration_proxy_z"]
BASE29 = BASE_PREDS + EVT_PROXY_PREDS

# 11 MGWR stage5b predictors → MGWR-bandwidth-matched scale (km)
MGWR_SMOOTH = {
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
WRONG_SCALE_KM = 300   # uniform scale used as control


# ── Data preparation (same logic as foresttype_encoding_compare.py) ────────────

def zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    std = v.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(v), dtype=np.float32), index=v.index)
    return ((v - v.mean()) / std).astype(np.float32)


def log1p_z(s: pd.Series) -> pd.Series:
    return zscore(np.log1p(pd.to_numeric(s, errors="coerce").astype(float).clip(lower=0)))


RAW_TO_Z = {
    "TS_elev_m_z": "TS_elev_m", "TS_slope_deg_z": "TS_slope_deg",
    "TS_northness_z": "TS_northness", "TS_eastness_z": "TS_eastness",
    "TS_twi_z": "TS_twi", "TS_roughness_z": "TS_roughness",
    "TS_SOC_0_30cm_z": "TS_SOC_0_30cm", "FS_TCC_t0_z": "FS_TCC_t0",
    "FS_CBH_t0agg_z": "FS_CBH_t0agg", "HUM_roaddens_r5km_z": "HUM_roaddens_r5km",
    "HUM_traildens_r10km_z": "HUM_traildens_r10km",
    "HUM_imperv_near_t0_z": "HUM_imperv_near_t0",
    "CLIM_pr_sum_pre_z": "CLIM_pr_sum_pre", "CLIM_eto_sum_pre_z": "CLIM_eto_sum_pre",
    "CLIM_tmmn_mean_pre_z": "CLIM_tmmn_mean_pre",
    "CLIM_hot_days_35C_pre_z": "CLIM_hot_days_35C_pre",
    "CLIM_aridity_pre_z": "CLIM_aridity_pre",
    "CLIM_tmmx_std_pre_z": "CLIM_tmmx_std_pre",
    "CLIM_vpd_mean_pre_z": "CLIM_vpd_mean_pre",
    "CLIM_vpd_std_pre_z": "CLIM_vpd_std_pre",
    "FS_EVT_resistance_proxy_z": "FS_EVT_resistance_proxy",
    "FS_EVT_regeneration_proxy_z": "FS_EVT_regeneration_proxy",
}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    for z_col, raw_col in RAW_TO_Z.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out.get("HUM_popdens_win10km", pd.Series(dtype=float)))
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out.get("HUM_viirs_near_t0", pd.Series(dtype=float)))
    for col, expr in [
        ("x_sq_z", out["x"] ** 2),
        ("y_sq_z", out["y"] ** 2),
        ("xy_z",   out["x"] * out["y"]),
    ]:
        if col not in out.columns:
            out[col] = zscore(expr)
    return out


# ── Leak-free Gaussian smooth ──────────────────────────────────────────────────

def smooth_train_test(
    train_coords: np.ndarray,
    train_X: np.ndarray,
    test_coords: np.ndarray,
    radius_m: float,
    max_k: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    tree = KDTree(train_coords)

    def _apply(dists, idxs):
        w = np.exp(-0.5 * (dists / radius_m) ** 2)
        w[dists > 2.5 * radius_m] = 0.0
        w_sum = w.sum(axis=1)
        no_nb = w_sum == 0
        w_sum = np.where(no_nb, 1.0, w_sum)
        result = np.einsum("ij,ijp->ip", w, train_X[idxs]) / w_sum[:, None]
        result[no_nb] = train_X[idxs[no_nb, 0]]
        return result

    k_tr = min(len(train_coords) - 1, max_k)
    d_tr, i_tr = tree.query(train_coords, k=k_tr + 1)
    s_train = _apply(d_tr[:, 1:], i_tr[:, 1:])

    k_te = min(len(train_coords), max_k)
    d_te, i_te = tree.query(test_coords, k=k_te)
    s_test = _apply(d_te, i_te)

    return s_train, s_test


# ── CV helpers ─────────────────────────────────────────────────────────────────

def block_split(coords: np.ndarray, block_m: float, seed: int) -> np.ndarray:
    bx = np.floor(coords[:, 0] / block_m).astype(int).astype(str)
    by = np.floor(coords[:, 1] / block_m).astype(int).astype(str)
    bids = np.char.add(np.char.add(bx, "_"), by)
    unique = np.unique(bids)
    test_set = set(np.random.default_rng(seed).choice(
        unique, size=max(1, round(len(unique) * 0.2)), replace=False))
    return np.array([b in test_set for b in bids])


def build_features(
    feature_set: str,
    tr: np.ndarray, te: np.ndarray,
    coords: np.ndarray,
    X_base: np.ndarray,
    X_mgwr: np.ndarray,
    mgwr_preds: list[str],
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    c_tr, c_te = coords[tr], coords[te]
    Xb_tr, Xb_te = X_base[tr], X_base[te]
    Xm_tr = X_mgwr[tr]

    if feature_set == "base29":
        return Xb_tr, Xb_te

    elif feature_set == "base29+mgwr":
        # Group by MGWR scale
        scale_groups: dict[int, list[int]] = {}
        for j, p in enumerate(mgwr_preds):
            r = MGWR_SMOOTH[p]
            scale_groups.setdefault(r, []).append(j)
        parts_tr, parts_te = [Xb_tr], [Xb_te]
        for r_km, js in sorted(scale_groups.items()):
            s_tr, s_te = smooth_train_test(c_tr, Xm_tr[:, js], c_te, r_km * 1000.0, max_k)
            parts_tr.append(s_tr)
            parts_te.append(s_te)
        return np.hstack(parts_tr), np.hstack(parts_te)

    else:  # base29+wrong
        s_tr, s_te = smooth_train_test(c_tr, Xm_tr, c_te, WRONG_SCALE_KM * 1000.0, max_k)
        return np.hstack([Xb_tr, s_tr]), np.hstack([Xb_te, s_te])


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
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...", flush=True)
    raw = pd.read_parquet(args.input)
    df  = prepare(raw)

    avail_base29 = [p for p in BASE29 if p in df.columns]
    avail_mgwr   = [p for p in MGWR_SMOOTH if p in df.columns]
    needed = list(dict.fromkeys([args.response, "x", "y"] + avail_base29 + avail_mgwr))
    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    print(f"n={len(work)}  base29={len(avail_base29)}  mgwr_preds={len(avail_mgwr)}", flush=True)

    coords = work[["x", "y"]].to_numpy(dtype=float)
    y      = work[args.response].to_numpy(dtype=float)
    X_base = work[avail_base29].to_numpy(dtype=float)
    X_mgwr = work[avail_mgwr].to_numpy(dtype=float)

    rng = np.random.default_rng(args.random_state)
    is_test_rnd = rng.random(len(work)) < 0.2
    is_test_blk = block_split(coords, args.block_m, args.random_state)

    print(f"rnd: train={(~is_test_rnd).sum()} test={is_test_rnd.sum()}"
          f"  blk: train={(~is_test_blk).sum()} test={is_test_blk.sum()}\n", flush=True)

    rf = RandomForestRegressor(
        n_estimators=args.n_trees, n_jobs=args.n_jobs, random_state=args.random_state
    )

    rows = []
    for fs in ["base29", "base29+mgwr", "base29+wrong"]:
        print(f"[{fs}]", flush=True)
        for cv_name, tr_mask, te_mask in [
            ("randCV", ~is_test_rnd, is_test_rnd),
            ("blkCV",  ~is_test_blk, is_test_blk),
        ]:
            print(f"  {cv_name} smoothing ...", flush=True)
            X_tr, X_te = build_features(
                fs, tr_mask, te_mask, coords,
                X_base, X_mgwr, avail_mgwr, args.max_k
            )
            print(f"  {cv_name} RF fit ({X_tr.shape[1]} features) ...", flush=True)
            rf.fit(X_tr, y[tr_mask])
            y_pred = rf.predict(X_te)
            r2   = float(r2_score(y[te_mask], y_pred))
            rmse = float(np.sqrt(np.mean((y[te_mask] - y_pred) ** 2)))
            print(f"  {cv_name}: R²={r2:.4f}  RMSE={rmse:.4f}", flush=True)
            rows.append({"feature_set": fs, "cv": cv_name,
                         "n_features": X_tr.shape[1], "r2": r2, "rmse": rmse})
        print()

    out = pd.DataFrame(rows)
    out.to_csv(outdir / "rf_model2_compare_details.csv", index=False)

    wide = out.pivot(index="feature_set", columns="cv", values=["r2", "rmse", "n_features"])
    wide.columns = ["_".join(c) for c in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(outdir / "rf_model2_compare_summary.csv", index=False)

    print("=== Summary ===")
    print(wide[["feature_set", "n_features_blkCV", "r2_randCV", "r2_blkCV", "rmse_blkCV"]]
          .to_string(index=False))
    print(f"\nOutputs → {outdir}")


if __name__ == "__main__":
    main()
