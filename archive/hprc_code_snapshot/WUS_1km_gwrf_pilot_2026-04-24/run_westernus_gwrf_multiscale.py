#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KDTree

TODAY = date.today().strftime("%Y-%m-%d")
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_BLOCK_KM = 100.0

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
EVT_PREDS = ["FS_EVT_resistance_proxy_z", "FS_EVT_regeneration_proxy_z"]

FIRE_REGIME_MAP = {
    7053: "surface_fire", 7054: "surface_fire", 7031: "surface_fire",
    7017: "surface_fire", 7019: "surface_fire", 7020: "surface_fire",
    7022: "surface_fire", 7035: "surface_fire", 7036: "surface_fire",
    7063: "surface_fire",
    7050: "stand_replacing", 7055: "stand_replacing", 7056: "stand_replacing",
    7046: "stand_replacing", 7041: "stand_replacing", 7044: "stand_replacing",
    7032: "stand_replacing", 7033: "stand_replacing", 7058: "stand_replacing",
    7061: "stand_replacing", 7062: "stand_replacing", 7113: "stand_replacing",
    7114: "stand_replacing", 7118: "stand_replacing",
    7045: "mixed_severity", 7047: "mixed_severity", 7166: "mixed_severity",
    7051: "mixed_severity", 7018: "mixed_severity", 7027: "mixed_severity",
    7028: "mixed_severity", 7172: "mixed_severity", 7030: "mixed_severity",
    7265: "mixed_severity",
    7037: "coastal_mixed", 7039: "coastal_mixed", 7042: "coastal_mixed",
    7174: "coastal_mixed", 7043: "coastal_mixed", 7014: "coastal_mixed",
    7015: "coastal_mixed",
    7011: "hardwood", 7029: "hardwood", 7010: "hardwood",
    7008: "hardwood", 7021: "hardwood",
}

BASE_TO_Z = {
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
    "CLIM_aridity_pre_z": "CLIM_aridity_pre", "CLIM_tmmx_std_pre_z": "CLIM_tmmx_std_pre",
    "CLIM_vpd_mean_pre_z": "CLIM_vpd_mean_pre", "CLIM_vpd_std_pre_z": "CLIM_vpd_std_pre",
    "FS_EVT_resistance_proxy_z": "FS_EVT_resistance_proxy",
    "FS_EVT_regeneration_proxy_z": "FS_EVT_regeneration_proxy",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--k-spatial", type=int, nargs="+", default=[2000, 5000, 10000])
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--block-km", type=float, default=DEFAULT_BLOCK_KM)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def zscore(s):
    v = pd.to_numeric(s, errors="coerce").astype(float)
    std = v.std(ddof=1)
    return ((v - v.mean()) / std).astype(np.float32) if std > 0 else pd.Series(np.zeros(len(v), dtype=np.float32), index=v.index)

def log1p_z(s):
    return zscore(np.log1p(pd.to_numeric(s, errors="coerce").astype(float).clip(lower=0)))


def ensure_columns(df):
    out = df.copy()
    for z_col, raw_col in BASE_TO_Z.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    for col, expr in [("x_sq_z", out["x"]**2), ("y_sq_z", out["y"]**2), ("xy_z", out["x"]*out["y"])]:
        if col not in out.columns:
            out[col] = zscore(expr)
    regime = out["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    dummies = pd.get_dummies(regime, prefix="EVT_regime").astype(np.float32)
    if "EVT_regime_other" in dummies.columns:
        dummies = dummies.drop(columns=["EVT_regime_other"])
    out = pd.concat([out, dummies], axis=1)
    return out, list(dummies.columns)


def block_groups(df, block_km):
    bm = block_km * 1000.0
    return pd.Series(
        [f"{int(np.floor(x/bm))}_{int(np.floor(y/bm))}" for x, y in zip(df["x"], df["y"])],
        index=df.index,
    )


def predict_one_local(i, test_coords_i, test_X_i, train_coords, train_X, train_y, k_spatial, n_trees):
    """Fit a local RF using k_spatial nearest training neighbors, return prediction."""
    dist, idx = kdtree_query(train_coords, test_coords_i.reshape(1, -1), k=k_spatial)
    local_X = train_X[idx[0]]
    local_y = train_y[idx[0]]
    # Gaussian kernel weights based on max distance in neighborhood
    bw = dist[0].max()
    weights = np.exp(-0.5 * (dist[0] / (bw + 1e-10)) ** 2) if bw > 0 else np.ones(k_spatial)
    weights = weights / weights.sum()
    m = RandomForestRegressor(n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=1)
    m.fit(local_X, local_y, sample_weight=weights)
    return float(m.predict(test_X_i.reshape(1, -1))[0])


# global kdtree reference (set before parallel)
_kdtree = None
_train_coords = None

def kdtree_query(train_coords, q, k):
    tree = KDTree(train_coords)
    return tree.query(q, k=min(k, len(train_coords)))


def run_gwrf(train_df, test_df, predictors, k_spatial, n_trees, n_jobs):
    train_coords = train_df[["x", "y"]].to_numpy(dtype=float)
    test_coords = test_df[["x", "y"]].to_numpy(dtype=float)
    train_X = train_df[predictors].to_numpy(dtype=float)
    train_y = train_df["Resistance"].to_numpy(dtype=float)
    test_X = test_df[predictors].to_numpy(dtype=float)

    tree = KDTree(train_coords)
    k = min(k_spatial, len(train_df))

    def _predict_one(i):
        dist, idx = tree.query(test_coords[i:i+1], k=k)
        local_X = train_X[idx[0]]
        local_y = train_y[idx[0]]
        bw = dist[0].max()
        w = np.exp(-0.5 * (dist[0] / (bw + 1e-10)) ** 2) if bw > 0 else np.ones(k)
        w /= w.sum()
        m = RandomForestRegressor(n_estimators=n_trees, random_state=RANDOM_STATE, n_jobs=1)
        m.fit(local_X, local_y, sample_weight=w)
        return float(m.predict(test_X[i:i+1])[0])

    preds = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_predict_one)(i) for i in range(len(test_df)))
    y_test = test_df["Resistance"].to_numpy()
    return {
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }, np.array(preds)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    df, evt_cols = ensure_columns(df)
    predictors = [c for c in BASE_PREDS + EVT_PREDS + evt_cols if c in df.columns]

    work = (
        df[list(dict.fromkeys(["Resistance", "x", "y"] + predictors))]
        .replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    )
    print(f"Rows: {len(work)}, Predictors: {len(predictors)}", flush=True)

    idx = np.arange(len(work))
    rand_tr, rand_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    groups = block_groups(work, args.block_km)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    blk_tr, blk_te = next(gss.split(idx, groups=groups))

    rows = []

    # Global RF baseline (variant E)
    print("Fitting global RF...", flush=True)
    for split_name, (tr, te) in [("random", (rand_tr, rand_te)), ("block", (blk_tr, blk_te))]:
        train, test = work.iloc[tr], work.iloc[te]
        m = RandomForestRegressor(n_estimators=args.trees * 3, random_state=RANDOM_STATE, n_jobs=args.n_jobs)
        m.fit(train[predictors], train["Resistance"])
        pred = m.predict(test[predictors])
        y = test["Resistance"].to_numpy()
        rows.append({
            "model": "global_rf_E",
            "k_spatial": None,
            "split": split_name,
            "r2": float(r2_score(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        })
        print(f"  global_rf {split_name}: R2={rows[-1]['r2']:.4f}", flush=True)

    # GWRF with different spatial bandwidths — block split only (the honest one)
    for k in args.k_spatial:
        print(f"\nGWRF k_spatial={k} (block split)...", flush=True)
        train_b, test_b = work.iloc[blk_tr], work.iloc[blk_te]
        metrics, preds_arr = run_gwrf(train_b, test_b, predictors, k, args.trees, args.n_jobs)
        rows.append({"model": "gwrf", "k_spatial": k, "split": "block", **metrics})
        print(f"  GWRF k={k} block: R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f}", flush=True)

        # save per-point predictions for spatial residual analysis
        out_df = test_b[["x", "y", "Resistance"]].copy()
        out_df["pred"] = preds_arr
        out_df["residual"] = out_df["Resistance"] - out_df["pred"]
        out_df.to_csv(args.output_dir / f"gwrf_k{k}_block_predictions.csv", index=False)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output_dir / "gwrf_multiscale_metrics.csv", index=False)

    report = [
        f"# Multiscale GW-RF Pilot ({TODAY})",
        "",
        "| Model | k_spatial | Split | R2 | RMSE |",
        "|---|---|---|---|---|",
    ]
    for _, r in results_df.iterrows():
        report.append(f"| {r['model']} | {r['k_spatial']} | {r['split']} | {r['r2']:.4f} | {r['rmse']:.4f} |")
    report.append("")
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(json.dumps({"out_dir": str(args.output_dir), "rows": len(work)}))


if __name__ == "__main__":
    main()
