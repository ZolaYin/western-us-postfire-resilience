#!/usr/bin/env python3
"""
run_block_transfer_exp_v1.py
Goal: beat M2 block CV R² = 0.2960 via region / no-xy / large-scale / XGBoost.

Variants
--------
m2_baseline    : reproduce M2 (verify consistency)
m2_region      : M2 + 5-class geographic region dummies     [NEW]
m2_no_xy       : M2 without spatial polynomial terms         [NEW]
m2_region_no_xy: M2 - spatial poly + region                  [NEW]
large_region   : large-bandwidth preds + EVT + region        [NEW]
xgb_m2         : XGBoost on M2 predictors                    [NEW]
xgb_m2_region  : XGBoost on M2 + region                      [NEW]

Block CV: 100-km grid cells, GroupShuffleSplit 80/20 (consistent with prior runs).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed – XGB variants will be skipped")

# ── paths ──────────────────────────────────────────────────────────────────────
SCRATCH = Path("/scratch/user/YOUR_NETID")
LOCAL   = Path("/path/to/google-drive"
               "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km")
ROOT    = SCRATCH if SCRATCH.exists() else LOCAL
INPUT   = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
TODAY   = date.today().strftime("%Y-%m-%d")

RANDOM_STATE = 42
TEST_SIZE    = 0.20
BLOCK_KM     = 100.0
N_TREES      = 300
MORAN_K      = 8

# ── predictor definitions ──────────────────────────────────────────────────────
# Full M2 baseline continuous predictors (27 total, same as prior runs)
BASE_PREDS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z",
    "TS_twi_z", "TS_roughness_z", "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z", "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_z", "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_pre_z", "CLIM_eto_sum_pre_z", "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z", "CLIM_aridity_pre_z",
    "CLIM_tmmx_std_pre_z", "CLIM_vpd_mean_pre_z", "CLIM_vpd_std_pre_z",
    "x", "y", "x_sq_z", "y_sq_z", "xy_z",
]
SPACE_COLS = {"x", "y", "x_sq_z", "y_sq_z", "xy_z"}

# Large-bandwidth predictors only (MGWR bandwidth > 400 km)
LARGE_SCALE_PREDS = [
    "FS_TCC_t0_z",          # ~400 km
    "TS_slope_deg_z",       # ~540 km
    "CLIM_tmmn_mean_pre_z", # ~730 km
    "HUM_traildens_r10km_z",# ~840 km
    "HUM_viirs_near_t0_log_z",  # ~1350 km
    "HUM_imperv_near_t0_z",     # ~1380 km
    # extra climate (large-scale by nature)
    "CLIM_aridity_pre_z", "CLIM_vpd_mean_pre_z", "CLIM_eto_sum_pre_z",
]

RAW_TO_Z = {
    "TS_elev_m_z":       "TS_elev_m",
    "TS_slope_deg_z":    "TS_slope_deg",
    "TS_northness_z":    "TS_northness",
    "TS_eastness_z":     "TS_eastness",
    "TS_twi_z":          "TS_twi",
    "TS_roughness_z":    "TS_roughness",
    "TS_SOC_0_30cm_z":   "TS_SOC_0_30cm",
    "FS_TCC_t0_z":       "FS_TCC_t0",
    "FS_CBH_t0agg_z":    "FS_CBH_t0agg",
    "HUM_roaddens_r5km_z":     "HUM_roaddens_r5km",
    "HUM_traildens_r10km_z":   "HUM_traildens_r10km",
    "HUM_imperv_near_t0_z":    "HUM_imperv_near_t0",
    "CLIM_pr_sum_pre_z":    "CLIM_pr_sum_pre",
    "CLIM_eto_sum_pre_z":   "CLIM_eto_sum_pre",
    "CLIM_tmmn_mean_pre_z": "CLIM_tmmn_mean_pre",
    "CLIM_hot_days_35C_pre_z": "CLIM_hot_days_35C_pre",
    "CLIM_aridity_pre_z":   "CLIM_aridity_pre",
    "CLIM_tmmx_std_pre_z":  "CLIM_tmmx_std_pre",
    "CLIM_vpd_mean_pre_z":  "CLIM_vpd_mean_pre",
    "CLIM_vpd_std_pre_z":   "CLIM_vpd_std_pre",
}

# ── feature engineering ────────────────────────────────────────────────────────
def zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    std = v.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(v), dtype=np.float32), index=v.index)
    return ((v - v.mean()) / std).astype(np.float32)

def log1p_z(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float).clip(lower=0)
    return zscore(np.log1p(v))

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    for z_col, raw_col in RAW_TO_Z.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])

    if "HUM_popdens_win10km_log_z" not in out.columns and "HUM_popdens_win10km" in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns and "HUM_viirs_near_t0" in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])

    for col, expr in [("x_sq_z", out["x"]**2), ("y_sq_z", out["y"]**2), ("xy_z", out["x"]*out["y"])]:
        if col not in out.columns:
            out[col] = zscore(expr)

    # EVT group class dummies
    evt_clean = out["FS_EVT_group_class"].astype("string").fillna("unknown").astype(str)
    out = pd.concat([out, pd.get_dummies(evt_clean, prefix="EVT_group", dtype=np.float32)], axis=1)

    # Region dummies  (NEW)
    if "region" in out.columns:
        reg_clean = out["region"].astype("string").fillna("unknown").astype(str)
        out = pd.concat([out, pd.get_dummies(reg_clean, prefix="REG", dtype=np.float32)], axis=1)

    return out

# ── block split ────────────────────────────────────────────────────────────────
def block_groups(df: pd.DataFrame, block_km: float) -> pd.Series:
    block_m = block_km * 1000.0
    labels = [f"{int(np.floor(x/block_m))}_{int(np.floor(y/block_m))}"
              for x, y in zip(df["x"], df["y"])]
    return pd.Series(labels, index=df.index)

def build_masks(work: pd.DataFrame, block_km: float, rs: int) -> dict[str, np.ndarray]:
    idx = np.arange(len(work))
    _, rnd_te = train_test_split(idx, test_size=TEST_SIZE, random_state=rs)
    groups = block_groups(work, block_km)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=rs)
    _, blk_te = next(gss.split(idx, groups=groups))
    is_rnd = np.zeros(len(work), bool); is_rnd[rnd_te] = True
    is_blk = np.zeros(len(work), bool); is_blk[blk_te] = True
    return {"random": is_rnd, "block": is_blk}

# ── Moran's I ──────────────────────────────────────────────────────────────────
def moran_i(res: np.ndarray, coords: np.ndarray, k: int = MORAN_K) -> float:
    if len(res) <= 1: return float("nan")
    tree = KDTree(coords)
    k_eff = min(k, len(res) - 1)
    _, idx = tree.query(coords, k=k_eff + 1); idx = idx[:, 1:]
    z = res - res.mean()
    n = len(res)
    W = np.zeros((n, n), np.float32)
    for i in range(n): W[i, idx[i]] = 1.0
    rs = W.sum(1, keepdims=True); rs[rs == 0] = 1.0; W /= rs
    num = n * float(np.einsum("ij,i,j->", W, z, z))
    den = float(W.sum()) * float((z**2).sum())
    return num / den if den else float("nan")

# ── variant feature sets ───────────────────────────────────────────────────────
def evt_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("EVT_group_")]

def reg_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("REG_")]

def variant_features(name: str, df: pd.DataFrame) -> list[str]:
    base = [c for c in BASE_PREDS if c in df.columns]
    base_no_xy = [c for c in base if c not in SPACE_COLS]
    large = [c for c in LARGE_SCALE_PREDS if c in df.columns]
    evt = evt_cols(df); reg = reg_cols(df)

    specs = {
        "m2_baseline":     base + evt,
        "m2_region":       base + evt + reg,
        "m2_no_xy":        base_no_xy + evt,
        "m2_region_no_xy": base_no_xy + evt + reg,
        "large_region":    large + evt + reg,
        "xgb_m2":          base + evt,
        "xgb_m2_region":   base + evt + reg,
    }
    return specs[name]

VARIANTS = [
    "m2_baseline", "m2_region", "m2_no_xy", "m2_region_no_xy",
    "large_region", "xgb_m2", "xgb_m2_region",
]

# ── model factory ──────────────────────────────────────────────────────────────
def make_model(name: str, n_jobs: int):
    if name.startswith("xgb_"):
        if not HAS_XGB:
            return None
        return XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            reg_lambda=3.0, reg_alpha=0.5,
            tree_method="hist", n_jobs=n_jobs, random_state=RANDOM_STATE,
        )
    return RandomForestRegressor(
        n_estimators=N_TREES, max_features="sqrt",
        min_samples_leaf=5, n_jobs=n_jobs, random_state=RANDOM_STATE,
    )

# ── evaluation ─────────────────────────────────────────────────────────────────
def evaluate(model, Xtr, ytr, Xte, yte, coords_te) -> dict:
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    res  = yte - pred
    return {
        "r2":      round(float(r2_score(yte, pred)), 4),
        "rmse":    round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
        "moran_i": round(float(moran_i(res, coords_te)), 4),
    }

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response",   default="Resistance")
    parser.add_argument("--n-jobs",     type=int, default=-1)
    parser.add_argument("--sample-n",   type=int, default=None)
    parser.add_argument("--variants",   nargs="+", default=VARIANTS)
    args = parser.parse_args()

    out_dir = args.output_dir; out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} …")
    raw = pd.read_parquet(args.input)
    if args.sample_n:
        raw = raw.sample(args.sample_n, random_state=RANDOM_STATE)
        print(f"  Subsampled to {len(raw)} rows")

    print("Preparing features …")
    df = prepare(raw)
    df = df.dropna(subset=[args.response]).reset_index(drop=True)
    print(f"  After dropna: {len(df)} rows")
    print(f"  EVT dummies: {evt_cols(df)}")
    print(f"  Region dummies: {reg_cols(df)}")

    y = df[args.response].to_numpy(np.float32)
    coords = df[["x", "y"]].to_numpy(np.float64)
    masks = build_masks(df, BLOCK_KM, RANDOM_STATE)

    rows = []
    for vname in args.variants:
        if vname not in VARIANTS:
            print(f"  SKIP unknown variant: {vname}"); continue
        feats = variant_features(vname, df)
        feats = [f for f in feats if f in df.columns]
        X = df[feats].to_numpy(np.float32)
        model = make_model(vname, args.n_jobs)
        if model is None:
            print(f"  SKIP {vname} (XGBoost not available)"); continue

        print(f"\n── {vname}  ({len(feats)} features) ──")
        for split_name, is_test in masks.items():
            is_tr = ~is_test
            Xtr, ytr, ctr = X[is_tr], y[is_tr], coords[is_tr]
            Xte, yte, cte = X[is_test], y[is_test], coords[is_test]
            metrics = evaluate(model, Xtr, ytr, Xte, yte, cte)
            print(f"  {split_name:8s}  R²={metrics['r2']:.4f}  "
                  f"RMSE={metrics['rmse']:.4f}  Moran's I={metrics['moran_i']:.4f}")
            rows.append({"variant": vname, "split": split_name,
                         "n_features": len(feats), **metrics})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "metrics_summary.csv", index=False)

    # pivot for easy reading
    wide = results.pivot_table(index="variant", columns="split",
                               values=["r2","rmse","moran_i"], aggfunc="first")
    wide.columns = [f"{v}_{s}" for v, s in wide.columns]
    wide.to_csv(out_dir / "metrics_wide.csv")
    print(f"\n── Block CV results ──")
    print(wide[["r2_block","rmse_block","moran_i_block"]].sort_values("r2_block", ascending=False).to_string())

    # write report
    report_lines = [
        f"# Block Transfer Experiments v1 ({TODAY})\n",
        f"- Input: `{args.input.name}`",
        f"- Rows: {len(df):,}",
        f"- Block CV: {BLOCK_KM} km grid, GroupShuffleSplit 80/20\n",
        "## Block CV Results\n",
        wide[["r2_block","rmse_block","moran_i_block"]]
            .sort_values("r2_block", ascending=False).to_string(),
        "\n\n## Random CV Results\n",
        wide[["r2_random","rmse_random","moran_i_random"]]
            .sort_values("r2_random", ascending=False).to_string(),
        "\n\n## All rows\n",
        results.to_string(index=False),
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines))
    print(f"\nSaved → {out_dir}")

if __name__ == "__main__":
    main()
