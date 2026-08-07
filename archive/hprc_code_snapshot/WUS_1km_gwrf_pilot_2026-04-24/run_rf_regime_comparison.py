#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

TODAY = date.today().strftime("%Y-%m-%d")
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_BLOCK_KM = 100.0

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
SPACE_COLS = {"x", "y", "x_sq_z", "y_sq_z", "xy_z"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--regime-assignments", type=Path, required=True)
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--block-km", type=float, default=DEFAULT_BLOCK_KM)
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


def fit_eval(train, test, predictors, n_estimators):
    y_train = train["Resistance"].to_numpy()
    y_test = test["Resistance"].to_numpy()
    m = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(train[predictors], y_train)
    pred = m.predict(test[predictors])
    return {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
    }, m


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    df, evt_regime_cols = ensure_columns(df)

    # merge latent regime assignments
    assignments = pd.read_csv(args.regime_assignments)[["pixel_id", "kmeans_k6"]]
    if "pixel_id" in df.columns:
        df = df.merge(assignments, on="pixel_id", how="left")
    else:
        df["kmeans_k6"] = assignments["kmeans_k6"].values

    # latent regime dummies (drop R4 which has only 15 points - merge into R3)
    df["kmeans_k6_clean"] = df["kmeans_k6"].replace({4: 3})
    latent_dummies = pd.get_dummies(df["kmeans_k6_clean"], prefix="latent_r").astype(np.float32)
    latent_dummies = latent_dummies.drop(columns=[latent_dummies.columns[0]])  # drop first
    df = pd.concat([df, latent_dummies], axis=1)
    latent_dummy_cols = list(latent_dummies.columns)

    # coarse region dummies
    if "region" in df.columns:
        region_dummies = pd.get_dummies(df["region"], prefix="region").astype(np.float32)
        region_dummies = region_dummies.drop(columns=[region_dummies.columns[0]])
        df = pd.concat([df, region_dummies], axis=1)
        region_dummy_cols = list(region_dummies.columns)
    else:
        region_dummy_cols = []

    base_e = BASE_PREDS + EVT_PREDS + evt_regime_cols
    variants = {
        "E_global": base_e,
        "E_latent_regime": base_e + latent_dummy_cols,
    }
    if region_dummy_cols:
        variants["E_coarse_region"] = base_e + region_dummy_cols

    all_cols = list(dict.fromkeys(["Resistance", "x", "y"] + base_e + latent_dummy_cols + region_dummy_cols))
    all_cols = [c for c in all_cols if c in df.columns]
    work = df[all_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    idx = np.arange(len(work))
    rand_tr, rand_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    groups = block_groups(work, args.block_km)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    blk_tr, blk_te = next(gss.split(idx, groups=groups))

    rows = []
    importance_rows = []
    for name, preds in variants.items():
        preds = [c for c in preds if c in work.columns]
        for split_name, (tr, te) in [("random", (rand_tr, rand_te)), ("block", (blk_tr, blk_te))]:
            metrics, model = fit_eval(work.iloc[tr], work.iloc[te], preds, args.trees)
            rows.append({"variant": name, "split": split_name, **metrics})
            if split_name == "block":
                imp = pd.Series(model.feature_importances_, index=preds).sort_values(ascending=False).head(15)
                for rank, (feat, val) in enumerate(imp.items(), 1):
                    importance_rows.append({"variant": name, "rank": rank, "feature": feat, "importance": float(val)})

    pd.DataFrame(rows).to_csv(args.output_dir / "comparison_metrics.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(args.output_dir / "block_importances.csv", index=False)

    report_lines = [
        f"# RF Regime Comparison ({TODAY})",
        "",
        "| Variant | Split | R2 | RMSE |",
        "|---|---|---|---|",
    ]
    for r in rows:
        report_lines.append(f"| {r['variant']} | {r['split']} | {r['r2']:.4f} | {r['rmse']:.4f} |")
    report_lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(report_lines))
    print(json.dumps({"out_dir": str(args.output_dir), "rows_used": len(work), "variants": list(variants.keys())}))


if __name__ == "__main__":
    main()
