#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from joblib import Parallel, delayed
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KDTree


TODAY = date.today().strftime("%Y-%m-%d")
RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_BLOCK_KM = 100.0
MORAN_K = 8

ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"

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
SPACE_COLS = {"x", "y", "x_sq_z", "y_sq_z", "xy_z"}

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
    "CLIM_vpd_mean_pre_z": "CLIM_vpd_mean_pre",
    "CLIM_vpd_std_pre_z": "CLIM_vpd_std_pre",
    "FS_EVT_resistance_proxy_z": "FS_EVT_resistance_proxy",
    "FS_EVT_regeneration_proxy_z": "FS_EVT_regeneration_proxy",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "More stable local RF bandwidth sweep for Western US Resistance. "
            "This is a GW-RF-related local RF pilot, not a full multiscale GW-RF."
        )
    )
    p.add_argument("--input", type=Path, default=INPUT)
    p.add_argument("--variant", default="E_EVT_fireregime_postclim", choices=["C_EVTproxy_postclim", "E_EVT_fireregime_postclim"])
    p.add_argument("--k-spatial", type=int, nargs="+", default=[2000, 5000, 10000])
    p.add_argument("--trees", type=int, default=100)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--max-features", default="sqrt")
    p.add_argument("--block-km", type=float, default=DEFAULT_BLOCK_KM)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--sample-n", type=int, default=None)
    p.add_argument("--xy-mode", choices=["with", "without", "both"], default="both")
    p.add_argument("--local-splits", choices=["block", "both"], default="block")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    std = v.std(ddof=1)
    return ((v - v.mean()) / std).astype(np.float32) if std > 0 else pd.Series(
        np.zeros(len(v), dtype=np.float32), index=v.index
    )


def log1p_z(s: pd.Series) -> pd.Series:
    return zscore(np.log1p(pd.to_numeric(s, errors="coerce").astype(float).clip(lower=0)))


def ensure_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    for z_col, raw_col in BASE_TO_Z.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    for col, expr in [
        ("x_sq_z", out["x"] ** 2),
        ("y_sq_z", out["y"] ** 2),
        ("xy_z", out["x"] * out["y"]),
    ]:
        if col not in out.columns:
            out[col] = zscore(expr)
    regime = out["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    dummies = pd.get_dummies(regime, prefix="EVT_regime").astype(np.float32)
    if "EVT_regime_other" in dummies.columns:
        dummies = dummies.drop(columns=["EVT_regime_other"])
    out = pd.concat([out, dummies], axis=1)
    return out, list(dummies.columns)


def build_variants(evt_cols: list[str]) -> OrderedDict[str, list[str]]:
    return OrderedDict(
        [
            ("C_EVTproxy_postclim", BASE_PREDS + EVT_PREDS),
            ("E_EVT_fireregime_postclim", BASE_PREDS + EVT_PREDS + evt_cols),
        ]
    )


def block_groups(df: pd.DataFrame, block_km: float) -> pd.Series:
    bm = block_km * 1000.0
    return pd.Series(
        [f"{int(np.floor(x / bm))}_{int(np.floor(y / bm))}" for x, y in zip(df["x"], df["y"])],
        index=df.index,
    )


def compute_moran(coords_df: pd.DataFrame, residuals: np.ndarray) -> float:
    k = min(MORAN_K, len(coords_df) - 1)
    if k < 1:
        return float("nan")
    weights = KNN.from_array(coords_df[["x", "y"]].to_numpy(), k=k)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return float(moran.I)


def fit_rf(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    n_trees: int,
    min_samples_leaf: int,
    max_features: str | int | float,
    n_jobs: int,
    sample_weight: np.ndarray | None = None,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_trees,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, coords_df: pd.DataFrame) -> dict[str, float]:
    residuals = y_true - y_pred
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "moran_i": compute_moran(coords_df, residuals),
    }


def run_local_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: list[str],
    k_spatial: int,
    n_trees: int,
    min_samples_leaf: int,
    max_features: str | int | float,
    n_jobs: int,
) -> tuple[dict[str, float], np.ndarray]:
    train_coords = train_df[["x", "y"]].to_numpy(dtype=float)
    test_coords = test_df[["x", "y"]].to_numpy(dtype=float)
    train_X = train_df[predictors].to_numpy(dtype=float)
    train_y = train_df["Resistance"].to_numpy(dtype=float)
    test_X = test_df[predictors].to_numpy(dtype=float)

    tree = KDTree(train_coords)
    k = min(k_spatial, len(train_df))

    def _predict_one(i: int) -> float:
        dist, idx = tree.query(test_coords[i:i + 1], k=k)
        local_X = train_X[idx[0]]
        local_y = train_y[idx[0]]
        bw = dist[0].max()
        if bw > 0:
            weights = np.exp(-0.5 * (dist[0] / (bw + 1e-10)) ** 2)
        else:
            weights = np.ones(k, dtype=float)
        weights = weights / weights.sum()
        model = fit_rf(
            local_X,
            local_y,
            n_trees=n_trees,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=1,
            sample_weight=weights,
        )
        return float(model.predict(test_X[i:i + 1])[0])

    preds = Parallel(n_jobs=n_jobs, verbose=5, prefer="threads")(
        delayed(_predict_one)(i) for i in range(len(test_df))
    )
    pred_arr = np.asarray(preds, dtype=float)
    metrics = metric_dict(
        test_df["Resistance"].to_numpy(dtype=float),
        pred_arr,
        test_df[["x", "y"]],
    )
    return metrics, pred_arr


def predictor_sets(predictors: list[str], xy_mode: str) -> OrderedDict[str, list[str]]:
    out: OrderedDict[str, list[str]] = OrderedDict()
    if xy_mode in {"with", "both"}:
        out["with_xy"] = predictors
    if xy_mode in {"without", "both"}:
        out["without_xy"] = [c for c in predictors if c not in SPACE_COLS]
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    df, evt_cols = ensure_columns(df)
    variants = build_variants(evt_cols)
    predictors_full = [c for c in variants[args.variant] if c in df.columns]

    work = (
        df[list(dict.fromkeys(["Resistance", "x", "y"] + predictors_full))]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    if args.sample_n is not None and args.sample_n < len(work):
        work = work.sample(n=args.sample_n, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Rows: {len(work)}, Predictors(full): {len(predictors_full)}", flush=True)

    idx = np.arange(len(work))
    rand_tr, rand_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    groups = block_groups(work, args.block_km)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    blk_tr, blk_te = next(gss.split(idx, groups=groups))

    split_map: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict(
        [
            ("random", (rand_tr, rand_te)),
            ("block", (blk_tr, blk_te)),
        ]
    )
    local_split_names = ["block"] if args.local_splits == "block" else ["random", "block"]

    rows: list[dict] = []
    metadata = {
        "input": str(args.input),
        "variant": args.variant,
        "k_spatial": args.k_spatial,
        "trees": args.trees,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "block_km": args.block_km,
        "sample_n": args.sample_n,
        "xy_mode": args.xy_mode,
        "local_splits": args.local_splits,
        "predictors_full": predictors_full,
        "rows": int(len(work)),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    for xy_label, predictors in predictor_sets(predictors_full, args.xy_mode).items():
        print(f"\n=== Predictor set: {xy_label} ({len(predictors)} vars) ===", flush=True)

        for split_name, (tr, te) in split_map.items():
            train, test = work.iloc[tr], work.iloc[te]
            global_model = fit_rf(
                train[predictors],
                train["Resistance"],
                n_trees=args.trees,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                n_jobs=args.n_jobs,
            )
            global_pred = global_model.predict(test[predictors])
            y = test["Resistance"].to_numpy(dtype=float)
            global_metrics = metric_dict(y, global_pred, test[["x", "y"]])
            rows.append(
                {
                    "model": "global_rf",
                    "variant": args.variant,
                    "xy_mode": xy_label,
                    "k_spatial": None,
                    "split": split_name,
                    **global_metrics,
                    "n_predictors": len(predictors),
                }
            )
            print(
                f"  global_rf {split_name} {xy_label}: "
                f"R2={global_metrics['r2']:.4f} RMSE={global_metrics['rmse']:.4f} "
                f"MoranI={global_metrics['moran_i']:.4f}",
                flush=True,
            )

            pred_df = test[["x", "y", "Resistance"]].copy()
            pred_df["pred"] = global_pred
            pred_df["residual"] = pred_df["Resistance"] - pred_df["pred"]
            pred_df.to_csv(
                args.output_dir / f"global_rf_{xy_label}_{split_name}_predictions.csv",
                index=False,
            )

        for k in args.k_spatial:
            for split_name in local_split_names:
                tr, te = split_map[split_name]
                train, test = work.iloc[tr], work.iloc[te]
                print(f"\nlocal_rf k={k} split={split_name} xy={xy_label} ...", flush=True)
                local_metrics, preds_arr = run_local_rf(
                    train_df=train,
                    test_df=test,
                    predictors=predictors,
                    k_spatial=k,
                    n_trees=args.trees,
                    min_samples_leaf=args.min_samples_leaf,
                    max_features=args.max_features,
                    n_jobs=args.n_jobs,
                )
                rows.append(
                    {
                        "model": "local_rf",
                        "variant": args.variant,
                        "xy_mode": xy_label,
                        "k_spatial": k,
                        "split": split_name,
                        **local_metrics,
                        "n_predictors": len(predictors),
                    }
                )
                print(
                    f"  local_rf k={k} {split_name} {xy_label}: "
                    f"R2={local_metrics['r2']:.4f} RMSE={local_metrics['rmse']:.4f} "
                    f"MoranI={local_metrics['moran_i']:.4f}",
                    flush=True,
                )

                out_df = test[["x", "y", "Resistance"]].copy()
                out_df["pred"] = preds_arr
                out_df["residual"] = out_df["Resistance"] - out_df["pred"]
                out_df.to_csv(
                    args.output_dir / f"local_rf_k{k}_{xy_label}_{split_name}_predictions.csv",
                    index=False,
                )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output_dir / "multiscale_local_rf_metrics.csv", index=False)

    report = [
        f"# Local RF Bandwidth Sweep v2 ({TODAY})",
        "",
        "This is a more stable GW-RF-related local RF pilot, not a full multiscale GW-RF implementation.",
        "",
        "| Model | Variant | XY mode | k_spatial | Split | Predictors | R2 | RMSE | Moran's I |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in results_df.sort_values(["xy_mode", "split", "model", "k_spatial"], na_position="first").iterrows():
        k_label = "NA" if pd.isna(r["k_spatial"]) else str(int(r["k_spatial"]))
        report.append(
            f"| {r['model']} | {r['variant']} | {r['xy_mode']} | {k_label} | "
            f"{r['split']} | {int(r['n_predictors'])} | {r['r2']:.4f} | {r['rmse']:.4f} | {r['moran_i']:.4f} |"
        )
    report.append("")
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(json.dumps({"out_dir": str(args.output_dir), "rows": len(work)}, indent=2))


if __name__ == "__main__":
    main()
