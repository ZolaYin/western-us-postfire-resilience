#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
TODAY = date.today().strftime("%Y-%m-%d")

RANDOM_STATE = 42
TEST_SIZE = 0.2
BLOCK_KM = 100.0
N_ESTIMATORS = 300
MORAN_K = 8
MAX_K = 5000

BASE_PREDS = [
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
    "CLIM_vpd_mean_pre_z",
    "CLIM_vpd_std_pre_z",
    "x",
    "y",
    "x_sq_z",
    "y_sq_z",
    "xy_z",
]

STAGE5B_PREDS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "HUM_roaddens_r5km_z",
    "HUM_traildens_r10km_z",
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
]

SMOOTH_RADIUS_KM = {
    "FS_CBH_t0agg_z": 150,
    "HUM_roaddens_r5km_z": 150,
    "TS_SOC_0_30cm_z": 300,
    "CLIM_pr_sum_pre_z": 300,
    "TS_elev_m_z": 600,
    "FS_TCC_t0_z": 600,
    "TS_slope_deg_z": 600,
    "CLIM_tmmn_mean_pre_z": 1200,
    "HUM_traildens_r10km_z": 1200,
    "HUM_viirs_near_t0_log_z": 1200,
    "HUM_imperv_near_t0_z": 1200,
}
UNIFORM_CONTROL_KM = 300

SPACE_COLS = {"x", "y", "x_sq_z", "y_sq_z", "xy_z"}

RAW_TO_Z = {
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
}

STAGE5B_ALIAS_MAP = {
    "TS_SOC_0_30cm_z": "TS_SOC_0_30cm_clean_z",
}


@dataclass(frozen=True)
class VariantSpec:
    name: str
    family: str
    baseline_name: str
    base_key: str
    smooth_mode: str
    note: str


VARIANT_SPECS = [
    VariantSpec("stage5b_raw", "stage5b", "stage5b_raw", "stage5b", "none", "11 continuous stage5b predictors only."),
    VariantSpec("stage5b_uniform300", "stage5b", "stage5b_raw", "stage5b", "uniform300", "Stage5b + uniform 300-km smooth control."),
    VariantSpec("stage5b_mgwr", "stage5b", "stage5b_raw", "stage5b", "mgwr", "Stage5b + MGWR-matched smooth features."),
    VariantSpec("m2_baseline", "m2", "m2_baseline", "m2", "none", "Main-text M2 baseline: baseline predictors + FS_EVT_group_class dummies."),
    VariantSpec("m2_uniform300", "m2", "m2_baseline", "m2", "uniform300", "M2 baseline + uniform 300-km smooth control for stage5b variables."),
    VariantSpec("m2_mgwr", "m2", "m2_baseline", "m2", "mgwr", "M2 baseline + MGWR-matched smooth features for stage5b variables."),
    VariantSpec("m3_baseline", "m3", "m3_baseline", "m3", "none", "Main-text M3 baseline: baseline predictors + raw EVT code dummies."),
    VariantSpec("m3_mgwr", "m3", "m3_baseline", "m3", "mgwr", "M3 baseline + MGWR-matched smooth features for stage5b variables."),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round-2 MGWR-informed multiscale RF comparison for Western US Resistance. "
            "Anchors smoothing experiments directly to stage5b and M2/M3 main-text baselines."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response", default="Resistance")
    parser.add_argument("--n-trees", type=int, default=N_ESTIMATORS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--block-km", type=float, default=BLOCK_KM)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[spec.name for spec in VARIANT_SPECS],
        help="Subset of variant names to run.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save test-set predictions for each variant and split.",
    )
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def log1p_z(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float).clip(lower=0)
    return zscore(np.log1p(vals))


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

    for canonical, alias in STAGE5B_ALIAS_MAP.items():
        if canonical not in out.columns and alias in out.columns:
            out[canonical] = pd.to_numeric(out[alias], errors="coerce").astype(np.float32)

    for col, expr in [
        ("x_sq_z", out["x"] ** 2),
        ("y_sq_z", out["y"] ** 2),
        ("xy_z", out["x"] * out["y"]),
    ]:
        if col not in out.columns:
            out[col] = zscore(expr)

    out["FS_EVT_group_class_clean"] = (
        out["FS_EVT_group_class"].astype("string").fillna("unknown").astype(str)
    )
    group_dummies = pd.get_dummies(
        out["FS_EVT_group_class_clean"], prefix="EVT_group", dtype=np.float32
    )
    out = pd.concat([out, group_dummies], axis=1)

    code_str = out["FS_EVT2022_code"].astype("Int64").astype(str).fillna("missing")
    code_dummies = pd.get_dummies(code_str, prefix="EVT_code", dtype=np.float32)
    out = pd.concat([out, code_dummies], axis=1)
    return out


def block_groups(df: pd.DataFrame, block_km: float) -> pd.Series:
    block_m = block_km * 1000.0
    labels = [
        f"{int(np.floor(x / block_m))}_{int(np.floor(y / block_m))}"
        for x, y in zip(df["x"], df["y"])
    ]
    return pd.Series(labels, index=df.index)


def build_split_masks(
    work: pd.DataFrame, block_km: float, random_state: int
) -> dict[str, np.ndarray]:
    idx = np.arange(len(work))
    _, rand_te = train_test_split(idx, test_size=TEST_SIZE, random_state=random_state)
    groups = block_groups(work, block_km)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_state)
    _, blk_te = next(gss.split(idx, groups=groups))

    is_test_rnd = np.zeros(len(work), dtype=bool)
    is_test_blk = np.zeros(len(work), dtype=bool)
    is_test_rnd[rand_te] = True
    is_test_blk[blk_te] = True
    return {"random": is_test_rnd, "block": is_test_blk}


def moran_i(residuals: np.ndarray, coords: np.ndarray, k: int = MORAN_K) -> float:
    if len(residuals) <= 1:
        return float("nan")
    k_eff = min(k, len(residuals) - 1)
    tree = KDTree(coords)
    _, idxs = tree.query(coords, k=k_eff + 1)
    idxs = idxs[:, 1:]
    n = len(residuals)
    z = residuals - residuals.mean()
    weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        weights[i, idxs[i]] = 1.0
    row_sum = weights.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    weights = weights / row_sum
    num = n * float(np.einsum("ij,i,j->", weights, z, z))
    den = float(weights.sum()) * float((z**2).sum())
    return num / den if den != 0 else float("nan")


def smooth_train_test(
    train_coords: np.ndarray,
    train_X: np.ndarray,
    test_coords: np.ndarray,
    radius_m: float,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree = KDTree(train_coords)

    def _weighted(dists: np.ndarray, idxs: np.ndarray) -> np.ndarray:
        weights = np.exp(-0.5 * (dists / radius_m) ** 2)
        weights[dists > 2.5 * radius_m] = 0.0
        weight_sum = weights.sum(axis=1)
        no_nb = weight_sum == 0
        weight_sum = np.where(no_nb, 1.0, weight_sum)
        out = np.einsum("ij,ijp->ip", weights, train_X[idxs]) / weight_sum[:, None]
        if no_nb.any():
            out[no_nb] = train_X[idxs[no_nb, 0]]
        return out.astype(np.float32)

    k_train = min(len(train_coords) - 1, max_k)
    d_train, i_train = tree.query(train_coords, k=k_train + 1)
    smooth_train = _weighted(d_train[:, 1:], i_train[:, 1:])

    k_test = min(len(train_coords), max_k)
    d_test, i_test = tree.query(test_coords, k=k_test)
    smooth_test = _weighted(d_test, i_test)
    return smooth_train, smooth_test


def build_smooth_bundle(
    train_coords: np.ndarray,
    stage5b_train: np.ndarray,
    test_coords: np.ndarray,
    max_k: int,
) -> dict[str, np.ndarray]:
    n_train, n_stage5 = stage5b_train.shape
    matched_train = np.zeros((n_train, n_stage5), dtype=np.float32)
    matched_test = np.zeros((len(test_coords), n_stage5), dtype=np.float32)

    grouped_indices: dict[int, list[int]] = {}
    for j, pred in enumerate(STAGE5B_PREDS):
        grouped_indices.setdefault(SMOOTH_RADIUS_KM[pred], []).append(j)

    for radius_km, js in sorted(grouped_indices.items()):
        s_train, s_test = smooth_train_test(
            train_coords=train_coords,
            train_X=stage5b_train[:, js],
            test_coords=test_coords,
            radius_m=radius_km * 1000.0,
            max_k=max_k,
        )
        matched_train[:, js] = s_train
        matched_test[:, js] = s_test

    uniform_train, uniform_test = smooth_train_test(
        train_coords=train_coords,
        train_X=stage5b_train,
        test_coords=test_coords,
        radius_m=UNIFORM_CONTROL_KM * 1000.0,
        max_k=max_k,
    )

    return {
        "mgwr_train": matched_train,
        "mgwr_test": matched_test,
        "uniform300_train": uniform_train,
        "uniform300_test": uniform_test,
    }


def block_feature_names(mode: str) -> list[str]:
    if mode == "mgwr":
        return [f"{pred}_s{SMOOTH_RADIUS_KM[pred]}km_mgwr" for pred in STAGE5B_PREDS]
    if mode == "uniform300":
        return [f"{pred}_s{UNIFORM_CONTROL_KM}km_uniform" for pred in STAGE5B_PREDS]
    raise ValueError(f"Unsupported smooth mode: {mode}")


def feature_group(feature: str) -> str:
    if feature in STAGE5B_PREDS:
        return "stage5b_raw"
    if feature.endswith("_mgwr"):
        return "stage5b_smooth_mgwr"
    if feature.endswith("_uniform"):
        return "stage5b_smooth_uniform300"
    if feature.startswith("EVT_group_"):
        return "forest_type_group"
    if feature.startswith("EVT_code_"):
        return "forest_type_code"
    if feature in SPACE_COLS:
        return "space"
    return "baseline_other"


def get_variant_specs(selected: list[str]) -> list[VariantSpec]:
    spec_map = {spec.name: spec for spec in VARIANT_SPECS}
    unknown = [name for name in selected if name not in spec_map]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return [spec_map[name] for name in selected]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    variant_specs = get_variant_specs(args.variants)

    print("Loading and preparing data ...", flush=True)
    raw = pd.read_parquet(args.input)
    df = prepare(raw)

    group_cols = sorted([c for c in df.columns if c.startswith("EVT_group_")])
    code_cols = sorted([c for c in df.columns if c.startswith("EVT_code_")])

    base_feature_map = {
        "stage5b": STAGE5B_PREDS,
        "m2": BASE_PREDS + group_cols,
        "m3": BASE_PREDS + code_cols,
    }

    needed_cols = [args.response, "x", "y"]
    for cols in base_feature_map.values():
        needed_cols.extend(cols)
    work = (
        df[list(dict.fromkeys([c for c in needed_cols if c in df.columns]))]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    if args.sample_n is not None and args.sample_n < len(work):
        work = work.sample(n=args.sample_n, random_state=args.random_state).reset_index(drop=True)
    print(f"Rows used: {len(work):,}", flush=True)

    split_masks = build_split_masks(work, args.block_km, args.random_state)
    coords = work[["x", "y"]].to_numpy(dtype=np.float64)
    y = work[args.response].to_numpy(dtype=np.float32)

    X_map: dict[str, np.ndarray] = {}
    feature_name_map: dict[str, list[str]] = {}
    for key, cols in base_feature_map.items():
        available = [c for c in cols if c in work.columns]
        feature_name_map[key] = available
        X_map[key] = work[available].to_numpy(dtype=np.float32)

    metadata = {
        "date": TODAY,
        "input": str(args.input),
        "rows_used": int(len(work)),
        "response": args.response,
        "random_state": args.random_state,
        "block_km": args.block_km,
        "n_trees": args.n_trees,
        "max_k": args.max_k,
        "sample_n": args.sample_n,
        "stage5b_predictors": STAGE5B_PREDS,
        "m2_predictor_count": len(feature_name_map["m2"]),
        "m3_predictor_count": len(feature_name_map["m3"]),
        "group_dummy_count": len(group_cols),
        "code_dummy_count": len(code_cols),
        "smooth_radius_km": SMOOTH_RADIUS_KM,
        "uniform_control_km": UNIFORM_CONTROL_KM,
        "variants": [asdict(spec) for spec in variant_specs],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    rf = RandomForestRegressor(
        n_estimators=args.n_trees,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    metric_rows: list[dict] = []
    importance_rows: list[dict] = []

    for split_name, is_test in split_masks.items():
        print(f"\n=== Split: {split_name} ===", flush=True)
        tr = ~is_test
        te = is_test

        smooth_bundle = build_smooth_bundle(
            train_coords=coords[tr],
            stage5b_train=X_map["stage5b"][tr],
            test_coords=coords[te],
            max_k=args.max_k,
        )

        smooth_arrays = {
            "none": (None, None, []),
            "mgwr": (
                smooth_bundle["mgwr_train"],
                smooth_bundle["mgwr_test"],
                block_feature_names("mgwr"),
            ),
            "uniform300": (
                smooth_bundle["uniform300_train"],
                smooth_bundle["uniform300_test"],
                block_feature_names("uniform300"),
            ),
        }

        for spec in variant_specs:
            base_train = X_map[spec.base_key][tr]
            base_test = X_map[spec.base_key][te]
            feature_names = list(feature_name_map[spec.base_key])

            smooth_train, smooth_test, smooth_names = smooth_arrays[spec.smooth_mode]
            if smooth_train is None:
                X_train = base_train
                X_test = base_test
            else:
                X_train = np.hstack([base_train, smooth_train]).astype(np.float32)
                X_test = np.hstack([base_test, smooth_test]).astype(np.float32)
                feature_names.extend(smooth_names)

            rf.fit(X_train, y[tr])
            y_pred = rf.predict(X_test)
            residuals = y[te] - y_pred
            metrics = {
                "variant": spec.name,
                "family": spec.family,
                "baseline_name": spec.baseline_name,
                "base_key": spec.base_key,
                "smooth_mode": spec.smooth_mode,
                "split": split_name,
                "rows": int(len(work)),
                "train_rows": int(tr.sum()),
                "test_rows": int(te.sum()),
                "n_features": int(X_train.shape[1]),
                "r2": float(r2_score(y[te], y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y[te], y_pred))),
                "moran_i": float(moran_i(residuals, coords[te], k=MORAN_K)),
            }
            metric_rows.append(metrics)
            print(
                f"{spec.name:18s} "
                f"R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} "
                f"MoranI={metrics['moran_i']:.4f} p={metrics['n_features']}",
                flush=True,
            )

            if split_name == "block":
                for feature, importance in sorted(
                    zip(feature_names, rf.feature_importances_),
                    key=lambda pair: pair[1],
                    reverse=True,
                ):
                    importance_rows.append(
                        {
                            "variant": spec.name,
                            "feature": feature,
                            "importance": float(importance),
                            "group": feature_group(feature),
                        }
                    )

            if args.save_predictions:
                pred_df = pd.DataFrame(
                    {
                        "x": coords[te, 0],
                        "y": coords[te, 1],
                        "observed": y[te],
                        "predicted": y_pred,
                        "residual": residuals,
                    }
                )
                pred_df.to_csv(
                    args.output_dir / f"{spec.name}_{split_name}_predictions.csv",
                    index=False,
                )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.output_dir / "metrics_long.csv", index=False)

    summary_df = metrics_df.copy()
    base_lookup = summary_df.set_index(["variant", "split"])[["r2", "rmse", "moran_i"]]
    delta_r2 = []
    delta_rmse = []
    delta_moran = []
    for _, row in summary_df.iterrows():
        key = (row["baseline_name"], row["split"])
        if key in base_lookup.index:
            base = base_lookup.loc[key]
            delta_r2.append(float(row["r2"] - base["r2"]))
            delta_rmse.append(float(row["rmse"] - base["rmse"]))
            delta_moran.append(float(row["moran_i"] - base["moran_i"]))
        else:
            delta_r2.append(float("nan"))
            delta_rmse.append(float("nan"))
            delta_moran.append(float("nan"))
    summary_df["delta_r2_vs_family_base"] = delta_r2
    summary_df["delta_rmse_vs_family_base"] = delta_rmse
    summary_df["delta_moran_vs_family_base"] = delta_moran
    summary_df = summary_df[
        [
            "variant",
            "family",
            "split",
            "rows",
            "train_rows",
            "test_rows",
            "n_features",
            "smooth_mode",
            "r2",
            "rmse",
            "moran_i",
            "delta_r2_vs_family_base",
            "delta_rmse_vs_family_base",
            "delta_moran_vs_family_base",
        ]
    ].sort_values(["split", "family", "variant"])
    summary_df.to_csv(args.output_dir / "metrics_summary.csv", index=False)

    wide = summary_df.pivot(
        index=["variant", "family", "smooth_mode", "n_features"],
        columns="split",
        values=[
            "r2",
            "rmse",
            "moran_i",
            "delta_r2_vs_family_base",
            "delta_rmse_vs_family_base",
            "delta_moran_vs_family_base",
        ],
    )
    wide.columns = ["_".join(col) for col in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(args.output_dir / "metrics_wide.csv", index=False)

    if importance_rows:
        importance_df = pd.DataFrame(importance_rows)
        importance_df.to_csv(args.output_dir / "block_feature_importance.csv", index=False)

        group_importance = (
            importance_df.groupby(["variant", "group"], as_index=False)["importance"].sum()
            .sort_values(["variant", "importance"], ascending=[True, False])
        )
        group_importance.to_csv(args.output_dir / "block_group_importance.csv", index=False)

    report_lines = [
        f"# Western US Multiscale RF Round 2 ({TODAY})",
        "",
        f"- Input: `{args.input.name}`",
        f"- Response: `{args.response}`",
        f"- Rows used: `{len(work):,}`",
        f"- Random state: `{args.random_state}`",
        f"- Block size: `{args.block_km:g} km`",
        f"- Trees per RF: `{args.n_trees}`",
        f"- Max smoothing neighbors: `{args.max_k}`",
        "",
        "## Variants",
        "",
    ]
    for spec in variant_specs:
        report_lines.append(f"- `{spec.name}`: {spec.note}")
    report_lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "```text",
            summary_df.round(4).to_string(index=False),
            "```",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report_lines))

    print(f"\nOutputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
