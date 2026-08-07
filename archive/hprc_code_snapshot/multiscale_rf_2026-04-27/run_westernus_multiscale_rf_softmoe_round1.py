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
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


TODAY = date.today().strftime("%Y-%m-%d")
INPUT = Path("westernus_current_candidate_table_plus_regions.parquet")

RANDOM_STATE = 42
TEST_SIZE = 0.2
BLOCK_KM = 100.0
GLOBAL_TREES = 300
EXPERT_TREES = 120
MORAN_K = 8
MAX_K = 5000
GATE_FIT_SAMPLE_N = 60000
EXPERT_MIN_LEAF = 10
MIN_EFFECTIVE_N = 300.0
GATE_SOFTNESS = 1.0

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

SPACE_COLS = ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]

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
    parent_name: str
    model_kind: str
    base_key: str
    smooth_mode: str
    gate_mode: str
    n_experts: int
    note: str


VARIANT_SPECS = [
    VariantSpec(
        "m2_baseline",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "rf",
        "m2",
        "none",
        "none",
        0,
        "Main-text M2 baseline: baseline predictors + FS_EVT_group_class dummies.",
    ),
    VariantSpec(
        "m2_mgwr",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "rf",
        "m2",
        "mgwr",
        "none",
        0,
        "M2 baseline + MGWR-matched smooth features for stage5b variables.",
    ),
    VariantSpec(
        "m2_soft3_rawsmooth",
        "m2",
        "m2_baseline",
        "m2_mgwr",
        "softmoe",
        "m2",
        "mgwr",
        "rawsmooth",
        3,
        "Soft-gated residual experts over M2+MGWR, gate on raw+smooth stage5b gradients.",
    ),
    VariantSpec(
        "m2_soft4_rawsmooth",
        "m2",
        "m2_baseline",
        "m2_mgwr",
        "softmoe",
        "m2",
        "mgwr",
        "rawsmooth",
        4,
        "Four soft residual experts over M2+MGWR, gate on raw+smooth stage5b gradients.",
    ),
    VariantSpec(
        "m2_soft3_rawsmooth_space",
        "m2",
        "m2_baseline",
        "m2_mgwr",
        "softmoe",
        "m2",
        "mgwr",
        "rawsmooth_space",
        3,
        "Soft residual experts over M2+MGWR, gate on raw+smooth stage5b plus broad space terms.",
    ),
    VariantSpec(
        "m2_soft3_rawsmooth_group",
        "m2",
        "m2_baseline",
        "m2_mgwr",
        "softmoe",
        "m2",
        "mgwr",
        "rawsmooth_group",
        3,
        "Soft residual experts over M2+MGWR, gate on raw+smooth stage5b plus forest-type groups.",
    ),
    VariantSpec(
        "m3_baseline",
        "m3",
        "m3_baseline",
        "m3_baseline",
        "rf",
        "m3",
        "none",
        "none",
        0,
        "Main-text M3 baseline: baseline predictors + raw EVT code dummies.",
    ),
    VariantSpec(
        "m3_mgwr",
        "m3",
        "m3_baseline",
        "m3_baseline",
        "rf",
        "m3",
        "mgwr",
        "none",
        0,
        "M3 baseline + MGWR-matched smooth features for stage5b variables.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Soft-MoE round-1 continuation of the Western US MGWR-informed multiscale RF line. "
            "Builds soft-gated residual RF experts on top of M2+MGWR while keeping the same "
            "block CV, RMSE, and residual Moran diagnostics as the round-2 benchmark."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response", default="Resistance")
    parser.add_argument("--global-trees", type=int, default=GLOBAL_TREES)
    parser.add_argument("--expert-trees", type=int, default=EXPERT_TREES)
    parser.add_argument("--expert-min-leaf", type=int, default=EXPERT_MIN_LEAF)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--block-km", type=float, default=BLOCK_KM)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument("--gate-fit-sample-n", type=int, default=GATE_FIT_SAMPLE_N)
    parser.add_argument("--gate-softness", type=float, default=GATE_SOFTNESS)
    parser.add_argument("--min-effective-n", type=float, default=MIN_EFFECTIVE_N)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[spec.name for spec in VARIANT_SPECS],
        help="Subset of variant names to run.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save test-set predictions and, for soft models, gate weights.",
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

    return {
        "mgwr_train": matched_train,
        "mgwr_test": matched_test,
    }


def block_feature_names() -> list[str]:
    return [f"{pred}_s{SMOOTH_RADIUS_KM[pred]}km_mgwr" for pred in STAGE5B_PREDS]


def feature_group(feature: str) -> str:
    if feature in STAGE5B_PREDS:
        return "stage5b_raw"
    if feature.endswith("_mgwr"):
        return "stage5b_smooth_mgwr"
    if feature.startswith("EVT_group_"):
        return "forest_type_group"
    if feature.startswith("EVT_code_"):
        return "forest_type_code"
    if feature in set(SPACE_COLS):
        return "space"
    return "baseline_other"


def get_variant_specs(selected: list[str]) -> list[VariantSpec]:
    spec_map = {spec.name: spec for spec in VARIANT_SPECS}
    unknown = [name for name in selected if name not in spec_map]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return [spec_map[name] for name in selected]


def standardize_train_test(
    train_X: np.ndarray, test_X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0, ddof=1)
    std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
    train_std = ((train_X - mean) / std).astype(np.float32)
    test_std = ((test_X - mean) / std).astype(np.float32)
    return train_std, test_std, mean.astype(np.float32), std.astype(np.float32)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(shifted)
    denom = expv.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return (expv / denom).astype(np.float32)


def effective_n(weights: np.ndarray) -> float:
    num = float(weights.sum()) ** 2
    den = float(np.square(weights).sum())
    return num / den if den > 0 else 0.0


def fit_soft_gate(
    train_G: np.ndarray,
    test_G: np.ndarray,
    gate_features: list[str],
    n_experts: int,
    random_state: int,
    gate_fit_sample_n: int,
    gate_softness: float,
) -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]]:
    train_std, test_std, mean, std = standardize_train_test(train_G, test_G)

    fit_X = train_std
    if gate_fit_sample_n and len(train_std) > gate_fit_sample_n:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(train_std), size=gate_fit_sample_n, replace=False)
        fit_X = train_std[idx]

    kmeans = KMeans(n_clusters=n_experts, random_state=random_state, n_init=20)
    kmeans.fit(fit_X)
    centers = kmeans.cluster_centers_.astype(np.float32)

    d2_train = cdist(train_std, centers, metric="sqeuclidean").astype(np.float32)
    d2_test = cdist(test_std, centers, metric="sqeuclidean").astype(np.float32)
    hard = d2_train.argmin(axis=1)
    hard_counts = np.bincount(hard, minlength=n_experts)
    order = np.argsort(-hard_counts)

    centers = centers[order]
    d2_train = d2_train[:, order]
    d2_test = d2_test[:, order]
    hard = d2_train.argmin(axis=1)
    hard_counts = np.bincount(hard, minlength=n_experts)

    fallback_sigma = float(np.median(d2_train.min(axis=1)))
    if not np.isfinite(fallback_sigma) or fallback_sigma <= 0:
        fallback_sigma = 1.0

    sigma2 = np.zeros(n_experts, dtype=np.float32)
    for k in range(n_experts):
        own = d2_train[hard == k, k]
        if len(own) == 0:
            sigma2[k] = fallback_sigma
        else:
            sigma2[k] = max(float(np.median(own)), fallback_sigma * 0.25, 1e-6)

    temp = max(gate_softness, 1e-6) ** 2
    logits_train = -0.5 * d2_train / (sigma2[None, :] * temp)
    logits_test = -0.5 * d2_test / (sigma2[None, :] * temp)
    w_train = softmax_rows(logits_train)
    w_test = softmax_rows(logits_test)

    entropy_train = -np.sum(w_train * np.log(np.clip(w_train, 1e-12, 1.0)), axis=1)
    diag_rows = []
    center_rows = []
    mean_max_w = float(w_train.max(axis=1).mean())
    mean_entropy = float(entropy_train.mean())
    centers_unscaled = centers * std[None, :] + mean[None, :]
    for k in range(n_experts):
        diag_rows.append(
            {
                "cluster": int(k + 1),
                "hard_train_n": int(hard_counts[k]),
                "soft_train_weight_sum": float(w_train[:, k].sum()),
                "soft_train_effective_n": float(effective_n(w_train[:, k])),
                "sigma2": float(sigma2[k]),
                "mean_max_weight_train": mean_max_w,
                "mean_entropy_train": mean_entropy,
            }
        )
        for feature, value in zip(gate_features, centers_unscaled[k]):
            center_rows.append(
                {
                    "cluster": int(k + 1),
                    "feature": feature,
                    "center_value": float(value),
                }
            )
    return w_train, w_test, diag_rows, center_rows


def build_model_matrix(
    base_train: np.ndarray,
    base_test: np.ndarray,
    smooth_train: np.ndarray | None,
    smooth_test: np.ndarray | None,
    base_features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = list(base_features)
    if smooth_train is None or smooth_test is None:
        return base_train.astype(np.float32), base_test.astype(np.float32), features
    features.extend(block_feature_names())
    return (
        np.hstack([base_train, smooth_train]).astype(np.float32),
        np.hstack([base_test, smooth_test]).astype(np.float32),
        features,
    )


def make_rf(
    n_trees: int,
    random_state: int,
    n_jobs: int,
    min_samples_leaf: int = 1,
    oob_score: bool = False,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_trees,
        random_state=random_state,
        n_jobs=n_jobs,
        min_samples_leaf=min_samples_leaf,
        oob_score=oob_score,
    )


def run_softmoe_variant(
    spec: VariantSpec,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    gate_train: np.ndarray,
    gate_test: np.ndarray,
    gate_features: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    coords_test: np.ndarray,
    args: argparse.Namespace,
    split_name: str,
) -> tuple[dict, list[dict], list[dict], list[dict], pd.DataFrame | None]:
    base_rf = make_rf(
        n_trees=args.global_trees,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        min_samples_leaf=1,
        oob_score=True,
    )
    base_rf.fit(X_train, y_train)
    base_train_pred = getattr(base_rf, "oob_prediction_", None)
    if base_train_pred is None or len(base_train_pred) != len(y_train):
        base_train_pred = base_rf.predict(X_train)
    else:
        base_train_pred = np.asarray(base_train_pred, dtype=np.float32)
        missing = ~np.isfinite(base_train_pred)
        if missing.any():
            base_train_pred[missing] = base_rf.predict(X_train[missing])
    base_test_pred = base_rf.predict(X_test).astype(np.float32)
    residual_train = (y_train - base_train_pred).astype(np.float32)

    w_train, w_test, gate_diag, gate_centers = fit_soft_gate(
        train_G=gate_train,
        test_G=gate_test,
        gate_features=gate_features,
        n_experts=spec.n_experts,
        random_state=args.random_state,
        gate_fit_sample_n=args.gate_fit_sample_n,
        gate_softness=args.gate_softness,
    )
    for row in gate_centers:
        row["variant"] = spec.name
        row["split"] = split_name
        row["gate_mode"] = spec.gate_mode
        row["n_experts"] = spec.n_experts

    expert_test_preds = np.zeros((len(X_test), spec.n_experts), dtype=np.float32)
    importance_rows = []
    for feature, importance in sorted(
        zip(feature_names, base_rf.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    ):
        importance_rows.append(
            {
                "variant": spec.name,
                "source": "base_global",
                "expert": 0,
                "feature": feature,
                "importance": float(importance),
                "group": feature_group(feature),
            }
        )

    weighted_expert_importance = np.zeros(len(feature_names), dtype=np.float64)
    for k in range(spec.n_experts):
        sw = w_train[:, k].astype(np.float64)
        eff_n = effective_n(sw)
        gate_diag[k]["skipped"] = int(eff_n < args.min_effective_n)
        gate_diag[k]["variant"] = spec.name
        gate_diag[k]["split"] = split_name
        gate_diag[k]["gate_mode"] = spec.gate_mode
        gate_diag[k]["n_experts"] = spec.n_experts
        gate_diag[k]["global_trees"] = args.global_trees
        gate_diag[k]["expert_trees"] = args.expert_trees
        gate_diag[k]["expert_min_leaf"] = args.expert_min_leaf

        if eff_n < args.min_effective_n or sw.sum() <= 0:
            continue

        expert_rf = make_rf(
            n_trees=args.expert_trees,
            random_state=args.random_state + 100 + k,
            n_jobs=args.n_jobs,
            min_samples_leaf=args.expert_min_leaf,
            oob_score=False,
        )
        expert_rf.fit(X_train, residual_train, sample_weight=sw)
        expert_test_preds[:, k] = expert_rf.predict(X_test).astype(np.float32)
        weighted_expert_importance += (sw.mean() * expert_rf.feature_importances_)

        for feature, importance in sorted(
            zip(feature_names, expert_rf.feature_importances_),
            key=lambda pair: pair[1],
            reverse=True,
        ):
            importance_rows.append(
                {
                    "variant": spec.name,
                    "source": f"expert_{k + 1}",
                    "expert": int(k + 1),
                    "feature": feature,
                    "importance": float(importance),
                    "group": feature_group(feature),
                }
            )

    correction_test = np.sum(w_test * expert_test_preds, axis=1).astype(np.float32)
    y_pred = (base_test_pred + correction_test).astype(np.float32)
    residuals = (y_test - y_pred).astype(np.float32)

    if weighted_expert_importance.sum() > 0:
        weighted_expert_importance = weighted_expert_importance / weighted_expert_importance.sum()
        for feature, importance in sorted(
            zip(feature_names, weighted_expert_importance),
            key=lambda pair: pair[1],
            reverse=True,
        ):
            importance_rows.append(
                {
                    "variant": spec.name,
                    "source": "experts_weighted",
                    "expert": -1,
                    "feature": feature,
                    "importance": float(importance),
                    "group": feature_group(feature),
                }
            )

    metrics = {
        "variant": spec.name,
        "family": spec.family,
        "baseline_name": spec.baseline_name,
        "parent_name": spec.parent_name,
        "model_kind": spec.model_kind,
        "base_key": spec.base_key,
        "smooth_mode": spec.smooth_mode,
        "gate_mode": spec.gate_mode,
        "n_experts": spec.n_experts,
        "split": split_name,
        "n_features": int(X_train.shape[1]),
        "n_gate_features": int(gate_train.shape[1]),
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "moran_i": float(moran_i(residuals, coords_test, k=MORAN_K)),
        "mean_gate_entropy_test": float(
            -np.sum(w_test * np.log(np.clip(w_test, 1e-12, 1.0)), axis=1).mean()
        ),
        "mean_gate_max_weight_test": float(w_test.max(axis=1).mean()),
    }

    pred_df = None
    if args.save_predictions:
        pred_df = pd.DataFrame(
            {
                "x": coords_test[:, 0],
                "y": coords_test[:, 1],
                "observed": y_test,
                "predicted": y_pred,
                "residual": residuals,
                "base_pred": base_test_pred,
                "expert_correction": correction_test,
            }
        )
        for k in range(spec.n_experts):
            pred_df[f"gate_w_{k + 1}"] = w_test[:, k]
            pred_df[f"expert_pred_{k + 1}"] = expert_test_preds[:, k]

    return metrics, importance_rows, gate_diag, gate_centers, pred_df


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

    group_array = work[group_cols].to_numpy(dtype=np.float32) if group_cols else np.zeros((len(work), 0))
    space_array = work[SPACE_COLS].to_numpy(dtype=np.float32)

    metadata = {
        "date": TODAY,
        "input": str(args.input),
        "rows_used": int(len(work)),
        "response": args.response,
        "random_state": args.random_state,
        "block_km": args.block_km,
        "global_trees": args.global_trees,
        "expert_trees": args.expert_trees,
        "expert_min_leaf": args.expert_min_leaf,
        "max_k": args.max_k,
        "sample_n": args.sample_n,
        "gate_fit_sample_n": args.gate_fit_sample_n,
        "gate_softness": args.gate_softness,
        "min_effective_n": args.min_effective_n,
        "stage5b_predictors": STAGE5B_PREDS,
        "smooth_radius_km": SMOOTH_RADIUS_KM,
        "variants": [asdict(spec) for spec in variant_specs],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    metric_rows: list[dict] = []
    importance_rows: list[dict] = []
    gate_diag_rows: list[dict] = []
    gate_center_rows: list[dict] = []

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

        matrix_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, list[str]]] = {}
        for base_key in ["m2", "m3"]:
            matrix_cache[(base_key, "none")] = build_model_matrix(
                base_train=X_map[base_key][tr],
                base_test=X_map[base_key][te],
                smooth_train=None,
                smooth_test=None,
                base_features=feature_name_map[base_key],
            )
            matrix_cache[(base_key, "mgwr")] = build_model_matrix(
                base_train=X_map[base_key][tr],
                base_test=X_map[base_key][te],
                smooth_train=smooth_bundle["mgwr_train"],
                smooth_test=smooth_bundle["mgwr_test"],
                base_features=feature_name_map[base_key],
            )

        gate_cache: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {
            "raw": (
                X_map["stage5b"][tr],
                X_map["stage5b"][te],
                list(STAGE5B_PREDS),
            ),
            "rawsmooth": (
                np.hstack([X_map["stage5b"][tr], smooth_bundle["mgwr_train"]]).astype(np.float32),
                np.hstack([X_map["stage5b"][te], smooth_bundle["mgwr_test"]]).astype(np.float32),
                list(STAGE5B_PREDS) + block_feature_names(),
            ),
            "rawsmooth_space": (
                np.hstack(
                    [X_map["stage5b"][tr], smooth_bundle["mgwr_train"], space_array[tr]]
                ).astype(np.float32),
                np.hstack(
                    [X_map["stage5b"][te], smooth_bundle["mgwr_test"], space_array[te]]
                ).astype(np.float32),
                list(STAGE5B_PREDS) + block_feature_names() + list(SPACE_COLS),
            ),
            "rawsmooth_group": (
                np.hstack(
                    [X_map["stage5b"][tr], smooth_bundle["mgwr_train"], group_array[tr]]
                ).astype(np.float32),
                np.hstack(
                    [X_map["stage5b"][te], smooth_bundle["mgwr_test"], group_array[te]]
                ).astype(np.float32),
                list(STAGE5B_PREDS) + block_feature_names() + list(group_cols),
            ),
        }

        for spec in variant_specs:
            X_train, X_test, feature_names = matrix_cache[(spec.base_key, spec.smooth_mode)]
            if spec.model_kind == "rf":
                rf = make_rf(
                    n_trees=args.global_trees,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    min_samples_leaf=1,
                    oob_score=False,
                )
                rf.fit(X_train, y[tr])
                y_pred = rf.predict(X_test).astype(np.float32)
                residuals = (y[te] - y_pred).astype(np.float32)
                metrics = {
                    "variant": spec.name,
                    "family": spec.family,
                    "baseline_name": spec.baseline_name,
                    "parent_name": spec.parent_name,
                    "model_kind": spec.model_kind,
                    "base_key": spec.base_key,
                    "smooth_mode": spec.smooth_mode,
                    "gate_mode": spec.gate_mode,
                    "n_experts": spec.n_experts,
                    "split": split_name,
                    "n_features": int(X_train.shape[1]),
                    "n_gate_features": 0,
                    "r2": float(r2_score(y[te], y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y[te], y_pred))),
                    "moran_i": float(moran_i(residuals, coords[te], k=MORAN_K)),
                    "mean_gate_entropy_test": float("nan"),
                    "mean_gate_max_weight_test": float("nan"),
                }
                metric_rows.append(metrics)
                print(
                    f"{spec.name:26s} "
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
                                "source": "model",
                                "expert": 0,
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
                continue

            gate_train, gate_test, gate_features = gate_cache[spec.gate_mode]
            metrics, imp_rows, diag_rows, center_rows, pred_df = run_softmoe_variant(
                spec=spec,
                X_train=X_train,
                X_test=X_test,
                feature_names=feature_names,
                gate_train=gate_train,
                gate_test=gate_test,
                gate_features=gate_features,
                y_train=y[tr],
                y_test=y[te],
                coords_test=coords[te],
                args=args,
                split_name=split_name,
            )
            metric_rows.append(metrics)
            gate_diag_rows.extend(diag_rows)
            gate_center_rows.extend(center_rows)
            if split_name == "block":
                importance_rows.extend(imp_rows)
            print(
                f"{spec.name:26s} "
                f"R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} "
                f"MoranI={metrics['moran_i']:.4f} p={metrics['n_features']} "
                f"gate={metrics['n_gate_features']} K={spec.n_experts}",
                flush=True,
            )
            if pred_df is not None:
                pred_df.to_csv(
                    args.output_dir / f"{spec.name}_{split_name}_predictions.csv",
                    index=False,
                )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.output_dir / "metrics_long.csv", index=False)

    summary_df = metrics_df.copy()
    lookup = summary_df.set_index(["variant", "split"])[["r2", "rmse", "moran_i"]]
    delta_cols = {
        "delta_r2_vs_family_base": [],
        "delta_rmse_vs_family_base": [],
        "delta_moran_vs_family_base": [],
        "delta_r2_vs_parent": [],
        "delta_rmse_vs_parent": [],
        "delta_moran_vs_parent": [],
        "delta_r2_vs_m2_baseline": [],
        "delta_rmse_vs_m2_baseline": [],
        "delta_moran_vs_m2_baseline": [],
    }
    for _, row in summary_df.iterrows():
        family_key = (row["baseline_name"], row["split"])
        parent_key = (row["parent_name"], row["split"])
        m2_key = ("m2_baseline", row["split"])

        def add_delta(prefix: str, key: tuple[str, str]) -> None:
            if key in lookup.index:
                base = lookup.loc[key]
                delta_cols[f"delta_r2_{prefix}"].append(float(row["r2"] - base["r2"]))
                delta_cols[f"delta_rmse_{prefix}"].append(float(row["rmse"] - base["rmse"]))
                delta_cols[f"delta_moran_{prefix}"].append(float(row["moran_i"] - base["moran_i"]))
            else:
                delta_cols[f"delta_r2_{prefix}"].append(float("nan"))
                delta_cols[f"delta_rmse_{prefix}"].append(float("nan"))
                delta_cols[f"delta_moran_{prefix}"].append(float("nan"))

        add_delta("vs_family_base", family_key)
        add_delta("vs_parent", parent_key)
        add_delta("vs_m2_baseline", m2_key)

    for col, values in delta_cols.items():
        summary_df[col] = values

    split_size_map = {
        name: {"test_rows": int(mask.sum()), "train_rows": int((~mask).sum())}
        for name, mask in split_masks.items()
    }
    summary_df["train_rows"] = summary_df["split"].map(
        {name: sizes["train_rows"] for name, sizes in split_size_map.items()}
    )
    summary_df["test_rows"] = summary_df["split"].map(
        {name: sizes["test_rows"] for name, sizes in split_size_map.items()}
    )
    summary_df["rows"] = int(len(work))
    summary_df = summary_df[
        [
            "variant",
            "family",
            "model_kind",
            "split",
            "rows",
            "train_rows",
            "test_rows",
            "n_features",
            "n_gate_features",
            "smooth_mode",
            "gate_mode",
            "n_experts",
            "r2",
            "rmse",
            "moran_i",
            "mean_gate_entropy_test",
            "mean_gate_max_weight_test",
            "delta_r2_vs_family_base",
            "delta_rmse_vs_family_base",
            "delta_moran_vs_family_base",
            "delta_r2_vs_parent",
            "delta_rmse_vs_parent",
            "delta_moran_vs_parent",
            "delta_r2_vs_m2_baseline",
            "delta_rmse_vs_m2_baseline",
            "delta_moran_vs_m2_baseline",
        ]
    ].sort_values(["split", "family", "variant"])
    summary_df.to_csv(args.output_dir / "metrics_summary.csv", index=False)

    wide = summary_df.pivot(
        index=[
            "variant",
            "family",
            "model_kind",
            "smooth_mode",
            "gate_mode",
            "n_experts",
            "n_features",
            "n_gate_features",
        ],
        columns="split",
        values=[
            "r2",
            "rmse",
            "moran_i",
            "mean_gate_entropy_test",
            "mean_gate_max_weight_test",
            "delta_r2_vs_m2_baseline",
            "delta_rmse_vs_m2_baseline",
            "delta_moran_vs_m2_baseline",
            "delta_r2_vs_parent",
            "delta_rmse_vs_parent",
            "delta_moran_vs_parent",
        ],
    )
    wide.columns = ["_".join(col) for col in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(args.output_dir / "metrics_wide.csv", index=False)

    if importance_rows:
        importance_df = pd.DataFrame(importance_rows)
        importance_df.to_csv(args.output_dir / "block_feature_importance.csv", index=False)

        group_importance = (
            importance_df.groupby(["variant", "source", "group"], as_index=False)["importance"]
            .sum()
            .sort_values(["variant", "source", "importance"], ascending=[True, True, False])
        )
        group_importance.to_csv(args.output_dir / "block_group_importance.csv", index=False)

    if gate_diag_rows:
        pd.DataFrame(gate_diag_rows).to_csv(args.output_dir / "gate_diagnostics.csv", index=False)
    if gate_center_rows:
        pd.DataFrame(gate_center_rows).to_csv(args.output_dir / "gate_centers.csv", index=False)

    block_rank = (
        summary_df[summary_df["split"] == "block"]
        .sort_values(["r2", "moran_i"], ascending=[False, True])
        .reset_index(drop=True)
    )

    report_lines = [
        f"# Western US Soft-MoE Multiscale RF Round 1 ({TODAY})",
        "",
        f"- Input: `{args.input}`",
        f"- Response: `{args.response}`",
        f"- Rows used: `{len(work):,}`",
        f"- Block size: `{args.block_km:g} km`",
        f"- Global RF trees: `{args.global_trees}`",
        f"- Expert RF trees: `{args.expert_trees}`",
        f"- Expert min leaf: `{args.expert_min_leaf}`",
        f"- Gate softness: `{args.gate_softness}`",
        f"- Gate fit sample: `{args.gate_fit_sample_n}`",
        "",
        "## Variants",
        "",
    ]
    for spec in variant_specs:
        report_lines.append(f"- `{spec.name}`: {spec.note}")
    report_lines.extend(
        [
            "",
            "## Block Ranking",
            "",
            "```text",
            block_rank.round(4).to_string(index=False),
            "```",
            "",
            "## Full Summary",
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
