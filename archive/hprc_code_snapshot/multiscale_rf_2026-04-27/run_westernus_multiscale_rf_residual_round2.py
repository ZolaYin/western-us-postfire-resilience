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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split


TODAY = date.today().strftime("%Y-%m-%d")
INPUT = Path("westernus_current_candidate_table_plus_regions.parquet")

RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_TREES = 300
RESID_TREES = 200
MAX_K = 5000
MORAN_K = 8
INNER_CV_FOLDS = 5
DEFAULT_BLOCK_KM_LIST = [100.0, 75.0]
RESID_MIN_LEAF = 100
RESID_MAX_DEPTH = 14
RESID_MAX_FEATURES = 0.5

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

CORE6_PREDS = [
    "FS_CBH_t0agg_z",
    "HUM_roaddens_r5km_z",
    "TS_SOC_0_30cm_z",
    "CLIM_pr_sum_pre_z",
    "TS_elev_m_z",
    "FS_TCC_t0_z",
]
INTERACT4_PREDS = [
    "FS_CBH_t0agg_z",
    "HUM_roaddens_r5km_z",
    "TS_SOC_0_30cm_z",
    "CLIM_pr_sum_pre_z",
]
COUPLING_SPECS = [
    ("localcorr_elev_pr", "TS_elev_m_z", "CLIM_pr_sum_pre_z", 600),
    ("localcorr_tcc_cbh", "FS_TCC_t0_z", "FS_CBH_t0agg_z", 600),
    ("localcorr_soc_pr", "TS_SOC_0_30cm_z", "CLIM_pr_sum_pre_z", 300),
]

LOCAL_RADIUS_KM = {
    "FS_CBH_t0agg_z": 150,
    "HUM_roaddens_r5km_z": 150,
    "TS_SOC_0_30cm_z": 300,
    "CLIM_pr_sum_pre_z": 300,
    "TS_elev_m_z": 600,
    "FS_TCC_t0_z": 600,
}
BROAD_RADIUS_KM = 1200
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
    residual_mode: str
    note: str


@dataclass(frozen=True)
class SplitSpec:
    name: str
    split_kind: str
    block_km: float | None
    inner_block_km: float


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
        "Main-text M2 baseline: baseline predictors + FS_EVT_group_class dummies.",
    ),
    VariantSpec(
        "m2_mgwr_anchor",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "rf",
        "m2",
        "mgwr",
        "none",
        "Anchor only: M2 baseline + MGWR-matched smooth features for all 11 stage5b variables.",
    ),
    VariantSpec(
        "m2_resid_localmean6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localmean6",
        "Residual RF on six local-mean multiscale context features.",
    ),
    VariantSpec(
        "m2_resid_localanom6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localanom6",
        "Residual RF on six local anomaly features (raw minus local mean).",
    ),
    VariantSpec(
        "m2_resid_localbroad6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localbroad6",
        "Residual RF on six local-minus-broad context contrasts.",
    ),
    VariantSpec(
        "m2_resid_localsd6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localsd6",
        "Residual RF on six local neighborhood standard-deviation features.",
    ),
    VariantSpec(
        "m2_resid_localz6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localz6",
        "Residual RF on six standardized local anomaly features.",
    ),
    VariantSpec(
        "m2_resid_mean_anom6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "mean_anom6",
        "Residual RF on six local means plus six local anomalies.",
    ),
    VariantSpec(
        "m2_resid_localanom_lisa6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localanom_lisa6",
        "Residual RF on six local anomalies plus six predictor-local LISA features.",
    ),
    VariantSpec(
        "m2_resid_localanom_lisa_int4",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localanom_lisa_int4",
        "Residual RF on six local anomalies, six predictor-local LISA features, and four raw-by-context interaction pairs.",
    ),
    VariantSpec(
        "m2_resid_localanom_lisa_int4_couple3",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localanom_lisa_int4_couple3",
        "Residual RF on six local anomalies, six predictor-local LISA features, four raw-by-context interaction pairs, and three local coupling features.",
    ),
    VariantSpec(
        "m2_resid_localanom_lisa_sd6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "localanom_lisa_sd6",
        "Residual RF on six local anomalies, six predictor-local LISA features, and six local SD features.",
    ),
    VariantSpec(
        "m2_resid_mean_anom_lisa6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "mean_anom_lisa6",
        "Residual RF on six local means, six local anomalies, and six predictor-local LISA features.",
    ),
    VariantSpec(
        "m2_resid_mean_anom_lisa_sd6",
        "m2",
        "m2_baseline",
        "m2_baseline",
        "resid_rf",
        "m2",
        "none",
        "mean_anom_lisa_sd6",
        "Residual RF on six local means, six local anomalies, six predictor-local LISA features, and six local SD features.",
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
        "Main-text M3 baseline: baseline predictors + raw EVT code dummies.",
    ),
    VariantSpec(
        "m3_mgwr_anchor",
        "m3",
        "m3_baseline",
        "m3_baseline",
        "rf",
        "m3",
        "mgwr",
        "none",
        "Anchor only: M3 baseline + MGWR-matched smooth features for all 11 stage5b variables.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Residual multiscale RF round 2 for Western US post-fire forest resistance. "
            "Fits M2 as the main model and uses restrained multiscale features only to "
            "model M2 residual structure, while reporting both 100-km primary block CV "
            "and 75-km sensitivity."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response", default="Resistance")
    parser.add_argument("--base-trees", type=int, default=BASE_TREES)
    parser.add_argument("--resid-trees", type=int, default=RESID_TREES)
    parser.add_argument("--resid-min-leaf", type=int, default=RESID_MIN_LEAF)
    parser.add_argument("--resid-max-depth", type=int, default=RESID_MAX_DEPTH)
    parser.add_argument("--resid-max-features", type=float, default=RESID_MAX_FEATURES)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--inner-cv-folds", type=int, default=INNER_CV_FOLDS)
    parser.add_argument("--sample-n", type=int, default=None)
    parser.add_argument(
        "--block-km-list",
        nargs="+",
        type=float,
        default=DEFAULT_BLOCK_KM_LIST,
        help="Primary and sensitivity block sizes, e.g. 100 75.",
    )
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


def block_groups_from_coords(coords: np.ndarray, block_km: float) -> np.ndarray:
    block_m = block_km * 1000.0
    return np.array(
        [
            f"{int(np.floor(x / block_m))}_{int(np.floor(y / block_m))}"
            for x, y in coords
        ],
        dtype=object,
    )


def build_split_specs(block_km_list: list[float]) -> list[SplitSpec]:
    unique_blocks: list[float] = []
    for val in block_km_list:
        fval = float(val)
        if fval not in unique_blocks:
            unique_blocks.append(fval)
    primary = unique_blocks[0]
    specs = [SplitSpec("random", "random", None, primary)]
    for block_km in unique_blocks:
        specs.append(
            SplitSpec(
                name=f"block_{int(block_km)}km",
                split_kind="block",
                block_km=block_km,
                inner_block_km=block_km,
            )
        )
    return specs


def build_split_masks(
    coords: np.ndarray, split_specs: list[SplitSpec], random_state: int
) -> dict[str, np.ndarray]:
    idx = np.arange(len(coords))
    _, rand_te = train_test_split(idx, test_size=TEST_SIZE, random_state=random_state)
    masks = {"random": np.zeros(len(coords), dtype=bool)}
    masks["random"][rand_te] = True
    for spec in split_specs:
        if spec.split_kind != "block" or spec.block_km is None:
            continue
        groups = block_groups_from_coords(coords, spec.block_km)
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=random_state)
        _, blk_te = next(gss.split(idx, groups=groups))
        mask = np.zeros(len(coords), dtype=bool)
        mask[blk_te] = True
        masks[spec.name] = mask
    return masks


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


def standardize_train_test(
    train_X: np.ndarray, test_X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0, ddof=1)
    std = np.where((std == 0) | ~np.isfinite(std), 1.0, std)
    train_std = ((train_X - mean) / std).astype(np.float32)
    test_std = ((test_X - mean) / std).astype(np.float32)
    return train_std, test_std, mean.astype(np.float32), std.astype(np.float32)


def build_mgwr_smooth_bundle(
    train_coords: np.ndarray,
    stage5b_train: np.ndarray,
    test_coords: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    return matched_train, matched_test


def build_core_context_bundle(
    train_coords: np.ndarray,
    core_train: np.ndarray,
    test_coords: np.ndarray,
    max_k: int,
) -> dict[str, np.ndarray]:
    local_train = np.zeros_like(core_train, dtype=np.float32)
    local_test = np.zeros((len(test_coords), core_train.shape[1]), dtype=np.float32)
    localsq_train = np.zeros_like(core_train, dtype=np.float32)
    localsq_test = np.zeros((len(test_coords), core_train.shape[1]), dtype=np.float32)

    grouped_indices: dict[int, list[int]] = {}
    for j, pred in enumerate(CORE6_PREDS):
        grouped_indices.setdefault(LOCAL_RADIUS_KM[pred], []).append(j)

    for radius_km, js in sorted(grouped_indices.items()):
        s_train, s_test = smooth_train_test(
            train_coords=train_coords,
            train_X=core_train[:, js],
            test_coords=test_coords,
            radius_m=radius_km * 1000.0,
            max_k=max_k,
        )
        s2_train, s2_test = smooth_train_test(
            train_coords=train_coords,
            train_X=np.square(core_train[:, js]).astype(np.float32),
            test_coords=test_coords,
            radius_m=radius_km * 1000.0,
            max_k=max_k,
        )
        local_train[:, js] = s_train
        local_test[:, js] = s_test
        localsq_train[:, js] = s2_train
        localsq_test[:, js] = s2_test

    broad_train, broad_test = smooth_train_test(
        train_coords=train_coords,
        train_X=core_train,
        test_coords=test_coords,
        radius_m=BROAD_RADIUS_KM * 1000.0,
        max_k=max_k,
    )
    localvar_train = np.maximum(localsq_train - np.square(local_train), 1e-6).astype(np.float32)
    localvar_test = np.maximum(localsq_test - np.square(local_test), 1e-6).astype(np.float32)
    localsd_train = np.sqrt(localvar_train).astype(np.float32)
    localsd_test = np.sqrt(localvar_test).astype(np.float32)

    return {
        "localmean_train": local_train.astype(np.float32),
        "localmean_test": local_test.astype(np.float32),
        "localanom_train": (core_train - local_train).astype(np.float32),
        "localanom_test": (np.zeros((len(test_coords), core_train.shape[1]), dtype=np.float32)),
        "localsd_train": localsd_train,
        "localsd_test": localsd_test,
        "localz_train": ((core_train - local_train) / np.maximum(localsd_train, 1e-3)).astype(
            np.float32
        ),
        "localz_test": (np.zeros((len(test_coords), core_train.shape[1]), dtype=np.float32)),
        "localbroad_train": (local_train - broad_train).astype(np.float32),
        "localbroad_test": (local_test - broad_test).astype(np.float32),
    }


def fill_test_localanom(
    context_bundle: dict[str, np.ndarray],
    core_test: np.ndarray,
) -> None:
    context_bundle["localanom_test"] = (core_test - context_bundle["localmean_test"]).astype(
        np.float32
    )
    context_bundle["localz_test"] = (
        context_bundle["localanom_test"] / np.maximum(context_bundle["localsd_test"], 1e-3)
    ).astype(np.float32)


def build_lisa_bundle(
    train_coords: np.ndarray,
    core_train: np.ndarray,
    test_coords: np.ndarray,
    core_test: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    z_train, z_test, _, _ = standardize_train_test(core_train, core_test)
    lag_train = np.zeros_like(z_train, dtype=np.float32)
    lag_test = np.zeros((len(test_coords), core_train.shape[1]), dtype=np.float32)

    grouped_indices: dict[int, list[int]] = {}
    for j, pred in enumerate(CORE6_PREDS):
        grouped_indices.setdefault(LOCAL_RADIUS_KM[pred], []).append(j)

    for radius_km, js in sorted(grouped_indices.items()):
        s_train, s_test = smooth_train_test(
            train_coords=train_coords,
            train_X=z_train[:, js],
            test_coords=test_coords,
            radius_m=radius_km * 1000.0,
            max_k=max_k,
        )
        lag_train[:, js] = s_train
        lag_test[:, js] = s_test

    lisa_train = (z_train * lag_train).astype(np.float32)
    lisa_test = (z_test * lag_test).astype(np.float32)
    return lisa_train, lisa_test


def build_interaction_bundle(
    context_bundle: dict[str, np.ndarray],
    core_train: np.ndarray,
    core_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_to_idx = {pred: idx for idx, pred in enumerate(CORE6_PREDS)}
    train_cols: list[np.ndarray] = []
    test_cols: list[np.ndarray] = []
    for pred in INTERACT4_PREDS:
        j = pred_to_idx[pred]
        raw_train = core_train[:, j : j + 1]
        raw_test = core_test[:, j : j + 1]
        train_cols.append((raw_train * context_bundle["localanom_train"][:, j : j + 1]).astype(np.float32))
        train_cols.append((raw_train * context_bundle["lisa_train"][:, j : j + 1]).astype(np.float32))
        test_cols.append((raw_test * context_bundle["localanom_test"][:, j : j + 1]).astype(np.float32))
        test_cols.append((raw_test * context_bundle["lisa_test"][:, j : j + 1]).astype(np.float32))
    return np.hstack(train_cols).astype(np.float32), np.hstack(test_cols).astype(np.float32)


def build_coupling_bundle(
    train_coords: np.ndarray,
    core_train: np.ndarray,
    test_coords: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred_to_idx = {pred: idx for idx, pred in enumerate(CORE6_PREDS)}
    train_out = np.zeros((len(train_coords), len(COUPLING_SPECS)), dtype=np.float32)
    test_out = np.zeros((len(test_coords), len(COUPLING_SPECS)), dtype=np.float32)

    specs_by_radius: dict[int, list[tuple[int, str, str, str]]] = {}
    for out_idx, (name, pred_a, pred_b, radius_km) in enumerate(COUPLING_SPECS):
        specs_by_radius.setdefault(radius_km, []).append((out_idx, name, pred_a, pred_b))

    for radius_km, specs in sorted(specs_by_radius.items()):
        moment_arrays: list[np.ndarray] = []
        moment_lookup: dict[tuple[str, str], int] = {}

        def register_moment(key: tuple[str, str], values: np.ndarray) -> None:
            if key not in moment_lookup:
                moment_lookup[key] = len(moment_arrays)
                moment_arrays.append(values.astype(np.float32))

        for _, name, pred_a, pred_b in specs:
            a = core_train[:, pred_to_idx[pred_a]]
            b = core_train[:, pred_to_idx[pred_b]]
            register_moment((pred_a, "mean"), a)
            register_moment((pred_b, "mean"), b)
            register_moment((pred_a, "sq"), np.square(a))
            register_moment((pred_b, "sq"), np.square(b))
            register_moment((name, "cross"), a * b)

        moment_matrix = np.column_stack(moment_arrays).astype(np.float32)
        smooth_train, smooth_test = smooth_train_test(
            train_coords=train_coords,
            train_X=moment_matrix,
            test_coords=test_coords,
            radius_m=radius_km * 1000.0,
            max_k=max_k,
        )

        for out_idx, name, pred_a, pred_b in specs:
            mean_a_train = smooth_train[:, moment_lookup[(pred_a, "mean")]]
            mean_b_train = smooth_train[:, moment_lookup[(pred_b, "mean")]]
            mean_a2_train = smooth_train[:, moment_lookup[(pred_a, "sq")]]
            mean_b2_train = smooth_train[:, moment_lookup[(pred_b, "sq")]]
            cross_train = smooth_train[:, moment_lookup[(name, "cross")]]

            mean_a_test = smooth_test[:, moment_lookup[(pred_a, "mean")]]
            mean_b_test = smooth_test[:, moment_lookup[(pred_b, "mean")]]
            mean_a2_test = smooth_test[:, moment_lookup[(pred_a, "sq")]]
            mean_b2_test = smooth_test[:, moment_lookup[(pred_b, "sq")]]
            cross_test = smooth_test[:, moment_lookup[(name, "cross")]]

            var_a_train = np.maximum(mean_a2_train - np.square(mean_a_train), 1e-6)
            var_b_train = np.maximum(mean_b2_train - np.square(mean_b_train), 1e-6)
            var_a_test = np.maximum(mean_a2_test - np.square(mean_a_test), 1e-6)
            var_b_test = np.maximum(mean_b2_test - np.square(mean_b_test), 1e-6)

            corr_train = (cross_train - mean_a_train * mean_b_train) / np.sqrt(var_a_train * var_b_train)
            corr_test = (cross_test - mean_a_test * mean_b_test) / np.sqrt(var_a_test * var_b_test)
            train_out[:, out_idx] = np.clip(corr_train, -1.0, 1.0).astype(np.float32)
            test_out[:, out_idx] = np.clip(corr_test, -1.0, 1.0).astype(np.float32)

    return train_out, test_out


def mgwr_feature_names() -> list[str]:
    return [f"{pred}_s{SMOOTH_RADIUS_KM[pred]}km_mgwr" for pred in STAGE5B_PREDS]


def residual_feature_bundle(context_bundle: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    mean_names = [f"{pred}_localmean" for pred in CORE6_PREDS]
    anom_names = [f"{pred}_localanom" for pred in CORE6_PREDS]
    broad_names = [f"{pred}_localminusbroad" for pred in CORE6_PREDS]
    sd_names = [f"{pred}_localsd" for pred in CORE6_PREDS]
    z_names = [f"{pred}_localz" for pred in CORE6_PREDS]
    lisa_names = [f"{pred}_lisa" for pred in CORE6_PREDS]
    interaction_names = [
        name
        for pred in INTERACT4_PREDS
        for name in (f"{pred}_x_localanom", f"{pred}_x_lisa")
    ]
    coupling_names = [name for name, _, _, _ in COUPLING_SPECS]

    return {
        "localmean6": (
            context_bundle["localmean_train"],
            context_bundle["localmean_test"],
            mean_names,
        ),
        "localanom6": (
            context_bundle["localanom_train"],
            context_bundle["localanom_test"],
            anom_names,
        ),
        "localbroad6": (
            context_bundle["localbroad_train"],
            context_bundle["localbroad_test"],
            broad_names,
        ),
        "localsd6": (
            context_bundle["localsd_train"],
            context_bundle["localsd_test"],
            sd_names,
        ),
        "localz6": (
            context_bundle["localz_train"],
            context_bundle["localz_test"],
            z_names,
        ),
        "mean_anom6": (
            np.hstack([context_bundle["localmean_train"], context_bundle["localanom_train"]]).astype(
                np.float32
            ),
            np.hstack([context_bundle["localmean_test"], context_bundle["localanom_test"]]).astype(
                np.float32
            ),
            mean_names + anom_names,
        ),
        "localanom_lisa6": (
            np.hstack([context_bundle["localanom_train"], context_bundle["lisa_train"]]).astype(
                np.float32
            ),
            np.hstack([context_bundle["localanom_test"], context_bundle["lisa_test"]]).astype(
                np.float32
            ),
            anom_names + lisa_names,
        ),
        "localanom_lisa_sd6": (
            np.hstack(
                [
                    context_bundle["localanom_train"],
                    context_bundle["lisa_train"],
                    context_bundle["localsd_train"],
                ]
            ).astype(np.float32),
            np.hstack(
                [
                    context_bundle["localanom_test"],
                    context_bundle["lisa_test"],
                    context_bundle["localsd_test"],
                ]
            ).astype(np.float32),
            anom_names + lisa_names + sd_names,
        ),
        "mean_anom_lisa6": (
            np.hstack(
                [
                    context_bundle["localmean_train"],
                    context_bundle["localanom_train"],
                    context_bundle["lisa_train"],
                ]
            ).astype(np.float32),
            np.hstack(
                [
                    context_bundle["localmean_test"],
                    context_bundle["localanom_test"],
                    context_bundle["lisa_test"],
                ]
            ).astype(np.float32),
            mean_names + anom_names + lisa_names,
        ),
        "mean_anom_lisa_sd6": (
            np.hstack(
                [
                    context_bundle["localmean_train"],
                    context_bundle["localanom_train"],
                    context_bundle["lisa_train"],
                    context_bundle["localsd_train"],
                ]
            ).astype(np.float32),
            np.hstack(
                [
                    context_bundle["localmean_test"],
                    context_bundle["localanom_test"],
                    context_bundle["lisa_test"],
                    context_bundle["localsd_test"],
                ]
            ).astype(np.float32),
            mean_names + anom_names + lisa_names + sd_names,
        ),
        "localanom_lisa_int4": (
            np.hstack(
                [
                    context_bundle["localanom_train"],
                    context_bundle["lisa_train"],
                    context_bundle["interaction_train"],
                ]
            ).astype(np.float32),
            np.hstack(
                [
                    context_bundle["localanom_test"],
                    context_bundle["lisa_test"],
                    context_bundle["interaction_test"],
                ]
            ).astype(np.float32),
            anom_names + lisa_names + interaction_names,
        ),
        "localanom_lisa_int4_couple3": (
            np.hstack(
                [
                    context_bundle["localanom_train"],
                    context_bundle["lisa_train"],
                    context_bundle["interaction_train"],
                    context_bundle["coupling_train"],
                ]
            ).astype(np.float32),
            np.hstack(
                [
                    context_bundle["localanom_test"],
                    context_bundle["lisa_test"],
                    context_bundle["interaction_test"],
                    context_bundle["coupling_test"],
                ]
            ).astype(np.float32),
            anom_names + lisa_names + interaction_names + coupling_names,
        ),
    }


def feature_group(feature: str) -> str:
    if feature in STAGE5B_PREDS:
        return "stage5b_raw"
    if feature.endswith("_mgwr"):
        return "stage5b_smooth_mgwr"
    if feature.endswith("_x_localanom"):
        return "resid_interaction_anom"
    if feature.endswith("_x_lisa"):
        return "resid_interaction_lisa"
    if feature.startswith("localcorr_"):
        return "resid_localcoupling"
    if feature.endswith("_localmean"):
        return "resid_localmean"
    if feature.endswith("_localanom"):
        return "resid_localanom"
    if feature.endswith("_localminusbroad"):
        return "resid_localcontrast"
    if feature.endswith("_localsd"):
        return "resid_localsd"
    if feature.endswith("_localz"):
        return "resid_localz"
    if feature.endswith("_lisa"):
        return "resid_lisa"
    if feature.startswith("EVT_group_"):
        return "forest_type_group"
    if feature.startswith("EVT_code_"):
        return "forest_type_code"
    return "baseline_other"


def get_variant_specs(selected: list[str]) -> list[VariantSpec]:
    spec_map = {spec.name: spec for spec in VARIANT_SPECS}
    unknown = [name for name in selected if name not in spec_map]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return [spec_map[name] for name in selected]


def make_rf(
    n_trees: int,
    random_state: int,
    n_jobs: int,
    min_samples_leaf: int = 1,
    max_depth: int | None = None,
    max_features: float | str | None = 1.0,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_trees,
        random_state=random_state,
        n_jobs=n_jobs,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_features=max_features,
    )


def crossfit_base_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    coords_train: np.ndarray,
    inner_block_km: float,
    inner_cv_folds: int,
    n_trees: int,
    random_state: int,
    n_jobs: int,
) -> np.ndarray:
    groups = block_groups_from_coords(coords_train, inner_block_km)
    unique_groups = np.unique(groups)
    n_splits = min(inner_cv_folds, len(unique_groups))
    if n_splits < 2:
        model = make_rf(
            n_trees=n_trees,
            random_state=random_state,
            n_jobs=n_jobs,
            min_samples_leaf=1,
            max_depth=None,
            max_features=1.0,
        )
        model.fit(X_train, y_train)
        return model.predict(X_train).astype(np.float32)

    oof = np.full(len(X_train), np.nan, dtype=np.float32)
    gkf = GroupKFold(n_splits=n_splits)
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups)):
        model = make_rf(
            n_trees=n_trees,
            random_state=random_state + fold_idx,
            n_jobs=n_jobs,
            min_samples_leaf=1,
            max_depth=None,
            max_features=1.0,
        )
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof[va_idx] = model.predict(X_train[va_idx]).astype(np.float32)

    if np.isnan(oof).any():
        full_model = make_rf(
            n_trees=n_trees,
            random_state=random_state + 999,
            n_jobs=n_jobs,
            min_samples_leaf=1,
            max_depth=None,
            max_features=1.0,
        )
        full_model.fit(X_train, y_train)
        missing = np.isnan(oof)
        oof[missing] = full_model.predict(X_train[missing]).astype(np.float32)
    return oof


def build_anchor_matrix(
    base_train: np.ndarray,
    base_test: np.ndarray,
    smooth_train: np.ndarray | None,
    smooth_test: np.ndarray | None,
    base_features: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_names = list(base_features)
    if smooth_train is None or smooth_test is None:
        return base_train.astype(np.float32), base_test.astype(np.float32), feature_names
    feature_names.extend(mgwr_feature_names())
    return (
        np.hstack([base_train, smooth_train]).astype(np.float32),
        np.hstack([base_test, smooth_test]).astype(np.float32),
        feature_names,
    )


def summarize_metrics(
    metrics_df: pd.DataFrame,
    variant_specs: list[VariantSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    summary_df = summary_df.sort_values(["split", "family", "variant"])
    wide = summary_df.pivot(
        index=[
            "variant",
            "family",
            "model_kind",
            "n_features",
            "residual_mode",
        ],
        columns="split",
        values=[
            "r2",
            "rmse",
            "moran_i",
            "delta_r2_vs_m2_baseline",
            "delta_rmse_vs_m2_baseline",
            "delta_moran_vs_m2_baseline",
            "delta_r2_vs_parent",
            "delta_rmse_vs_parent",
            "delta_moran_vs_parent",
        ],
    )
    wide.columns = ["_".join([str(x) for x in col if str(x) != ""]) for col in wide.columns]
    wide = wide.reset_index()
    return summary_df, wide


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_specs = get_variant_specs(args.variants)
    split_specs = build_split_specs(args.block_km_list)

    print("Loading and preparing data ...", flush=True)
    raw = pd.read_parquet(args.input)
    df = prepare(raw)

    group_cols = sorted([c for c in df.columns if c.startswith("EVT_group_")])
    code_cols = sorted([c for c in df.columns if c.startswith("EVT_code_")])

    base_feature_map = {
        "stage5b": STAGE5B_PREDS,
        "core6": CORE6_PREDS,
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

    coords = work[["x", "y"]].to_numpy(dtype=np.float64)
    y = work[args.response].to_numpy(dtype=np.float32)
    split_masks = build_split_masks(coords, split_specs, args.random_state)
    split_map = {spec.name: spec for spec in split_specs}

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
        "base_trees": args.base_trees,
        "resid_trees": args.resid_trees,
        "resid_min_leaf": args.resid_min_leaf,
        "resid_max_depth": args.resid_max_depth,
        "resid_max_features": args.resid_max_features,
        "max_k": args.max_k,
        "inner_cv_folds": args.inner_cv_folds,
        "sample_n": args.sample_n,
        "block_km_list": [spec.block_km for spec in split_specs if spec.block_km is not None],
        "core6_predictors": CORE6_PREDS,
        "interact4_predictors": INTERACT4_PREDS,
        "stage5b_predictors": STAGE5B_PREDS,
        "local_radius_km": LOCAL_RADIUS_KM,
        "broad_radius_km": BROAD_RADIUS_KM,
        "coupling_specs": COUPLING_SPECS,
        "variants": [asdict(spec) for spec in variant_specs],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    metric_rows: list[dict] = []
    importance_rows: list[dict] = []

    for split_name, is_test in split_masks.items():
        spec = split_map[split_name]
        print(f"\n=== Split: {split_name} ===", flush=True)
        tr = ~is_test
        te = is_test

        coords_train = coords[tr]
        coords_test = coords[te]
        y_train = y[tr]
        y_test = y[te]

        stage5_smooth_train, stage5_smooth_test = build_mgwr_smooth_bundle(
            train_coords=coords_train,
            stage5b_train=X_map["stage5b"][tr],
            test_coords=coords_test,
            max_k=args.max_k,
        )

        context_bundle = build_core_context_bundle(
            train_coords=coords_train,
            core_train=X_map["core6"][tr],
            test_coords=coords_test,
            max_k=args.max_k,
        )
        fill_test_localanom(context_bundle, X_map["core6"][te])
        lisa_train, lisa_test = build_lisa_bundle(
            train_coords=coords_train,
            core_train=X_map["core6"][tr],
            test_coords=coords_test,
            core_test=X_map["core6"][te],
            max_k=args.max_k,
        )
        context_bundle["lisa_train"] = lisa_train
        context_bundle["lisa_test"] = lisa_test
        interaction_train, interaction_test = build_interaction_bundle(
            context_bundle=context_bundle,
            core_train=X_map["core6"][tr],
            core_test=X_map["core6"][te],
        )
        context_bundle["interaction_train"] = interaction_train
        context_bundle["interaction_test"] = interaction_test
        coupling_train, coupling_test = build_coupling_bundle(
            train_coords=coords_train,
            core_train=X_map["core6"][tr],
            test_coords=coords_test,
            max_k=args.max_k,
        )
        context_bundle["coupling_train"] = coupling_train
        context_bundle["coupling_test"] = coupling_test
        resid_bundle = residual_feature_bundle(context_bundle)

        X_m2_train = X_map["m2"][tr]
        X_m2_test = X_map["m2"][te]
        X_m3_train = X_map["m3"][tr]
        X_m3_test = X_map["m3"][te]

        m2_base_model = make_rf(
            n_trees=args.base_trees,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            min_samples_leaf=1,
            max_depth=None,
            max_features=1.0,
        )
        m2_base_model.fit(X_m2_train, y_train)
        m2_base_test_pred = m2_base_model.predict(X_m2_test).astype(np.float32)

        m2_oof_train_pred = crossfit_base_predictions(
            X_train=X_m2_train,
            y_train=y_train,
            coords_train=coords_train,
            inner_block_km=spec.inner_block_km,
            inner_cv_folds=args.inner_cv_folds,
            n_trees=args.base_trees,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        m2_resid_target = (y_train - m2_oof_train_pred).astype(np.float32)

        matrix_cache = {
            ("m2", "none"): (X_m2_train, X_m2_test, list(feature_name_map["m2"])),
            ("m2", "mgwr"): build_anchor_matrix(
                X_m2_train, X_m2_test, stage5_smooth_train, stage5_smooth_test, feature_name_map["m2"]
            ),
            ("m3", "none"): (X_m3_train, X_m3_test, list(feature_name_map["m3"])),
            ("m3", "mgwr"): build_anchor_matrix(
                X_m3_train, X_m3_test, stage5_smooth_train, stage5_smooth_test, feature_name_map["m3"]
            ),
        }

        for variant in variant_specs:
            rows_common = {
                "variant": variant.name,
                "family": variant.family,
                "baseline_name": variant.baseline_name,
                "parent_name": variant.parent_name,
                "model_kind": variant.model_kind,
                "split": split_name,
                "split_kind": spec.split_kind,
                "block_km": float(spec.block_km) if spec.block_km is not None else float("nan"),
                "rows": int(len(work)),
                "train_rows": int(tr.sum()),
                "test_rows": int(te.sum()),
                "residual_mode": variant.residual_mode,
            }

            if variant.model_kind == "rf":
                X_train, X_test, feature_names = matrix_cache[(variant.base_key, variant.smooth_mode)]
                model = m2_base_model if variant.name == "m2_baseline" else None
                if model is None:
                    model = make_rf(
                        n_trees=args.base_trees,
                        random_state=args.random_state,
                        n_jobs=args.n_jobs,
                        min_samples_leaf=1,
                        max_depth=None,
                        max_features=1.0,
                    )
                    model.fit(X_train, y_train)
                y_pred = (
                    m2_base_test_pred
                    if variant.name == "m2_baseline"
                    else model.predict(X_test).astype(np.float32)
                )
                residuals = (y_test - y_pred).astype(np.float32)
                metrics = {
                    **rows_common,
                    "n_features": int(X_train.shape[1]),
                    "r2": float(r2_score(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    "moran_i": float(moran_i(residuals, coords_test, k=MORAN_K)),
                }
                metric_rows.append(metrics)
                print(
                    f"{variant.name:24s} R2={metrics['r2']:.4f} "
                    f"RMSE={metrics['rmse']:.4f} MoranI={metrics['moran_i']:.4f} "
                    f"p={metrics['n_features']}",
                    flush=True,
                )

                if spec.split_kind == "block":
                    for feature, importance in sorted(
                        zip(feature_names, model.feature_importances_),
                        key=lambda pair: pair[1],
                        reverse=True,
                    ):
                        importance_rows.append(
                            {
                                "variant": variant.name,
                                "split": split_name,
                                "source": "model",
                                "feature": feature,
                                "importance": float(importance),
                                "group": feature_group(feature),
                            }
                        )

                if args.save_predictions:
                    pred_df = pd.DataFrame(
                        {
                            "x": coords_test[:, 0],
                            "y": coords_test[:, 1],
                            "observed": y_test,
                            "predicted": y_pred,
                            "residual": residuals,
                        }
                    )
                    pred_df.to_csv(
                        args.output_dir / f"{variant.name}_{split_name}_predictions.csv",
                        index=False,
                    )
                continue

            resid_train, resid_test, resid_feature_names = resid_bundle[variant.residual_mode]
            resid_model = make_rf(
                n_trees=args.resid_trees,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                min_samples_leaf=args.resid_min_leaf,
                max_depth=args.resid_max_depth,
                max_features=args.resid_max_features,
            )
            resid_model.fit(resid_train, m2_resid_target)
            resid_correction = resid_model.predict(resid_test).astype(np.float32)
            y_pred = (m2_base_test_pred + resid_correction).astype(np.float32)
            residuals = (y_test - y_pred).astype(np.float32)

            metrics = {
                **rows_common,
                "n_features": int(resid_train.shape[1]),
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "moran_i": float(moran_i(residuals, coords_test, k=MORAN_K)),
            }
            metric_rows.append(metrics)
            print(
                f"{variant.name:24s} R2={metrics['r2']:.4f} "
                f"RMSE={metrics['rmse']:.4f} MoranI={metrics['moran_i']:.4f} "
                f"p={metrics['n_features']}",
                flush=True,
            )

            if spec.split_kind == "block":
                for feature, importance in sorted(
                    zip(resid_feature_names, resid_model.feature_importances_),
                    key=lambda pair: pair[1],
                    reverse=True,
                ):
                    importance_rows.append(
                        {
                            "variant": variant.name,
                            "split": split_name,
                            "source": "residual_model",
                            "feature": feature,
                            "importance": float(importance),
                            "group": feature_group(feature),
                        }
                    )

            if args.save_predictions:
                pred_df = pd.DataFrame(
                    {
                        "x": coords_test[:, 0],
                        "y": coords_test[:, 1],
                        "observed": y_test,
                        "base_pred_m2": m2_base_test_pred,
                        "residual_correction": resid_correction,
                        "predicted": y_pred,
                        "residual": residuals,
                    }
                )
                pred_df.to_csv(
                    args.output_dir / f"{variant.name}_{split_name}_predictions.csv",
                    index=False,
                )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.output_dir / "metrics_long.csv", index=False)

    summary_df, wide_df = summarize_metrics(metrics_df, variant_specs)
    summary_df.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    wide_df.to_csv(args.output_dir / "metrics_wide.csv", index=False)

    if importance_rows:
        importance_df = pd.DataFrame(importance_rows)
        importance_df.to_csv(args.output_dir / "block_feature_importance.csv", index=False)
        group_df = (
            importance_df.groupby(["variant", "split", "source", "group"], as_index=False)["importance"]
            .sum()
            .sort_values(["split", "variant", "source", "importance"], ascending=[True, True, True, False])
        )
        group_df.to_csv(args.output_dir / "block_group_importance.csv", index=False)

    block100_df = summary_df[summary_df["split"] == "block_100km"].sort_values(
        ["r2", "moran_i"], ascending=[False, True]
    )
    block75_df = summary_df[summary_df["split"] == "block_75km"].sort_values(
        ["r2", "moran_i"], ascending=[False, True]
    )

    report_lines = [
        f"# Western US Residual Multiscale RF Round 2 ({TODAY})",
        "",
        f"- Input: `{args.input}`",
        f"- Response: `{args.response}`",
        f"- Rows used: `{len(work):,}`",
        f"- Primary and sensitivity block sizes: `{', '.join(str(int(x)) for x in args.block_km_list)} km`",
        f"- Base RF trees: `{args.base_trees}`",
        f"- Residual RF trees: `{args.resid_trees}`",
        f"- Residual RF min leaf: `{args.resid_min_leaf}`",
        f"- Residual RF max depth: `{args.resid_max_depth}`",
        f"- Residual RF max features: `{args.resid_max_features}`",
        "",
        "## Variants",
        "",
    ]
    for variant in variant_specs:
        report_lines.append(f"- `{variant.name}`: {variant.note}")
    report_lines.extend(
        [
            "",
            "## Block 100 km",
            "",
            "```text",
            block100_df.round(4).to_string(index=False),
            "```",
            "",
            "## Block 75 km",
            "",
            "```text",
            block75_df.round(4).to_string(index=False),
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
