#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from run_westernus_multiscale_rf_residual_round2 import (
    BASE_PREDS,
    BASE_TREES,
    CORE6_PREDS,
    INPUT,
    INNER_CV_FOLDS,
    LOCAL_RADIUS_KM,
    MAX_K,
    MORAN_K,
    RANDOM_STATE,
    RESID_MAX_DEPTH,
    RESID_MAX_FEATURES,
    RESID_MIN_LEAF,
    RESID_TREES,
    STAGE5B_PREDS,
    SplitSpec,
    TEST_SIZE,
    block_groups_from_coords,
    build_anchor_matrix,
    build_mgwr_smooth_bundle,
    build_split_masks,
    build_split_specs,
    crossfit_base_predictions,
    feature_group,
    make_rf,
    moran_i,
    prepare,
    smooth_train_test,
    summarize_metrics,
)


TODAY = date.today().strftime("%Y-%m-%d")
DEFAULT_BLOCK_KM_LIST = [100.0, 75.0]
TOP4_PREDS = [
    "CLIM_pr_sum_pre_z",
    "FS_TCC_t0_z",
    "TS_SOC_0_30cm_z",
    "FS_CBH_t0agg_z",
]


@dataclass(frozen=True)
class LocalzVariantSpec:
    name: str
    model_kind: str
    residual_mode: str
    preds: tuple[str, ...]
    radius_scale: float
    resid_min_leaf: int
    resid_max_depth: int | None
    resid_max_features: float | str | None
    note: str


VARIANT_SPECS = [
    LocalzVariantSpec(
        "m2_baseline",
        "rf",
        "none",
        tuple(CORE6_PREDS),
        1.0,
        RESID_MIN_LEAF,
        RESID_MAX_DEPTH,
        RESID_MAX_FEATURES,
        "Main-text M2 baseline: baseline predictors plus EVT group-class dummies.",
    ),
    LocalzVariantSpec(
        "m2_mgwr_anchor",
        "rf",
        "none",
        tuple(CORE6_PREDS),
        1.0,
        RESID_MIN_LEAF,
        RESID_MAX_DEPTH,
        RESID_MAX_FEATURES,
        "Anchor only: M2 baseline plus MGWR-matched smooth features for all 11 stage5b variables.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz6",
        "resid_rf",
        "localz",
        tuple(CORE6_PREDS),
        1.0,
        RESID_MIN_LEAF,
        RESID_MAX_DEPTH,
        RESID_MAX_FEATURES,
        "Round-2 winner: six local-z features at the MGWR-matched radii.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz6_r075",
        "resid_rf",
        "localz",
        tuple(CORE6_PREDS),
        0.75,
        RESID_MIN_LEAF,
        RESID_MAX_DEPTH,
        RESID_MAX_FEATURES,
        "Six local-z features with modestly tighter radii (0.75x) to emphasize more local departures.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz6_r125",
        "resid_rf",
        "localz",
        tuple(CORE6_PREDS),
        1.25,
        RESID_MIN_LEAF,
        RESID_MAX_DEPTH,
        RESID_MAX_FEATURES,
        "Six local-z features with modestly broader radii (1.25x) to test slightly more regional context.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz6_cons",
        "resid_rf",
        "localz",
        tuple(CORE6_PREDS),
        1.0,
        150,
        12,
        0.4,
        "Six local-z features with a more conservative residual RF to favor spatial transfer over local fit.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz6_flex",
        "resid_rf",
        "localz",
        tuple(CORE6_PREDS),
        1.0,
        75,
        16,
        0.6,
        "Six local-z features with a slightly more flexible residual RF to test whether light extra capacity helps.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz5_noroad",
        "resid_rf",
        "localz",
        tuple(pred for pred in CORE6_PREDS if pred != "HUM_roaddens_r5km_z"),
        1.0,
        RESID_MIN_LEAF,
        RESID_MAX_DEPTH,
        RESID_MAX_FEATURES,
        "Drop road-density local-z to test whether the weakest human-context term is mostly noise at 100 km transfer.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz5_noelev",
        "resid_rf",
        "localz",
        tuple(pred for pred in CORE6_PREDS if pred != "TS_elev_m_z"),
        1.0,
        RESID_MIN_LEAF,
        RESID_MAX_DEPTH,
        RESID_MAX_FEATURES,
        "Drop elevation local-z to test whether the broadest topo context term is diluting transfer gains.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz5_noelev_cons",
        "resid_rf",
        "localz",
        tuple(pred for pred in CORE6_PREDS if pred != "TS_elev_m_z"),
        1.0,
        150,
        12,
        0.4,
        "Drop elevation local-z and use a more conservative residual RF to target modest but stable 100-km gains.",
    ),
    LocalzVariantSpec(
        "m2_resid_localz4_top4",
        "resid_rf",
        "localz",
        tuple(TOP4_PREDS),
        1.0,
        RESID_MIN_LEAF,
        RESID_MAX_DEPTH,
        RESID_MAX_FEATURES,
        "Keep only the four strongest local-z dimensions from round 2: precipitation, TCC, SOC, and CBH.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Localz-focused MGWR-informed multiscale RF round 3 for Western US post-fire "
            "forest resistance. Concentrates on the strongest residual-localz variants from "
            "round 2 and tests a small set of radius, subset, and regularization refinements."
        )
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--response", default="Resistance")
    parser.add_argument("--base-trees", type=int, default=BASE_TREES)
    parser.add_argument("--resid-trees", type=int, default=RESID_TREES)
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
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[spec.name for spec in VARIANT_SPECS],
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def get_variant_specs(selected: list[str]) -> list[LocalzVariantSpec]:
    spec_map = {spec.name: spec for spec in VARIANT_SPECS}
    unknown = [name for name in selected if name not in spec_map]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    return [spec_map[name] for name in selected]


def round_radius_km(base_km: float, scale: float) -> int:
    scaled = base_km * scale
    rounded = int(25 * round(scaled / 25))
    return max(75, rounded)


def scaled_radius_map(preds: tuple[str, ...], scale: float) -> dict[str, int]:
    return {pred: round_radius_km(LOCAL_RADIUS_KM[pred], scale) for pred in preds}


def build_localz_bundle(
    train_coords: np.ndarray,
    train_X: np.ndarray,
    test_coords: np.ndarray,
    test_X: np.ndarray,
    preds: tuple[str, ...],
    radius_km_map: dict[str, int],
    max_k: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    localmean_train = np.zeros_like(train_X, dtype=np.float32)
    localmean_test = np.zeros((len(test_coords), train_X.shape[1]), dtype=np.float32)
    localsq_train = np.zeros_like(train_X, dtype=np.float32)
    localsq_test = np.zeros((len(test_coords), train_X.shape[1]), dtype=np.float32)

    grouped_indices: dict[int, list[int]] = {}
    for j, pred in enumerate(preds):
        grouped_indices.setdefault(radius_km_map[pred], []).append(j)

    for radius_km, js in sorted(grouped_indices.items()):
        s_train, s_test = smooth_train_test(
            train_coords=train_coords,
            train_X=train_X[:, js],
            test_coords=test_coords,
            radius_m=radius_km * 1000.0,
            max_k=max_k,
        )
        s2_train, s2_test = smooth_train_test(
            train_coords=train_coords,
            train_X=np.square(train_X[:, js]).astype(np.float32),
            test_coords=test_coords,
            radius_m=radius_km * 1000.0,
            max_k=max_k,
        )
        localmean_train[:, js] = s_train
        localmean_test[:, js] = s_test
        localsq_train[:, js] = s2_train
        localsq_test[:, js] = s2_test

    localvar_train = np.maximum(localsq_train - np.square(localmean_train), 1e-6).astype(np.float32)
    localvar_test = np.maximum(localsq_test - np.square(localmean_test), 1e-6).astype(np.float32)
    localsd_train = np.sqrt(localvar_train).astype(np.float32)
    localsd_test = np.sqrt(localvar_test).astype(np.float32)
    localz_train = ((train_X - localmean_train) / np.maximum(localsd_train, 1e-3)).astype(np.float32)
    localz_test = ((test_X - localmean_test) / np.maximum(localsd_test, 1e-3)).astype(np.float32)
    names = [f"{pred}_localz" for pred in preds]
    return localz_train, localz_test, names


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_specs = get_variant_specs(args.variants)
    split_specs = build_split_specs(args.block_km_list)

    print("Loading and preparing data ...", flush=True)
    raw = pd.read_parquet(args.input)
    df = prepare(raw)

    group_cols = sorted([c for c in df.columns if c.startswith("EVT_group_")])
    base_feature_map = {
        "stage5b": STAGE5B_PREDS,
        "m2": BASE_PREDS + group_cols,
        "core6": CORE6_PREDS,
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

    feature_name_map: dict[str, list[str]] = {}
    X_map: dict[str, np.ndarray] = {}
    for key, cols in base_feature_map.items():
        feature_name_map[key] = [c for c in cols if c in work.columns]
        X_map[key] = work[feature_name_map[key]].to_numpy(dtype=np.float32)

    metadata = {
        "date": TODAY,
        "input": str(args.input),
        "response": args.response,
        "rows_used": int(len(work)),
        "random_state": args.random_state,
        "base_trees": args.base_trees,
        "resid_trees": args.resid_trees,
        "max_k": args.max_k,
        "inner_cv_folds": args.inner_cv_folds,
        "sample_n": args.sample_n,
        "block_km_list": [spec.block_km for spec in split_specs if spec.block_km is not None],
        "core6_predictors": CORE6_PREDS,
        "variant_specs": [asdict(spec) for spec in variant_specs],
        "scaled_radii_km": {
            spec.name: scaled_radius_map(spec.preds, spec.radius_scale)
            for spec in variant_specs
            if spec.model_kind == "resid_rf"
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    metric_rows: list[dict] = []
    importance_rows: list[dict] = []
    block_rows: list[dict] = []

    core_idx_map = {pred: idx for idx, pred in enumerate(CORE6_PREDS)}

    for split_name, is_test in split_masks.items():
        spec = split_map[split_name]
        print(f"\n=== Split: {split_name} ===", flush=True)
        tr = ~is_test
        te = is_test

        coords_train = coords[tr]
        coords_test = coords[te]
        y_train = y[tr]
        y_test = y[te]
        test_block_ids = (
            block_groups_from_coords(coords_test, spec.block_km)
            if spec.split_kind == "block" and spec.block_km is not None
            else None
        )

        X_m2_train = X_map["m2"][tr]
        X_m2_test = X_map["m2"][te]
        X_core6_train = X_map["core6"][tr]
        X_core6_test = X_map["core6"][te]

        stage5_smooth_train, stage5_smooth_test = build_mgwr_smooth_bundle(
            train_coords=coords_train,
            stage5b_train=X_map["stage5b"][tr],
            test_coords=coords_test,
            max_k=args.max_k,
        )

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

        matrix_cache: dict[tuple[tuple[str, ...], float], tuple[np.ndarray, np.ndarray, list[str]]] = {}
        anchor_train, anchor_test, anchor_features = build_anchor_matrix(
            X_m2_train,
            X_m2_test,
            stage5_smooth_train,
            stage5_smooth_test,
            feature_name_map["m2"],
        )

        for variant in variant_specs:
            rows_common = {
                "variant": variant.name,
                "family": "m2",
                "baseline_name": "m2_baseline",
                "parent_name": "m2_baseline",
                "model_kind": variant.model_kind,
                "split": split_name,
                "split_kind": spec.split_kind,
                "block_km": float(spec.block_km) if spec.block_km is not None else float("nan"),
                "rows": int(len(work)),
                "train_rows": int(tr.sum()),
                "test_rows": int(te.sum()),
                "residual_mode": variant.residual_mode,
            }

            if variant.name == "m2_baseline":
                y_pred = m2_base_test_pred
                feature_names = feature_name_map["m2"]
                model = m2_base_model
            elif variant.name == "m2_mgwr_anchor":
                model = make_rf(
                    n_trees=args.base_trees,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    min_samples_leaf=1,
                    max_depth=None,
                    max_features=1.0,
                )
                model.fit(anchor_train, y_train)
                y_pred = model.predict(anchor_test).astype(np.float32)
                feature_names = anchor_features
            else:
                cache_key = (variant.preds, variant.radius_scale)
                if cache_key not in matrix_cache:
                    pred_indices = [core_idx_map[pred] for pred in variant.preds]
                    localz_train, localz_test, localz_names = build_localz_bundle(
                        train_coords=coords_train,
                        train_X=X_core6_train[:, pred_indices],
                        test_coords=coords_test,
                        test_X=X_core6_test[:, pred_indices],
                        preds=variant.preds,
                        radius_km_map=scaled_radius_map(variant.preds, variant.radius_scale),
                        max_k=args.max_k,
                    )
                    matrix_cache[cache_key] = (localz_train, localz_test, localz_names)
                resid_train, resid_test, feature_names = matrix_cache[cache_key]

                model = make_rf(
                    n_trees=args.resid_trees,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    min_samples_leaf=variant.resid_min_leaf,
                    max_depth=variant.resid_max_depth,
                    max_features=variant.resid_max_features,
                )
                model.fit(resid_train, m2_resid_target)
                resid_correction = model.predict(resid_test).astype(np.float32)
                y_pred = (m2_base_test_pred + resid_correction).astype(np.float32)

            residuals = (y_test - y_pred).astype(np.float32)
            metrics = {
                **rows_common,
                "n_features": int(len(feature_names)),
                "r2": float(r2_score(y_test, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "moran_i": float(moran_i(residuals, coords_test, k=MORAN_K)),
            }
            metric_rows.append(metrics)
            print(
                f"{variant.name:26s} R2={metrics['r2']:.4f} "
                f"RMSE={metrics['rmse']:.4f} MoranI={metrics['moran_i']:.4f} "
                f"p={metrics['n_features']}",
                flush=True,
            )

            if spec.split_kind == "block":
                baseline_rmse_by_block: dict[str, float] = {}
                baseline_mae_by_block: dict[str, float] = {}
                baseline_r2_by_block: dict[str, float] = {}
                baseline_bias_by_block: dict[str, float] = {}
                if test_block_ids is not None:
                    for block_id in np.unique(test_block_ids):
                        mask = test_block_ids == block_id
                        obs_block = y_test[mask]
                        base_block_pred = m2_base_test_pred[mask]
                        base_resid = obs_block - base_block_pred
                        baseline_rmse_by_block[block_id] = float(
                            np.sqrt(mean_squared_error(obs_block, base_block_pred))
                        )
                        baseline_mae_by_block[block_id] = float(np.mean(np.abs(base_resid)))
                        baseline_bias_by_block[block_id] = float(np.mean(base_resid))
                        baseline_r2_by_block[block_id] = (
                            float(r2_score(obs_block, base_block_pred))
                            if len(obs_block) >= 2
                            else float("nan")
                        )
                    for block_id in np.unique(test_block_ids):
                        mask = test_block_ids == block_id
                        obs_block = y_test[mask]
                        pred_block = y_pred[mask]
                        resid_block = obs_block - pred_block
                        block_rows.append(
                            {
                                "variant": variant.name,
                                "split": split_name,
                                "block_km": float(spec.block_km),
                                "block_id": block_id,
                                "n": int(mask.sum()),
                                "mean_observed": float(np.mean(obs_block)),
                                "mean_predicted": float(np.mean(pred_block)),
                                "rmse": float(np.sqrt(mean_squared_error(obs_block, pred_block))),
                                "mae": float(np.mean(np.abs(resid_block))),
                                "bias": float(np.mean(resid_block)),
                                "abs_bias": float(np.abs(np.mean(resid_block))),
                                "r2": float(r2_score(obs_block, pred_block))
                                if len(obs_block) >= 2
                                else float("nan"),
                                "baseline_rmse": baseline_rmse_by_block[block_id],
                                "baseline_mae": baseline_mae_by_block[block_id],
                                "baseline_bias": baseline_bias_by_block[block_id],
                                "baseline_abs_bias": float(np.abs(baseline_bias_by_block[block_id])),
                                "baseline_r2": baseline_r2_by_block[block_id],
                                "delta_rmse_vs_m2": float(
                                    np.sqrt(mean_squared_error(obs_block, pred_block))
                                    - baseline_rmse_by_block[block_id]
                                ),
                                "delta_mae_vs_m2": float(
                                    np.mean(np.abs(resid_block)) - baseline_mae_by_block[block_id]
                                ),
                                "delta_bias_vs_m2": float(
                                    np.mean(resid_block) - baseline_bias_by_block[block_id]
                                ),
                                "delta_abs_bias_vs_m2": float(
                                    np.abs(np.mean(resid_block))
                                    - float(np.abs(baseline_bias_by_block[block_id]))
                                ),
                                "delta_r2_vs_m2": float(
                                    (
                                        float(r2_score(obs_block, pred_block))
                                        if len(obs_block) >= 2
                                        else float("nan")
                                    )
                                    - baseline_r2_by_block[block_id]
                                ),
                            }
                        )
                for feature, importance in sorted(
                    zip(feature_names, model.feature_importances_),
                    key=lambda pair: pair[1],
                    reverse=True,
                ):
                    importance_rows.append(
                        {
                            "variant": variant.name,
                            "split": split_name,
                            "source": "model" if variant.model_kind == "rf" else "residual_model",
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

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.output_dir / "metrics_long.csv", index=False)

    summary_df, wide_df = summarize_metrics(metrics_df, [])
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

    if block_rows:
        block_df = pd.DataFrame(block_rows)
        block_df.to_csv(args.output_dir / "block_unit_diagnostics.csv", index=False)
        block_summary_df = (
            block_df.groupby(["variant", "split", "block_km"], as_index=False)
            .agg(
                n_blocks=("block_id", "nunique"),
                mean_block_r2=("r2", "mean"),
                median_block_r2=("r2", "median"),
                mean_block_rmse=("rmse", "mean"),
                median_delta_rmse_vs_m2=("delta_rmse_vs_m2", "median"),
                mean_delta_rmse_vs_m2=("delta_rmse_vs_m2", "mean"),
                share_blocks_better_rmse=("delta_rmse_vs_m2", lambda s: float(np.mean(s < 0))),
                share_blocks_better_r2=("delta_r2_vs_m2", lambda s: float(np.mean(s > 0))),
                share_blocks_lower_abs_bias=("delta_abs_bias_vs_m2", lambda s: float(np.mean(s < 0))),
            )
            .sort_values(["split", "mean_block_r2"], ascending=[True, False])
        )
        block_summary_df.to_csv(args.output_dir / "block_diagnostic_summary.csv", index=False)

    block100_df = summary_df[summary_df["split"] == "block_100km"].sort_values(
        ["r2", "moran_i"], ascending=[False, True]
    )
    block75_df = summary_df[summary_df["split"] == "block_75km"].sort_values(
        ["r2", "moran_i"], ascending=[False, True]
    )

    report_lines = [
        f"# Western US Localz-Focused Multiscale RF Round 3 ({TODAY})",
        "",
        f"- Input: `{args.input}`",
        f"- Response: `{args.response}`",
        f"- Rows used: `{len(work):,}`",
        f"- Block sizes: `{', '.join(str(int(x)) for x in args.block_km_list)} km`",
        f"- Base RF trees: `{args.base_trees}`",
        f"- Residual RF trees: `{args.resid_trees}`",
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
