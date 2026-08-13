#!/usr/bin/env python3
"""Compare RF forest-type encodings under random and spatial-block validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


DEFAULT_RESPONSES = ["Resistance", "T50", "T80", "IRI_good_pow2", "STAB_good_pow2"]
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", default=None, help="Optional saved assignment Parquet.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--responses", nargs="+", default=DEFAULT_RESPONSES)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--block-km", type=float, default=100.0)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum RF tree depth; default None reproduces the primary unconstrained RF.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per RF leaf; default 1 reproduces the primary RF.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--sample-n", type=int, default=None, help="Smoke-test row limit.")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    return ((values - values.mean()) / std).astype(np.float32)


def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    out = raw.copy()
    for z_column, raw_column in BASE_TO_Z.items():
        if z_column not in out and raw_column in out:
            out[z_column] = zscore(out[raw_column])
    if "HUM_popdens_win10km_log_z" not in out:
        out["HUM_popdens_win10km_log_z"] = zscore(
            np.log1p(pd.to_numeric(out["HUM_popdens_win10km"], errors="coerce").clip(lower=0))
        )
    if "HUM_viirs_near_t0_log_z" not in out:
        out["HUM_viirs_near_t0_log_z"] = zscore(
            np.log1p(pd.to_numeric(out["HUM_viirs_near_t0"], errors="coerce").clip(lower=0))
        )
    out["x_sq_z"] = zscore(pd.to_numeric(out["x"], errors="coerce") ** 2)
    out["y_sq_z"] = zscore(pd.to_numeric(out["y"], errors="coerce") ** 2)
    out["xy_z"] = zscore(
        pd.to_numeric(out["x"], errors="coerce")
        * pd.to_numeric(out["y"], errors="coerce")
    )

    group = out["FS_EVT_group_class"].astype("string").fillna("unknown").astype(str)
    group_dummies = pd.get_dummies(group, prefix="EVT_group", dtype=np.float32)
    code = out["FS_EVT2022_code"].astype("Int64").astype(str).fillna("missing")
    code_dummies = pd.get_dummies(code, prefix="EVT_code", dtype=np.float32)
    out = pd.concat([out, group_dummies, code_dummies], axis=1)
    return out, sorted(group_dummies.columns), sorted(code_dummies.columns)


def generated_splits(
    work: pd.DataFrame, test_size: float, random_state: int, block_km: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    positions = np.arange(len(work))
    random_train, random_test = train_test_split(
        positions, test_size=test_size, random_state=random_state
    )
    size_m = block_km * 1000.0
    block_x = np.floor(work["x"].to_numpy(dtype=float) / size_m).astype(int)
    block_y = np.floor(work["y"].to_numpy(dtype=float) / size_m).astype(int)
    groups = np.asarray([f"{x}_{y}" for x, y in zip(block_x, block_y)], dtype=object)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    block_train, block_test = next(splitter.split(positions, groups=groups))
    return {"random": (random_train, random_test), "block": (block_train, block_test)}


def saved_split_positions(
    work: pd.DataFrame, assignments: pd.DataFrame, response: str, block_km: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    labels = assignments.set_index("pixel_id")
    random_col = f"random_{response}"
    block_col = f"block{int(block_km)}km_{response}"
    missing = [column for column in [random_col, block_col] if column not in labels]
    if missing:
        raise ValueError(f"Saved split table is missing columns: {missing}")
    joined = work[["pixel_id"]].join(labels[[random_col, block_col]], on="pixel_id")
    if joined[[random_col, block_col]].isna().any().any():
        raise ValueError(f"Eligible {response} rows have missing saved split labels")
    result = {}
    for name, column in [("random", random_col), ("block", block_col)]:
        train = np.flatnonzero(joined[column].eq("train").to_numpy())
        test = np.flatnonzero(joined[column].eq("test").to_numpy())
        if len(train) == 0 or len(test) == 0:
            raise ValueError(f"Empty {name} train/test partition for {response}")
        result[name] = (train, test)
    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).expanduser().resolve()
    print(f"[RF] Loading model table: {input_path}", flush=True)
    raw = pd.read_parquet(input_path)
    if args.sample_n is not None and args.sample_n < len(raw):
        raw = raw.sample(n=args.sample_n, random_state=args.random_state).sort_values("pixel_id")
    df, group_columns, code_columns = prepare(raw)
    assignments = pd.read_parquet(args.splits) if args.splits else None

    variants = {
        "M1_baseline_only": BASE_PREDS,
        "M2_evt_group_class": BASE_PREDS + group_columns,
        "M3_evt_raw_code": BASE_PREDS + code_columns,
    }
    metrics_rows: list[dict] = []
    importance_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    completed_models = 0
    expected_models = len(args.responses) * len(variants) * 2
    run_start = time.perf_counter()

    for response in args.responses:
        for variant, predictors in variants.items():
            # x and y already belong to BASE_PREDS. De-duplicate the selection so
            # the fitted matrix contains the advertised number of predictors.
            required = list(dict.fromkeys(["pixel_id", response, *predictors]))
            missing = [column for column in required if column not in df]
            if missing:
                raise ValueError(f"Missing columns for {response}/{variant}: {missing}")
            work = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
            if not work.columns.is_unique:
                raise ValueError("RF work table contains duplicate column names")
            if work[predictors].shape[1] != len(predictors):
                raise ValueError("RF predictor matrix does not match the declared predictor list")
            splits = (
                saved_split_positions(work, assignments, response, args.block_km)
                if assignments is not None
                else generated_splits(work, args.test_size, args.random_state, args.block_km)
            )

            for split_name, (train_pos, test_pos) in splits.items():
                train = work.iloc[train_pos]
                test = work.iloc[test_pos]
                completed_models += 1
                fit_start = time.perf_counter()
                print(
                    f"[RF {completed_models}/{expected_models}] "
                    f"response={response} variant={variant} split={split_name} "
                    f"train={len(train):,} test={len(test):,} predictors={len(predictors)}",
                    flush=True,
                )
                model = RandomForestRegressor(
                    n_estimators=args.trees,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    max_features=1.0,
                    max_depth=args.max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                )
                model.fit(train[predictors], train[response])
                predicted = model.predict(test[predictors])
                observed = test[response].to_numpy(dtype=float)
                r2 = float(r2_score(observed, predicted))
                rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
                metrics_rows.append(
                    {
                        "response": response,
                        "variant": variant,
                        "split": split_name,
                        "eligible_rows": int(len(work)),
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "n_predictors": int(len(predictors)),
                        "r2": r2,
                        "rmse": rmse,
                    }
                )
                print(
                    f"[RF {completed_models}/{expected_models}] done "
                    f"r2={r2:.6f} rmse={rmse:.6f} "
                    f"elapsed={time.perf_counter() - fit_start:.1f}s",
                    flush=True,
                )
                for feature, importance in zip(predictors, model.feature_importances_):
                    importance_rows.append(
                        {
                            "response": response,
                            "variant": variant,
                            "split": split_name,
                            "feature": feature,
                            "importance": float(importance),
                        }
                    )
                if args.save_predictions:
                    part = test[["pixel_id", "x", "y"]].copy()
                    part["response"] = response
                    part["variant"] = variant
                    part["split"] = split_name
                    part["observed"] = observed
                    part["predicted"] = predicted
                    prediction_rows.append(part)

    metrics = pd.DataFrame(metrics_rows).sort_values(["response", "split", "variant"])
    metrics.to_csv(output_dir / "rf_foresttype_metrics.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(output_dir / "rf_feature_importance.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_parquet(
            output_dir / "rf_predictions.parquet", index=False
        )
    metadata = {
        "input": Path(args.input).as_posix(),
        "saved_splits": Path(args.splits).as_posix() if args.splits else None,
        "responses": args.responses,
        "trees": args.trees,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "block_km": args.block_km,
        "max_features": 1.0,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "n_jobs": args.n_jobs,
        "sample_n": args.sample_n,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[RF] Completed {completed_models} fits in "
        f"{(time.perf_counter() - run_start) / 60:.1f} min; outputs: {output_dir}",
        flush=True,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
