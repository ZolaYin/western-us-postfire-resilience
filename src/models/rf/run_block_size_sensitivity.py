#!/usr/bin/env python3
"""Evaluate the final RF baseline under 50, 100, and 200 km spatial blocks."""
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

from run_foresttype_comparison import BASE_PREDS, prepare


DEFAULT_RESPONSES = ["Resistance", "IRI_good_pow2", "STAB_good_pow2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Analysis-ready model-table Parquet.")
    parser.add_argument(
        "--splits",
        default=None,
        help=(
            "Optional released split-assignment Parquet. When supplied, its random "
            "assignments and its saved 100 km assignments are reused exactly."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--responses", nargs="+", default=DEFAULT_RESPONSES)
    parser.add_argument("--block-km", nargs="+", type=float, default=[50.0, 100.0, 200.0])
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--sample-n", type=int, default=None, help="Smoke-test row limit.")
    return parser.parse_args()


def block_ids(work: pd.DataFrame, block_km: float) -> np.ndarray:
    size_m = block_km * 1000.0
    block_x = np.floor(work["x"].to_numpy(dtype=float) / size_m).astype(int)
    block_y = np.floor(work["y"].to_numpy(dtype=float) / size_m).astype(int)
    return np.asarray([f"{x}_{y}" for x, y in zip(block_x, block_y)], dtype=object)


def positions_from_labels(
    work: pd.DataFrame, assignments: pd.DataFrame, column: str
) -> tuple[np.ndarray, np.ndarray]:
    labels = assignments.set_index("pixel_id")
    if column not in labels:
        raise ValueError(f"Saved split table is missing column: {column}")
    joined = work[["pixel_id"]].join(labels[[column]], on="pixel_id")
    if joined[column].isna().any():
        raise ValueError(f"Eligible rows have missing saved labels in {column}")
    train = np.flatnonzero(joined[column].eq("train").to_numpy())
    test = np.flatnonzero(joined[column].eq("test").to_numpy())
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Empty train/test partition in {column}")
    return train, test


def generated_random_positions(
    n_rows: int, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(n_rows)
    return train_test_split(positions, test_size=test_size, random_state=random_state)


def generated_block_positions(
    groups: np.ndarray, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.arange(len(groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(splitter.split(positions, groups=groups))


def fit_and_score(
    work: pd.DataFrame,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    response: str,
    trees: int,
    random_state: int,
    n_jobs: int,
    label: str,
) -> tuple[float, float, float]:
    started = time.perf_counter()
    print(
        f"[RF] response={response} split={label} "
        f"train={len(train_pos):,} test={len(test_pos):,}",
        flush=True,
    )
    model = RandomForestRegressor(
        n_estimators=trees,
        random_state=random_state,
        n_jobs=n_jobs,
        max_features=1.0,
        max_depth=None,
        min_samples_leaf=1,
    )
    model.fit(work.iloc[train_pos][BASE_PREDS], work.iloc[train_pos][response])
    observed = work.iloc[test_pos][response].to_numpy(dtype=float)
    predicted = model.predict(work.iloc[test_pos][BASE_PREDS])
    r2 = float(r2_score(observed, predicted))
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    elapsed = time.perf_counter() - started
    print(
        f"[RF] response={response} split={label} done "
        f"r2={r2:.6f} rmse={rmse:.6f} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return r2, rmse, elapsed


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input).expanduser().resolve()
    print(f"[RF] Loading model table: {input_path}", flush=True)
    raw = pd.read_parquet(input_path)
    if args.sample_n is not None and args.sample_n < len(raw):
        raw = raw.sample(n=args.sample_n, random_state=args.random_state).sort_values("pixel_id")
    df, _, _ = prepare(raw)
    assignments = pd.read_parquet(args.splits) if args.splits else None

    rows: list[dict] = []
    run_start = time.perf_counter()
    for response in args.responses:
        # x and y already belong to BASE_PREDS; keep the model matrix strictly
        # one column per advertised predictor.
        required = list(dict.fromkeys(["pixel_id", response, *BASE_PREDS]))
        missing = [column for column in required if column not in df]
        if missing:
            raise ValueError(f"Missing columns for {response}: {missing}")
        work = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if not work.columns.is_unique:
            raise ValueError("RF work table contains duplicate column names")
        if work[BASE_PREDS].shape[1] != len(BASE_PREDS):
            raise ValueError("RF predictor matrix does not match the declared predictor list")

        random_positions = (
            positions_from_labels(work, assignments, f"random_{response}")
            if assignments is not None
            else generated_random_positions(len(work), args.test_size, args.random_state)
        )
        random_r2, random_rmse, random_seconds = fit_and_score(
            work,
            *random_positions,
            response,
            args.trees,
            args.random_state,
            args.n_jobs,
            "random",
        )

        for block_km in args.block_km:
            groups = block_ids(work, block_km)
            saved_column = f"block{int(block_km)}km_{response}"
            use_saved = assignments is not None and saved_column in assignments.columns
            block_positions = (
                positions_from_labels(work, assignments, saved_column)
                if use_saved
                else generated_block_positions(groups, args.test_size, args.random_state)
            )
            block_r2, block_rmse, block_seconds = fit_and_score(
                work,
                *block_positions,
                response,
                args.trees,
                args.random_state,
                args.n_jobs,
                f"block-{block_km:g}km",
            )
            block_train, block_test = block_positions
            random_train, random_test = random_positions
            rows.append(
                {
                    "response": response,
                    "block_km": block_km,
                    "occupied_blocks": int(pd.Series(groups).nunique()),
                    "eligible_rows": int(len(work)),
                    "n_predictors": int(len(BASE_PREDS)),
                    "random_train_rows": int(len(random_train)),
                    "random_test_rows": int(len(random_test)),
                    "block_train_rows": int(len(block_train)),
                    "block_test_rows": int(len(block_test)),
                    "block_train_groups": int(pd.Series(groups[block_train]).nunique()),
                    "block_test_groups": int(pd.Series(groups[block_test]).nunique()),
                    "random_r2": random_r2,
                    "random_rmse": random_rmse,
                    "block_r2": block_r2,
                    "block_rmse": block_rmse,
                    "r2_transferability_gap": random_r2 - block_r2,
                    "rmse_transferability_gap": block_rmse - random_rmse,
                    "random_fit_seconds": random_seconds,
                    "block_fit_seconds": block_seconds,
                    "block_assignment_source": "released" if use_saved else "generated",
                }
            )

    metrics = pd.DataFrame(rows).sort_values(["response", "block_km"])
    metrics.to_csv(output_dir / "rf_block_size_sensitivity.csv", index=False)
    metadata = {
        "input": Path(args.input).as_posix(),
        "saved_splits": Path(args.splits).as_posix() if args.splits else None,
        "responses": args.responses,
        "block_km": args.block_km,
        "predictors": BASE_PREDS,
        "trees": args.trees,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "max_features": 1.0,
        "max_depth": None,
        "min_samples_leaf": 1,
        "n_jobs": args.n_jobs,
        "sample_n": args.sample_n,
        "elapsed_minutes": (time.perf_counter() - run_start) / 60.0,
        "notes": [
            "The random split is fitted once per response and reused across block sizes.",
            "Released labels are used whenever the requested split column exists; other block sizes are deterministically generated in EPSG:5070.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"[RF] Completed {len(args.responses) * (1 + len(args.block_km))} fits in "
        f"{metadata['elapsed_minutes']:.1f} min; outputs: {output_dir}",
        flush=True,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
