#!/usr/bin/env python3
"""Fit the global OLS reference and compute KNN residual Moran diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import libpysal
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from scipy.spatial import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--predictors-file", required=True)
    parser.add_argument("--response", default="Resistance")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--global-permutations", type=int, default=999)
    parser.add_argument("--local-permutations", type=int, default=199)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def local_lag(values: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    tree = KDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    return values[indices[:, 1:]].mean(axis=1)


def classify(z: np.ndarray, lag_z: np.ndarray, p_values: np.ndarray) -> np.ndarray:
    labels = np.full(len(z), "NS", dtype=object)
    significant = p_values < 0.05
    labels[significant & (z > 0) & (lag_z > 0)] = "HH"
    labels[significant & (z < 0) & (lag_z < 0)] = "LL"
    labels[significant & (z > 0) & (lag_z < 0)] = "HL"
    labels[significant & (z < 0) & (lag_z > 0)] = "LH"
    return labels


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    predictors_path = Path(args.predictors_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictors = read_predictors(predictors_path)
    df = pd.read_parquet(input_path)
    identifier_columns = [column for column in ["pixel_id", "x", "y"] if column in df]
    required = [args.response, "x", "y", *predictors]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    work = df[list(dict.fromkeys([*identifier_columns, *required]))].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()

    x_matrix = work[predictors].to_numpy(dtype=float)
    observed = work[args.response].to_numpy(dtype=float)
    coords = work[["x", "y"]].to_numpy(dtype=float)
    model = LinearRegression().fit(x_matrix, observed)
    predicted = model.predict(x_matrix)
    residual = observed - predicted
    residual_z = (residual - residual.mean()) / residual.std(ddof=1)

    weights = libpysal.weights.KNN.from_array(coords, k=args.neighbors)
    weights.transform = "r"
    global_moran = Moran(residual, weights, permutations=args.global_permutations)
    local_moran = Moran_Local(
        residual,
        weights,
        permutations=args.local_permutations,
        seed=args.random_state,
    )
    lag_z = local_lag(residual_z, coords, args.neighbors)
    clusters = classify(residual_z, lag_z, local_moran.p_sim)

    residual_table = work[identifier_columns + [args.response]].copy()
    residual_table["ols_predicted"] = predicted
    residual_table["ols_residual"] = residual
    residual_table["ols_residual_z"] = residual_z
    residual_table["local_lag_residual_z"] = lag_z
    residual_table["local_moran_i"] = local_moran.Is
    residual_table["local_moran_p"] = local_moran.p_sim
    residual_table["local_cluster"] = clusters
    residual_table.to_parquet(output_dir / "ols_residuals.parquet", index=False)

    counts = (
        pd.Series(clusters, name="cluster")
        .value_counts()
        .reindex(["HH", "LL", "HL", "LH", "NS"], fill_value=0)
        .rename_axis("cluster")
        .reset_index(name="n")
    )
    counts["percent"] = counts["n"] / len(work) * 100.0
    counts.to_csv(output_dir / "local_moran_cluster_counts.csv", index=False)

    summary = {
        "input": Path(args.input).as_posix(),
        "response": args.response,
        "rows": int(len(work)),
        "predictors": predictors,
        "intercept": float(model.intercept_),
        "coefficients": {name: float(value) for name, value in zip(predictors, model.coef_)},
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "moran_i": float(global_moran.I),
        "moran_p_sim": float(global_moran.p_sim),
        "neighbors": int(args.neighbors),
        "global_permutations": int(args.global_permutations),
        "local_permutations": int(args.local_permutations),
        "random_state": int(args.random_state),
    }
    (output_dir / "ols_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
