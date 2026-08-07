#!/usr/bin/env python3
"""
Post-hoc block CV for MGWR results.

MGWR was fit on all data — we can't redo it, but we can approximate block CV:
  1. Split into train/test blocks (100 km grid, 20% test)
  2. For each test point, IDW-interpolate coefficients from nearby TRAINING points
  3. Predict: Ŷ_i = Σ_j β̂_j(interp) × X_ij
  4. Compute R²(test)

Also computes random CV the same way.

Usage (single result dir):
  python compute_mgwr_block_cv.py \
    --coef-dir  results/Resistance_stage5b_seed42 \
    --input     samples/sample_n12000_seed42.parquet \
    --output    results/Resistance_stage5b_seed42/block_cv.csv

Usage (batch — all result dirs):
  python compute_mgwr_block_cv.py \
    --results-root results/ \
    --samples-dir  samples/ \
    --output       mgwr_block_cv_all.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# ── Helpers ───────────────────────────────────────────────────────────────────

def idw_interpolate(
    train_coords: np.ndarray,  # (n_train, 2)
    train_coefs: np.ndarray,   # (n_train, n_coef)
    test_coords: np.ndarray,   # (n_test,  2)
    k: int = 10,
    power: float = 2.0,
) -> np.ndarray:               # (n_test,  n_coef)
    tree = KDTree(train_coords)
    dists, idxs = tree.query(test_coords, k=min(k, len(train_coords)))
    dists = np.maximum(dists, 1e-10)          # avoid div-by-zero
    weights = 1.0 / dists ** power            # (n_test, k)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ij,ijk->ik", weights, train_coefs[idxs])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ── Core CV routine ───────────────────────────────────────────────────────────

def compute_cv(
    coef_dir: Path,
    input_path: Path,
    knn: int = 10,
    block_size: float = 100_000.0,
    random_state: int = 42,
) -> dict:
    metrics_file = coef_dir / "mgwr_metrics.json"
    coef_file    = coef_dir / "mgwr_coefficients.parquet"

    if not metrics_file.exists() or not coef_file.exists():
        return {"coef_dir": str(coef_dir), "status": "missing"}

    metrics     = json.loads(metrics_file.read_text())
    response    = metrics["response"]
    predictors  = metrics["predictors"]
    coef_cols   = ["Intercept"] + predictors

    coef_df  = pd.read_parquet(coef_file)
    sample   = pd.read_parquet(input_path)

    needed = [response, "x", "y"] + predictors
    data   = (
        sample[needed]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    # Inner join on (x, y) to align coefficients with data rows
    merged = data.merge(
        coef_df[["x", "y"] + coef_cols], on=["x", "y"], how="inner"
    )
    if len(merged) == 0:
        return {"coef_dir": str(coef_dir), "status": "no_overlap"}

    coords  = merged[["x", "y"]].to_numpy(dtype=float)
    Y       = merged[response].to_numpy(dtype=float)
    X_mat   = np.column_stack([np.ones(len(merged)),
                               merged[predictors].to_numpy(dtype=float)])
    B_mat   = merged[coef_cols].to_numpy(dtype=float)

    # In-sample (fitted values from mgwr itself)
    y_insample = (X_mat * B_mat).sum(axis=1)
    r2_in      = r2_score(Y, y_insample)

    # Random CV ----------------------------------------------------------------
    rng          = np.random.default_rng(random_state)
    is_test_rnd  = rng.random(len(merged)) < 0.2
    train_rnd    = ~is_test_rnd
    interp_rnd   = idw_interpolate(
        coords[train_rnd], B_mat[train_rnd], coords[is_test_rnd], k=knn
    )
    y_rnd  = (X_mat[is_test_rnd] * interp_rnd).sum(axis=1)
    r2_rnd = r2_score(Y[is_test_rnd], y_rnd)

    # Block CV -----------------------------------------------------------------
    bx = np.floor(coords[:, 0] / block_size).astype(int).astype(str)
    by = np.floor(coords[:, 1] / block_size).astype(int).astype(str)
    block_ids     = np.char.add(np.char.add(bx, "_"), by)
    unique_blocks = np.unique(block_ids)
    n_test_blk    = max(1, round(len(unique_blocks) * 0.2))
    test_blocks   = set(
        np.random.default_rng(random_state).choice(
            unique_blocks, size=n_test_blk, replace=False
        )
    )
    is_test_blk = np.array([b in test_blocks for b in block_ids])
    train_blk   = ~is_test_blk
    interp_blk  = idw_interpolate(
        coords[train_blk], B_mat[train_blk], coords[is_test_blk], k=knn
    )
    y_blk  = (X_mat[is_test_blk] * interp_blk).sum(axis=1)
    r2_blk = r2_score(Y[is_test_blk], y_blk)

    print(
        f"  {coef_dir.name:45s}  "
        f"in={r2_in:.3f}  rndCV={r2_rnd:.3f}  blkCV={r2_blk:.3f}"
    )

    return {
        "coef_dir":     str(coef_dir),
        "response":     response,
        "stage":        metrics.get("predictors_file", ""),
        "n":            int(len(merged)),
        "knn":          knn,
        "r2_insample":  r2_in,
        "r2_randCV":    r2_rnd,
        "r2_blkCV":     r2_blk,
        "aic":          metrics.get("aic"),
        "adj_r2_mgwr":  metrics.get("adj_r2"),
        "status":       "ok",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Single-dir mode
    p.add_argument("--coef-dir",  default=None,
                   help="Single MGWR result directory")
    p.add_argument("--input",     default=None,
                   help="Sample parquet used for that run")
    p.add_argument("--output",    default=None,
                   help="Output CSV path")
    # Batch mode
    p.add_argument("--results-root", default=None,
                   help="Root dir containing multiple result dirs")
    p.add_argument("--samples-dir",  default=None,
                   help="Dir containing sample_n*_seed*.parquet files")
    p.add_argument("--batch-output", default="mgwr_block_cv_all.csv",
                   help="Output CSV for batch mode")
    # Shared
    p.add_argument("--knn",          type=int,   default=10)
    p.add_argument("--block-size",   type=float, default=100_000.0)
    p.add_argument("--random-state", type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.coef_dir:
        # ── Single mode ───────────────────────────────────────────────────────
        result = compute_cv(
            coef_dir=Path(args.coef_dir),
            input_path=Path(args.input),
            knn=args.knn,
            block_size=args.block_size,
            random_state=args.random_state,
        )
        out = args.output or str(Path(args.coef_dir) / "block_cv.csv")
        pd.DataFrame([result]).to_csv(out, index=False)
        print(f"Saved → {out}")

    elif args.results_root:
        # ── Batch mode ────────────────────────────────────────────────────────
        results_root = Path(args.results_root)
        samples_dir  = Path(args.samples_dir)
        rows = []
        for coef_dir in sorted(results_root.iterdir()):
            if not coef_dir.is_dir():
                continue
            # Infer seed from dir name, e.g. Resistance_stage5b_seed42
            parts = coef_dir.name.split("_seed")
            if len(parts) != 2:
                continue
            seed = parts[1]
            sample_path = samples_dir / f"sample_n12000_seed{seed}.parquet"
            if not sample_path.exists():
                print(f"  [skip] sample not found: {sample_path.name}")
                continue
            row = compute_cv(
                coef_dir=coef_dir,
                input_path=sample_path,
                knn=args.knn,
                block_size=args.block_size,
                random_state=args.random_state,
            )
            rows.append(row)

        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.batch_output, index=False)
        print(f"\nBatch done. {len(rows)} dirs → {args.batch_output}")
        if len(out_df):
            ok = out_df[out_df.status == "ok"]
            print(ok[["coef_dir", "r2_insample", "r2_randCV", "r2_blkCV"]].to_string(index=False))
    else:
        print("Error: provide --coef-dir or --results-root")


if __name__ == "__main__":
    main()
