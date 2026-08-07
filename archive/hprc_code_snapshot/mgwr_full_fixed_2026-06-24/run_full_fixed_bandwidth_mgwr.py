#!/usr/bin/env python3
"""Full-point fixed-bandwidth MGWR-style backfitting.

This script is designed for the Western US post-fire resilience project.
It applies the multiscale bandwidth hierarchy estimated from the 12k MGWR
sample to all available pixels, scaling adaptive neighbor counts to the
full data size. It avoids a full MGWR bandwidth search, which is the part
that previously exceeded wall time on Grace.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors


DEFAULT_ALIAS = {
    "TS_SOC_0_30cm_clean_z": "TS_SOC_0_30cm_z",
}


def read_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def parse_aliases(values: list[str]) -> dict[str, str]:
    aliases = dict(DEFAULT_ALIAS)
    for item in values:
        if "=" not in item:
            raise ValueError(f"Alias must be old=new, got {item!r}")
        old, new = item.split("=", 1)
        aliases[old.strip()] = new.strip()
    return aliases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input parquet table.")
    parser.add_argument("--response", default="Resistance", help="Response column.")
    parser.add_argument("--predictors-file", required=True, help="One predictor per line.")
    parser.add_argument("--bandwidth-file", required=True, help="12k MGWR bandwidth CSV.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--reference-n", type=int, default=12000, help="Sample size used for the bandwidth file.")
    parser.add_argument("--scale-bandwidths", action="store_true", help="Scale adaptive bandwidths by n_full / reference_n.")
    parser.add_argument("--min-bw", type=int, default=40, help="Minimum adaptive neighbor count.")
    parser.add_argument("--max-iter", type=int, default=8, help="Maximum fixed-bandwidth backfitting iterations.")
    parser.add_argument("--tol", type=float, default=1e-5, help="Backfitting convergence tolerance.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Rows per nearest-neighbor chunk.")
    parser.add_argument(
        "--global-threshold-frac",
        type=float,
        default=0.90,
        help="Treat bandwidths >= this fraction of n as global smooth terms.",
    )
    parser.add_argument("--ridge", type=float, default=1e-10, help="Small denominator stabilizer.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if present.")
    parser.add_argument("--sample-n", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for smoke-test sampling.")
    parser.add_argument("--alias", action="append", default=[], help="Column alias old=new; may be repeated.")
    return parser.parse_args()


def resolve_predictors(df: pd.DataFrame, predictor_terms: list[str], aliases: dict[str, str]) -> tuple[list[str], list[str]]:
    input_cols = []
    output_terms = []
    for term in predictor_terms:
        col = term if term in df.columns else aliases.get(term, term)
        if col not in df.columns:
            raise ValueError(f"Predictor {term!r} resolved to {col!r}, but that column is absent.")
        input_cols.append(col)
        output_terms.append(term)
    return input_cols, output_terms


def load_and_prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(args.input)
    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.random_state).reset_index(drop=True)

    predictor_terms = read_lines(Path(args.predictors_file))
    aliases = parse_aliases(args.alias)
    input_cols, output_terms = resolve_predictors(df, predictor_terms, aliases)
    required = [args.response, "x", "y", *input_cols]
    work = (
        df[required]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    if work.empty:
        raise ValueError("No complete rows remain after filtering.")

    y = work[args.response].to_numpy(dtype=np.float64)
    coords = work[["x", "y"]].to_numpy(dtype=np.float64)
    X_pred = work[input_cols].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(work), dtype=np.float64), X_pred])
    terms = ["Intercept", *output_terms]
    return work, y, coords, X, terms


def load_bandwidths(
    path: Path,
    terms: list[str],
    n: int,
    reference_n: int,
    scale: bool,
    min_bw: int,
    aliases: dict[str, str],
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if not {"term", "bandwidth"}.issubset(raw.columns):
        raise ValueError("Bandwidth file must contain term and bandwidth columns.")
    bw_map = dict(zip(raw["term"].astype(str), pd.to_numeric(raw["bandwidth"], errors="raise")))
    inverse_alias = {v: k for k, v in aliases.items()}
    term_to_bw_term = {}
    missing = []
    for term in terms:
        if term in bw_map:
            term_to_bw_term[term] = term
        elif term in inverse_alias and inverse_alias[term] in bw_map:
            term_to_bw_term[term] = inverse_alias[term]
        else:
            missing.append(term)
    if missing:
        raise ValueError(f"Bandwidth file is missing terms: {missing}")
    ratio = n / float(reference_n)
    rows = []
    for term in terms:
        bw_term = term_to_bw_term[term]
        source_bw = float(bw_map[bw_term])
        bw = int(round(source_bw * ratio)) if scale else int(round(source_bw))
        bw = max(min_bw, min(n - 1, bw))
        rows.append(
            {
                "term": term,
                "bandwidth_term": bw_term,
                "source_bandwidth": source_bw,
                "scaled_bandwidth": bw,
                "scale_ratio": ratio if scale else 1.0,
            }
        )
    return pd.DataFrame(rows)


def initial_params_global(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.tile(beta.reshape(1, -1), (X.shape[0], 1))


def adaptive_bisquare_weights(distances: np.ndarray) -> np.ndarray:
    dmax = distances[:, -1].reshape(-1, 1)
    dmax = np.where(dmax <= 0, np.finfo(np.float64).eps, dmax)
    ratio = distances / dmax
    weights = np.square(1.0 - np.square(ratio))
    weights[ratio >= 1.0] = 0.0
    return weights


def global_univariate_beta(x: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    denom = float(np.dot(x, x) + ridge)
    beta = float(np.dot(x, target) / denom)
    return np.full_like(target, beta, dtype=np.float64)


def local_univariate_beta(
    nn: NearestNeighbors,
    coords: np.ndarray,
    x: np.ndarray,
    target: np.ndarray,
    bw: int,
    chunk_size: int,
    ridge: float,
) -> np.ndarray:
    n = len(target)
    beta = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        distances, indices = nn.kneighbors(coords[start:end], n_neighbors=bw, return_distance=True)
        weights = adaptive_bisquare_weights(distances)
        x_neighbors = x[indices]
        y_neighbors = target[indices]
        numerator = np.sum(weights * x_neighbors * y_neighbors, axis=1)
        denominator = np.sum(weights * x_neighbors * x_neighbors, axis=1) + ridge
        beta[start:end] = numerator / denominator
    return beta


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint_iter*.npz"))
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(output_dir: Path, iteration: int, params: np.ndarray, xb: np.ndarray, err: np.ndarray, history: list[dict]) -> None:
    tmp = output_dir / f"checkpoint_iter{iteration:03d}.tmp.npz"
    final = output_dir / f"checkpoint_iter{iteration:03d}.npz"
    np.savez_compressed(tmp, iteration=iteration, params=params, xb=xb, err=err, history=json.dumps(history))
    tmp.rename(final)


def load_checkpoint(path: Path) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    data = np.load(path, allow_pickle=False)
    history = json.loads(str(data["history"]))
    return int(data["iteration"]), data["params"], data["xb"], data["err"], history


def fixed_bandwidth_backfit(
    coords: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    bandwidths: np.ndarray,
    output_dir: Path,
    max_iter: int,
    tol: float,
    chunk_size: int,
    global_threshold_frac: float,
    ridge: float,
    resume: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    n, k = X.shape
    nn = NearestNeighbors(algorithm="kd_tree", metric="euclidean")
    nn.fit(coords)

    start_iter = 1
    history: list[dict] = []
    ckpt = latest_checkpoint(output_dir) if resume else None
    if ckpt is not None:
        last_iter, params, xb, err, history = load_checkpoint(ckpt)
        start_iter = last_iter + 1
        print(f"Resuming from {ckpt.name} at iteration {last_iter}", flush=True)
    else:
        params = initial_params_global(X, y)
        xb = params * X
        err = y - xb.sum(axis=1)
        save_checkpoint(output_dir, 0, params, xb, err, history)

    for iteration in range(start_iter, max_iter + 1):
        t0 = time.time()
        old_xb = xb.copy()
        new_xb = np.zeros_like(xb)
        new_params = np.zeros_like(params)

        print(f"Iteration {iteration}/{max_iter}", flush=True)
        for j in range(k):
            bw = int(bandwidths[j])
            target = xb[:, j] + err
            xj = X[:, j]
            if bw >= math.ceil(global_threshold_frac * n):
                beta = global_univariate_beta(xj, target, ridge)
                mode = "global"
            else:
                beta = local_univariate_beta(nn, coords, xj, target, bw, chunk_size, ridge)
                mode = "local"
            fitted_j = beta * xj
            err = target - fitted_j
            new_params[:, j] = beta
            new_xb[:, j] = fitted_j
            print(f"  term {j + 1:02d}/{k}: bw={bw:,} mode={mode}", flush=True)

        numerator = float(np.sum((new_xb - old_xb) ** 2) / n)
        denominator = float(np.sum(np.sum(new_xb, axis=1) ** 2))
        score = float(np.sqrt(numerator / denominator)) if denominator > 0 else float("nan")
        pred = new_xb.sum(axis=1)
        rmse = float(mean_squared_error(y, pred) ** 0.5)
        r2 = float(r2_score(y, pred))
        elapsed = time.time() - t0
        history.append(
            {
                "iteration": iteration,
                "score": score,
                "rmse": rmse,
                "r2": r2,
                "elapsed_seconds": elapsed,
            }
        )
        params = new_params
        xb = new_xb
        save_checkpoint(output_dir, iteration, params, xb, err, history)
        pd.DataFrame(history).to_csv(output_dir / "backfit_history.csv", index=False)
        print(f"  score={score:.8g} rmse={rmse:.6f} r2={r2:.6f} elapsed={elapsed/60:.1f} min", flush=True)
        if np.isfinite(score) and score < tol:
            print(f"Converged at iteration {iteration}", flush=True)
            break

    return params, xb.sum(axis=1), history


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aliases = parse_aliases(args.alias)
    work, y, coords, X, terms = load_and_prepare(args)
    bw_df = load_bandwidths(
        Path(args.bandwidth_file),
        terms,
        n=len(work),
        reference_n=args.reference_n,
        scale=args.scale_bandwidths,
        min_bw=args.min_bw,
        aliases=aliases,
    )
    bw_df.to_csv(output_dir / "fixed_scaled_bandwidths.csv", index=False)
    bandwidths = bw_df["scaled_bandwidth"].to_numpy(dtype=int)

    params, pred, history = fixed_bandwidth_backfit(
        coords=coords,
        X=X,
        y=y,
        bandwidths=bandwidths,
        output_dir=output_dir,
        max_iter=args.max_iter,
        tol=args.tol,
        chunk_size=args.chunk_size,
        global_threshold_frac=args.global_threshold_frac,
        ridge=args.ridge,
        resume=args.resume,
    )
    residual = y - pred

    coef_df = pd.DataFrame(params, columns=terms)
    coef_out = pd.concat([work[["x", "y"]].reset_index(drop=True), coef_df], axis=1)
    coef_out.to_parquet(output_dir / "mgwr_full_fixed_coefficients.parquet", index=False)

    resid_out = work[["x", "y"]].copy()
    resid_out["observed"] = y
    resid_out["predicted"] = pred
    resid_out["residual"] = residual
    resid_out.to_parquet(output_dir / "mgwr_full_fixed_residuals.parquet", index=False)

    metrics = {
        "method": "full_point_fixed_bandwidth_mgwr_backfitting",
        "input": str(Path(args.input).expanduser().resolve()),
        "response": args.response,
        "n_rows": int(len(work)),
        "terms": terms,
        "reference_n": int(args.reference_n),
        "scale_bandwidths": bool(args.scale_bandwidths),
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "chunk_size": int(args.chunk_size),
        "global_threshold_frac": float(args.global_threshold_frac),
        "r2": float(r2_score(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "bias_observed_minus_predicted": float(np.mean(residual)),
        "mean_abs_residual": float(np.mean(np.abs(residual))),
        "history": history,
    }
    (output_dir / "mgwr_full_fixed_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
