#!/usr/bin/env python3
"""Run a standard complete-sample MGWR calibration.

This uses mgwr.sel_bw.Sel_BW on the complete-case table, rather than applying
sample-derived bandwidths directly. A pilot bandwidth file can still be used to
define broad search bounds and a stable initial GWR bandwidth.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from scipy.linalg import LinAlgWarning


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
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--pilot-bandwidth-file", default=None, help="Optional sample MGWR bandwidth CSV.")
    parser.add_argument("--reference-n", type=int, default=12000, help="Sample size used by the pilot bandwidth file.")
    parser.add_argument("--min-bw-factor", type=float, default=0.25, help="Lower bound as a fraction of scaled pilot bandwidth.")
    parser.add_argument("--max-bw-factor", type=float, default=2.50, help="Upper bound as a fraction of scaled pilot bandwidth.")
    parser.add_argument("--hard-min-bw", type=int, default=80, help="Smallest adaptive neighbor count allowed.")
    parser.add_argument("--hard-max-frac", type=float, default=0.999, help="Largest adaptive bandwidth as a fraction of n.")
    parser.add_argument("--init-mode", choices=["none", "median", "p75", "max"], default="p75")
    parser.add_argument("--init-bandwidth", type=float, default=None, help="Override scalar initial GWR bandwidth.")
    parser.add_argument("--criterion", default="AICc", choices=["AICc", "AIC", "BIC", "CV"])
    parser.add_argument("--search-method", default="golden_section", choices=["golden_section", "interval"])
    parser.add_argument("--interval", type=float, default=0.0, help="Interval for interval search.")
    parser.add_argument("--tol", type=float, default=1e-5, help="Single bandwidth search tolerance.")
    parser.add_argument("--max-iter", type=int, default=80, help="Single bandwidth search max iterations.")
    parser.add_argument("--tol-multi", type=float, default=5e-5, help="MGWR backfitting tolerance.")
    parser.add_argument("--max-iter-multi", type=int, default=12, help="MGWR backfitting max iterations.")
    parser.add_argument("--bws-same-times", type=int, default=3, help="Stop after repeated identical bandwidth vectors.")
    parser.add_argument("--rss-score", action="store_true", help="Use RSS score for backfitting convergence.")
    parser.add_argument("--n-jobs", type=int, default=8, help="Joblib workers used by mgwr.")
    parser.add_argument("--fit-inference", action="store_true", help="Run MGWR.fit() after selection.")
    parser.add_argument("--n-chunks", type=int, default=4, help="Chunks for MGWR.fit() if inference is enabled.")
    parser.add_argument("--sample-n", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--alias", action="append", default=[], help="Column alias old=new; may be repeated.")
    return parser.parse_args()


def resolve_predictors(df: pd.DataFrame, predictor_terms: list[str], aliases: dict[str, str]) -> tuple[list[str], list[str]]:
    input_cols: list[str] = []
    output_terms: list[str] = []
    for term in predictor_terms:
        col = term if term in df.columns else aliases.get(term, term)
        if col not in df.columns:
            raise ValueError(f"Predictor {term!r} resolved to {col!r}, but that column is absent.")
        input_cols.append(col)
        output_terms.append(col if term in aliases else term)
    return input_cols, output_terms


def load_work(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    predictors = read_lines(Path(args.predictors_file))
    aliases = parse_aliases(args.alias)
    df = pd.read_parquet(args.input)
    if args.sample_n is not None and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.random_state).reset_index(drop=True)

    input_cols, output_terms = resolve_predictors(df, predictors, aliases)
    required = [args.response, "x", "y", *input_cols]
    work = df[required].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if work.empty:
        raise ValueError("No complete rows remain after filtering.")
    y = work[[args.response]].to_numpy(dtype=float)
    X = work[input_cols].to_numpy(dtype=float)
    coords = work[["x", "y"]].to_numpy(dtype=float)
    terms = ["Intercept", *output_terms]
    return work, y, X, coords, terms


def pilot_scaled_bandwidths(
    path: Path,
    terms: list[str],
    n: int,
    reference_n: int,
    aliases: dict[str, str],
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if not {"term", "bandwidth"}.issubset(raw.columns):
        raise ValueError("Pilot bandwidth file must contain term and bandwidth columns.")
    bw_map = dict(zip(raw["term"].astype(str), pd.to_numeric(raw["bandwidth"], errors="raise")))
    inverse_alias = {v: k for k, v in aliases.items()}
    rows = []
    ratio = n / float(reference_n)
    for term in terms:
        source_term = term
        if source_term not in bw_map and term in inverse_alias:
            source_term = inverse_alias[term]
        if source_term not in bw_map:
            raise ValueError(f"Pilot bandwidth file is missing {term!r}.")
        source_bw = float(bw_map[source_term])
        rows.append(
            {
                "term": term,
                "source_term": source_term,
                "source_bandwidth": source_bw,
                "scaled_bandwidth": source_bw * ratio,
                "scale_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)


def build_search_bounds(args: argparse.Namespace, terms: list[str], n: int) -> tuple[list[int | None], list[int | None], float | None, pd.DataFrame | None]:
    if args.pilot_bandwidth_file is None:
        return [None], [None], args.init_bandwidth, None

    aliases = parse_aliases(args.alias)
    pilot = pilot_scaled_bandwidths(Path(args.pilot_bandwidth_file), terms, n, args.reference_n, aliases)
    hard_max = max(args.hard_min_bw + 1, int(np.floor((n - 1) * args.hard_max_frac)))

    mins: list[int] = []
    maxs: list[int] = []
    for bw in pilot["scaled_bandwidth"].to_numpy(dtype=float):
        low = int(np.floor(bw * args.min_bw_factor))
        high = int(np.ceil(bw * args.max_bw_factor))
        low = max(args.hard_min_bw, min(low, n - 2))
        high = max(low + 1, min(high, hard_max))
        mins.append(low)
        maxs.append(high)

    init = args.init_bandwidth
    if init is None and args.init_mode != "none":
        values = pilot["scaled_bandwidth"].to_numpy(dtype=float)
        if args.init_mode == "median":
            init = float(np.median(values))
        elif args.init_mode == "p75":
            init = float(np.quantile(values, 0.75))
        elif args.init_mode == "max":
            init = float(np.max(values))
        init = float(np.clip(round(init), args.hard_min_bw, hard_max))

    pilot = pilot.copy()
    pilot["search_min"] = mins
    pilot["search_max"] = maxs
    return mins, maxs, init, pilot


def compute_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    observed = y.ravel()
    fitted = pred.ravel()
    residual = observed - fitted
    sst = float(np.sum((observed - observed.mean()) ** 2))
    sse = float(np.sum(residual**2))
    return {
        "r2": float(1.0 - sse / sst),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias_observed_minus_predicted": float(np.mean(residual)),
        "mean_abs_residual": float(np.mean(np.abs(residual))),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    args = parse_args()
    t0 = time.time()
    input_path = Path(args.input).expanduser().resolve()
    predictors_path = Path(args.predictors_file).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    work, y, X, coords, terms = load_work(args)
    n = len(work)
    mins, maxs, init_bw, pilot = build_search_bounds(args, terms, n)
    if pilot is not None:
        pilot.to_csv(output_dir / "pilot_scaled_search_bounds.csv", index=False)

    print(
        json.dumps(
            {
                "event": "start",
                "input": Path(args.input).as_posix(),
                "response": args.response,
                "rows_used": n,
                "terms": terms,
                "init_bandwidth": init_bw,
                "multi_bw_min": mins,
                "multi_bw_max": maxs,
                "fit_inference": bool(args.fit_inference),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    selector = Sel_BW(
        coords,
        y,
        X,
        multi=True,
        kernel="bisquare",
        fixed=False,
        constant=True,
        n_jobs=args.n_jobs,
    )
    bandwidths = selector.search(
        search_method=args.search_method,
        criterion=args.criterion,
        interval=args.interval,
        tol=args.tol,
        max_iter=args.max_iter,
        init_multi=init_bw,
        tol_multi=args.tol_multi,
        max_iter_multi=args.max_iter_multi,
        multi_bw_min=mins,
        multi_bw_max=maxs,
        bws_same_times=args.bws_same_times,
        rss_score=args.rss_score,
        verbose=True,
    )

    bandwidths = np.asarray(bandwidths, dtype=float).ravel()
    params = np.asarray(selector.params, dtype=float)
    X_const = np.column_stack([np.ones(n, dtype=float), X])
    pred = np.sum(X_const * params, axis=1).reshape(-1, 1)

    inference_metrics: dict[str, float | None] = {
        "aic": None,
        "bic": None,
        "adj_r2": None,
    }
    if args.fit_inference:
        print(json.dumps({"event": "fit_inference_start", "n_chunks": args.n_chunks}), flush=True)
        model = MGWR(coords, y, X, selector=selector, constant=True, n_jobs=args.n_jobs)
        results = model.fit(n_chunks=args.n_chunks)
        pred = results.predy
        params = results.params
        inference_metrics = {
            "aic": float(results.aic),
            "bic": float(results.bic),
            "adj_r2": float(results.adj_R2),
        }

    coef_df = pd.DataFrame(params, columns=terms)
    coef_out = pd.concat([work[["x", "y"]].reset_index(drop=True), coef_df.reset_index(drop=True)], axis=1)
    coef_out.to_parquet(output_dir / "mgwr_complete_coefficients.parquet", index=False)

    observed = y.ravel()
    fitted = pred.ravel()
    resid_out = work[["x", "y"]].copy()
    resid_out["observed"] = observed
    resid_out["predicted"] = fitted
    resid_out["residual"] = observed - fitted
    resid_out.to_parquet(output_dir / "mgwr_complete_residuals.parquet", index=False)

    pd.DataFrame({"term": terms, "bandwidth": bandwidths}).to_csv(output_dir / "mgwr_complete_bandwidths.csv", index=False)
    if len(selector.bw) > 2:
        pd.DataFrame(np.asarray(selector.bw[1], dtype=float), columns=terms).to_csv(
            output_dir / "mgwr_complete_bandwidth_history.csv",
            index_label="iteration",
        )
        pd.DataFrame({"score": np.asarray(selector.bw[2], dtype=float).ravel()}).to_csv(
            output_dir / "mgwr_complete_backfit_scores.csv",
            index_label="iteration",
        )

    metrics = {
        "method": "standard_complete_sample_mgwr_search",
        "input_path": Path(args.input).as_posix(),
        "predictors_file": Path(args.predictors_file).as_posix(),
        "response": args.response,
        "rows_used": int(n),
        "predictor_count": int(X.shape[1]),
        "terms": terms,
        "bandwidths": bandwidths.tolist(),
        "search": {
            "criterion": args.criterion,
            "search_method": args.search_method,
            "tol": args.tol,
            "max_iter": args.max_iter,
            "tol_multi": args.tol_multi,
            "max_iter_multi": args.max_iter_multi,
            "bws_same_times": args.bws_same_times,
            "rss_score": bool(args.rss_score),
            "init_bandwidth": init_bw,
            "multi_bw_min": mins,
            "multi_bw_max": maxs,
        },
        **compute_metrics(y, pred),
        **inference_metrics,
        "elapsed_seconds": float(time.time() - t0),
    }
    (output_dir / "mgwr_complete_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "output_dir": str(output_dir), **compute_metrics(y, pred)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
