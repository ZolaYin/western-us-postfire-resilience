#!/usr/bin/env python3
"""Exploratory full-data MGWR + RF residual-correction screen.

This is an upper-bound/descriptive screen, not a leakage-free validation:
the MGWR coefficient surfaces used here were fit on the full sample.
"""
from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


BASE = Path(__file__).resolve().parent
MODEL2 = runpy.run_path(str(BASE / "run_rf_mgwrscale_model2_compare.py"))
BASE29 = MODEL2["BASE29"]
prepare = MODEL2["prepare"]
block_split = MODEL2["block_split"]

INPUT = BASE / "rf_mgwrscale_complete_mgwr_screen_2026-07-02" / "common_resistance_iri_stab_candidate_table.parquet"
OUT_DIR = BASE / "rf_mgwrscale_complete_mgwr_screen_2026-07-02" / "global_mgwr_rf_residual_oracle"

COEFS = {
    "Resistance": BASE / "mgwr_complete_sample_2026-06-24" / "complete_sample_mgwr_20260624_113719" / "mgwr_complete_coefficients.parquet",
    "IRI_good_pow2": BASE / "mgwr_complete_sample_2026-06-24" / "complete_sample_mgwr_IRI_good_pow2_20260625_1216" / "mgwr_complete_coefficients.parquet",
    "STAB_good_pow2": BASE / "mgwr_complete_sample_2026-06-24" / "complete_sample_mgwr_STAB_good_pow2_20260625_1216" / "mgwr_complete_coefficients.parquet",
}


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def add_mgwr_prediction(work: pd.DataFrame, response: str) -> pd.DataFrame:
    coef = pd.read_parquet(COEFS[response])
    pred_cols = [c for c in coef.columns if c not in {"x", "y", "Intercept"}]
    keep_cols = ["x", "y", "Intercept"] + pred_cols
    merged = work.merge(coef[keep_cols], on=["x", "y"], how="inner", suffixes=("", "_coef"))
    pred = merged["Intercept"].to_numpy(dtype=float)
    for col in pred_cols:
        pred += merged[f"{col}_coef"].to_numpy(dtype=float) * merged[col].to_numpy(dtype=float)
    merged["mgwr_full_pred"] = pred
    merged["mgwr_full_residual"] = merged[response].to_numpy(dtype=float) - pred
    return merged


def eval_response(response: str, n_trees: int = 120, seed: int = 42) -> list[dict[str, object]]:
    raw = pd.read_parquet(INPUT)
    df = prepare(raw)
    avail_base = [p for p in BASE29 if p in df.columns]
    needed = list(dict.fromkeys([response, "x", "y"] + avail_base))
    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    work = add_mgwr_prediction(work, response)

    coords = work[["x", "y"]].to_numpy(dtype=float)
    y = work[response].to_numpy(dtype=float)
    X_base = work[avail_base].to_numpy(dtype=float)
    X_plus_pred = np.column_stack([X_base, work["mgwr_full_pred"].to_numpy(dtype=float)])
    residual = work["mgwr_full_residual"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    is_test_rnd = rng.random(len(work)) < 0.2
    is_test_blk = block_split(coords, 100_000.0, seed)

    rows: list[dict[str, object]] = []
    for cv, test_mask in [("randCV", is_test_rnd), ("blkCV", is_test_blk)]:
        train_mask = ~test_mask
        y_test = y[test_mask]

        # Full-data MGWR prediction, evaluated on held-out rows. This is leaky.
        pred_mgwr = work.loc[test_mask, "mgwr_full_pred"].to_numpy(dtype=float)
        rows.append({
            "response": response,
            "cv": cv,
            "model": "global_mgwr_full_data_oracle",
            "n": len(work),
            "n_features": len(avail_base),
            "r2": float(r2_score(y_test, pred_mgwr)),
            "rmse": rmse(y_test, pred_mgwr),
        })

        rf_base = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        rf_base.fit(X_base[train_mask], y[train_mask])
        pred_base = rf_base.predict(X_base[test_mask])
        rows.append({
            "response": response,
            "cv": cv,
            "model": "rf_base29",
            "n": len(work),
            "n_features": len(avail_base),
            "r2": float(r2_score(y_test, pred_base)),
            "rmse": rmse(y_test, pred_base),
        })

        rf_plus = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        rf_plus.fit(X_plus_pred[train_mask], y[train_mask])
        pred_plus = rf_plus.predict(X_plus_pred[test_mask])
        rows.append({
            "response": response,
            "cv": cv,
            "model": "rf_base29_plus_global_mgwr_pred_oracle",
            "n": len(work),
            "n_features": X_plus_pred.shape[1],
            "r2": float(r2_score(y_test, pred_plus)),
            "rmse": rmse(y_test, pred_plus),
        })

        rf_resid = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)
        rf_resid.fit(X_base[train_mask], residual[train_mask])
        pred_resid = pred_mgwr + rf_resid.predict(X_base[test_mask])
        rows.append({
            "response": response,
            "cv": cv,
            "model": "global_mgwr_plus_rf_residual_oracle",
            "n": len(work),
            "n_features": len(avail_base),
            "r2": float(r2_score(y_test, pred_resid)),
            "rmse": rmse(y_test, pred_resid),
        })

    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for response in ["Resistance", "IRI_good_pow2", "STAB_good_pow2"]:
        print(f"Running {response}", flush=True)
        all_rows.extend(eval_response(response))
    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_DIR / "global_mgwr_rf_residual_oracle_metrics.csv", index=False)
    print(out.pivot_table(index=["response", "model"], columns="cv", values="r2").to_string())
    print(OUT_DIR)


if __name__ == "__main__":
    main()
