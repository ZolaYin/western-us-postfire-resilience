#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from run_westernus_multiscale_rf_softmoe_round1 import (
    BASE_PREDS,
    RANDOM_STATE,
    build_split_masks,
    prepare,
)


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
OUT_DIR = ROOT / "predicted_observed_modelset_inputs_2026-06-10" / "m2_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    raw = pd.read_parquet(INPUT)
    df = prepare(raw)
    group_cols = sorted([c for c in df.columns if c.startswith("EVT_group_")])
    predictors = [c for c in BASE_PREDS + group_cols if c in df.columns]
    needed = list(dict.fromkeys(["Resistance", "x", "y", *predictors]))
    work = (
        df[needed]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    split_masks = build_split_masks(work, block_km=100.0, random_state=RANDOM_STATE)
    X = work[predictors].to_numpy(dtype=np.float32)
    y = work["Resistance"].to_numpy(dtype=np.float32)
    coords = work[["x", "y"]].to_numpy(dtype=np.float64)

    rows = []
    for split_name, is_test in split_masks.items():
        train = ~is_test
        test = is_test
        model = RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=8,
            min_samples_leaf=1,
            max_features=1.0,
        )
        model.fit(X[train], y[train])
        pred = model.predict(X[test]).astype(np.float32)
        residual = (y[test] - pred).astype(np.float32)
        out = pd.DataFrame(
            {
                "x": coords[test, 0],
                "y": coords[test, 1],
                "observed": y[test],
                "predicted": pred,
                "residual": residual,
            }
        )
        out_path = OUT_DIR / f"m2_baseline_{split_name}_predictions.csv"
        out.to_csv(out_path, index=False)
        rows.append(
            {
                "variant": "m2_baseline",
                "split": split_name,
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "n_features": len(predictors),
                "output": str(out_path),
            }
        )
        print(f"Wrote: {out_path}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "m2_baseline_prediction_manifest.csv", index=False)


if __name__ == "__main__":
    main()
