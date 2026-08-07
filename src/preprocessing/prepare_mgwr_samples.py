#!/usr/bin/env python3
"""Build the complete-case MGWR table and a shared deterministic sample."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESPONSES = ["Resistance", "IRI_good_pow2", "STAB_good_pow2"]
PREDICTOR_SOURCES = {
    "TS_elev_m_z": ("TS_elev_m", "z"),
    "TS_slope_deg_z": ("TS_slope_deg", "z"),
    "TS_SOC_0_30cm_z": ("TS_SOC_0_30cm", "z"),
    "FS_TCC_t0_z": ("FS_TCC_t0", "z"),
    "FS_CBH_t0agg_z": ("FS_CBH_t0agg", "z"),
    "HUM_roaddens_r5km_z": ("HUM_roaddens_r5km", "z"),
    "HUM_traildens_r10km_z": ("HUM_traildens_r10km", "z"),
    "HUM_viirs_near_t0_log_z": ("HUM_viirs_near_t0", "log1p_z"),
    "HUM_imperv_near_t0_z": ("HUM_imperv_near_t0", "z"),
    "CLIM_pr_sum_pre_z": ("CLIM_pr_sum_pre", "z"),
    "CLIM_tmmn_mean_pre_z": ("CLIM_tmmn_mean_pre", "z"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-n", type=int, default=12000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--responses", nargs="+", default=DEFAULT_RESPONSES)
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    return ((values - values.mean()) / std).astype(np.float32)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(input_path)
    for output_name, (source_name, transform) in PREDICTOR_SOURCES.items():
        if source_name not in raw:
            raise ValueError(f"Required source column is missing: {source_name}")
        source = pd.to_numeric(raw[source_name], errors="coerce")
        if transform == "log1p_z":
            source = np.log1p(source.clip(lower=0))
        raw[output_name] = zscore(source)

    predictors = list(PREDICTOR_SOURCES)
    required = ["pixel_id", "x", "y", *args.responses, *predictors]
    missing = [column for column in required if column not in raw]
    if missing:
        raise ValueError(f"Input table is missing columns: {missing}")

    complete = (
        raw[required]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values("pixel_id")
        .reset_index(drop=True)
    )
    complete_path = output_dir / "intersection_full.parquet"
    complete.to_parquet(complete_path, index=False)

    sample_n = min(args.sample_n, len(complete))
    rng = np.random.default_rng(args.random_state)
    positions = np.sort(rng.choice(len(complete), size=sample_n, replace=False))
    sample = complete.iloc[positions].reset_index(drop=True)
    sample_path = output_dir / f"sample_n{sample_n}_seed{args.random_state}.parquet"
    sample.to_parquet(sample_path, index=False)
    (output_dir / "predictors.txt").write_text("\n".join(predictors) + "\n", encoding="utf-8")

    metadata = {
        "input": Path(args.input).as_posix(),
        "responses": args.responses,
        "predictors": predictors,
        "complete_rows": int(len(complete)),
        "sample_rows": int(len(sample)),
        "random_state": int(args.random_state),
        "sampling": "numpy.default_rng.choice without replacement; selected positions sorted",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
