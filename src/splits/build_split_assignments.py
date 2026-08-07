#!/usr/bin/env python3
"""Create response-specific random and projected 100 km block assignments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


DEFAULT_RESPONSES = ["Resistance", "T50", "T80", "IRI_good_pow2", "STAB_good_pow2"]
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--responses", nargs="+", default=DEFAULT_RESPONSES)
    parser.add_argument("--block-km", type=float, default=100.0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    std = values.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=values.index)
    return ((values - values.mean()) / std).astype(np.float32)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
    return out


def block_ids(df: pd.DataFrame, block_km: float) -> pd.Series:
    size_m = block_km * 1000.0
    bx = np.floor(pd.to_numeric(df["x"], errors="coerce") / size_m).astype("Int64")
    by = np.floor(pd.to_numeric(df["y"], errors="coerce") / size_m).astype("Int64")
    return bx.astype(str) + "_" + by.astype(str)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    raw = pd.read_parquet(input_path)
    if "pixel_id" not in raw or not raw["pixel_id"].is_unique:
        raise ValueError("pixel_id must exist and be unique")
    df = prepare(raw)

    assignments = df[["pixel_id", "x", "y"]].copy()
    assignments[f"block_id_{int(args.block_km)}km"] = block_ids(df, args.block_km)
    summary_rows: list[dict] = []

    baseline = list(BASE_TO_Z) + [
        "HUM_popdens_win10km_log_z",
        "HUM_viirs_near_t0_log_z",
        "x",
        "y",
        "x_sq_z",
        "y_sq_z",
        "xy_z",
    ]
    for response in args.responses:
        required = ["pixel_id", response, *baseline]
        missing = [column for column in required if column not in df]
        if missing:
            raise ValueError(f"Missing columns for {response}: {missing}")
        work = df[required].replace([np.inf, -np.inf], np.nan).dropna().copy()
        positions = np.arange(len(work))

        random_train, random_test = train_test_split(
            positions, test_size=args.test_size, random_state=args.random_state
        )
        groups = block_ids(work, args.block_km)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.random_state
        )
        block_train, block_test = next(splitter.split(positions, groups=groups))

        eligible_col = f"eligible_{response}"
        random_col = f"random_{response}"
        block_col = f"block{int(args.block_km)}km_{response}"
        assignments[eligible_col] = assignments["pixel_id"].isin(work["pixel_id"])
        assignments[random_col] = pd.Series(pd.NA, index=assignments.index, dtype="string")
        assignments[block_col] = pd.Series(pd.NA, index=assignments.index, dtype="string")

        random_test_ids = set(work.iloc[random_test]["pixel_id"].tolist())
        block_test_ids = set(work.iloc[block_test]["pixel_id"].tolist())
        eligible = assignments[eligible_col]
        assignments.loc[eligible, random_col] = "train"
        assignments.loc[assignments["pixel_id"].isin(random_test_ids), random_col] = "test"
        assignments.loc[eligible, block_col] = "train"
        assignments.loc[assignments["pixel_id"].isin(block_test_ids), block_col] = "test"

        summary_rows.append(
            {
                "response": response,
                "eligible_rows": int(len(work)),
                "random_train_rows": int(len(random_train)),
                "random_test_rows": int(len(random_test)),
                "block_train_rows": int(len(block_train)),
                "block_test_rows": int(len(block_test)),
                "block_train_groups": int(groups.iloc[block_train].nunique()),
                "block_test_groups": int(groups.iloc[block_test].nunique()),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(output_path, index=False)
    pd.DataFrame(summary_rows).to_csv(output_path.with_name("split_summary.csv"), index=False)
    metadata = {
        "input": Path(args.input).as_posix(),
        "output": Path(args.output).as_posix(),
        "block_km": args.block_km,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "responses": args.responses,
        "note": "Eligibility matches the complete-case baseline used by the M1/M2/M3 RF comparison.",
    }
    output_path.with_name("split_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"metadata": metadata, "summary": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
