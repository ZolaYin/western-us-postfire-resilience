#!/usr/bin/env python3
"""Assign and validate the five broad reporting regions.

The regions are deterministic geographic reporting strata, not administrative
or ecological boundary products. Longitude and latitude must be WGS84 degrees.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REGION_ORDER = ["PNW", "CA_med", "S_Rockies", "N_Rockies", "SW_dry"]


def assign_reporting_regions(lon: pd.Series, lat: pd.Series) -> np.ndarray:
    """Return reporting-region labels for WGS84 coordinates.

    Rules, evaluated in order:
    - west of 118 W and at/north of 44 N: PNW
    - west of 118 W and south of 44 N: CA_med
    - at/east of 118 W and at/north of 44 N: N_Rockies
    - at/east of 118 W and from 37 N (inclusive) to 44 N: S_Rockies
    - at/east of 118 W and south of 37 N: SW_dry
    """

    if lon.isna().any() or lat.isna().any():
        raise ValueError("Longitude and latitude must not contain missing values")

    return np.select(
        [
            (lon < -118.0) & (lat >= 44.0),
            (lon < -118.0) & (lat < 44.0),
            (lon >= -118.0) & (lat >= 44.0),
            (lon >= -118.0) & (lat >= 37.0),
        ],
        ["PNW", "CA_med", "N_Rockies", "S_Rockies"],
        default="SW_dry",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path, help="Input Parquet table")
    parser.add_argument("--output", type=Path, help="Optional output Parquet path")
    parser.add_argument("--lon", default="lon_wgs84")
    parser.add_argument("--lat", default="lat_wgs84")
    parser.add_argument("--region", default="region")
    args = parser.parse_args()

    df = pd.read_parquet(args.table)
    assigned = assign_reporting_regions(df[args.lon], df[args.lat])

    if args.region in df.columns:
        mismatch = int((df[args.region].astype(str).to_numpy() != assigned).sum())
        print(f"Validated {len(df):,} rows; mismatches={mismatch:,}")
        if mismatch:
            raise SystemExit(1)
    else:
        df[args.region] = assigned

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output, index=False)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
