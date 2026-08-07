#!/usr/bin/env python3
"""Write a compact, machine-readable description of the released model table."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Parquet table.")
    parser.add_argument("--schema-output", required=True, help="Output schema CSV.")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    schema_path = Path(args.schema_output).expanduser().resolve()
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else schema_path.with_name("westernus_model_table_summary.json")
    )

    df = pd.read_parquet(input_path)
    rows = []
    for column in df.columns:
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "null_fraction": float(series.isna().mean()),
                "unique_non_null": int(series.nunique(dropna=True)),
            }
        )

    schema_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(schema_path, index=False)

    summary = {
        "file": input_path.name,
        "bytes": input_path.stat().st_size,
        "sha256": sha256(input_path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "pixel_id_unique": bool(df["pixel_id"].is_unique) if "pixel_id" in df else None,
        "crs_for_x_y": "EPSG:5070",
        "t0_year_min": int(df["t0_year"].min()) if "t0_year" in df else None,
        "t0_year_max": int(df["t0_year"].max()) if "t0_year" in df else None,
        "region_counts": (
            {str(k): int(v) for k, v in df["region"].value_counts(dropna=False).items()}
            if "region" in df
            else None
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
