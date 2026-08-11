# Data access

## GitHub

The following analysis inputs are small enough to remain versioned with the code:

- `data/processed/westernus_model_table.parquet`: final 133,409-row model table;
- `data/processed/westernus_model_table_schema.csv`: field names and physical types;
- `data/processed/westernus_model_table_summary.json`: compact validation summary;
- `data/splits/westernus_split_assignments.parquet`: deterministic random and 100 km spatial-block assignments;
- `data/splits/split_metadata.json` and `split_summary.csv`: split parameters and counts;
- `results/`: retained RF, OLS, MGWR, and zoning outputs.

## Google Drive

The public read-only release folder is:

**[WesternUS_Postfire_Resilience_Public_Data](https://drive.google.com/drive/folders/1C1kPp0hS7RW5zTaVD0c7O88LxmNuJ3wk)**

Directory layout:

```text
01_analysis_ready_data/       final table, splits, and portable data bundle
02_large_derived_artifacts/   larger intermediate/derived files used by zoning
03_documentation_and_checksums/ source notes, licenses, and SHA-256 manifests
```

Files in Google Drive are convenience copies; the checksums in `provenance/drive_release_manifest.csv` identify the exact release bytes.

## Raw source data

Raw third-party rasters are not mirrored. Use the complete official link table in [`DATA_SOURCES.md`](DATA_SOURCES.md) and run the released preprocessing code. In particular, the annual RESI and TCC Earth Engine collection identifiers and export scripts are fully recorded there.

## Integrity check

On macOS or Linux, verify a downloaded file with:

```bash
shasum -a 256 path/to/downloaded/file
```

Compare the output with `provenance/file_manifest.csv` or `provenance/drive_release_manifest.csv`.
