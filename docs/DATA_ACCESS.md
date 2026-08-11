# Data access

## GitHub

The following analysis inputs are small enough to remain versioned with the code:

- `data/processed/westernus_model_table.parquet`: final 133,409-row model table;
- `data/processed/westernus_model_table_schema.csv`: field names and physical types;
- `data/processed/westernus_model_table_dictionary.csv`: one-row-per-column units, windows, definitions, derivations, and sources;
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

The five exact inputs required to rebuild the final zoning are in `02_large_derived_artifacts/`:

- [`spatialized_point_zones_full_candidate.parquet`](https://drive.google.com/file/d/1Drj3YzEIK_0CY-4EyGaashdeosp2rt99/view);
- [`Resistance_mgwr_complete_coefficients.parquet`](https://drive.google.com/file/d/1r87WJvx7ufoo5aIbmpB7AzfTUmA6CH9w/view);
- [`IRI_good_pow2_mgwr_complete_coefficients.parquet`](https://drive.google.com/file/d/1QJXVhXZRu99sYZtijmjyF5NvIBmTU616/view);
- [`STAB_good_pow2_mgwr_complete_coefficients.parquet`](https://drive.google.com/file/d/14iQ5FASFuP7lwhUFi1expS9ZgAGIgkUo/view);
- [`westernus_epa_l3_zoning_analysis_units.gpkg`](https://drive.google.com/file/d/1vze_NF-Oa8IaKI26510Zp0nSGs4FqA17/view).

The first file is the input point table; `points_with_epa_l3_assignment_multiresponse.parquet` is an output of the zoning step. The three complete-sample coefficient tables and exact EPA-derived analysis-unit layer close the former Step 9 provenance gap. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md#9-build-management-zones) for commands and SHA-256 values.

## Raw source data

Raw third-party rasters are not mirrored. Use the complete official link table in [`DATA_SOURCES.md`](DATA_SOURCES.md) and run the released preprocessing code. In particular, the annual RESI and TCC Earth Engine collection identifiers and export scripts are fully recorded there.

## Integrity check

On macOS or Linux, verify a downloaded file with:

```bash
shasum -a 256 path/to/downloaded/file
```

Compare the output with `provenance/file_manifest.csv` or `provenance/drive_release_manifest.csv`.
