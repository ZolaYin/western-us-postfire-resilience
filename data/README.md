# Data layout

`processed/westernus_model_table.parquet` is the final analysis-ready table. Its schema and compact summary are stored beside it. `splits/westernus_split_assignments.parquet` stores the deterministic random and 100 km spatial-block assignments keyed by `pixel_id`.

Raw rasters are intentionally not bundled. Their authoritative access points, roles, spatial/temporal coverage, and release status are documented in `../docs/DATA_SOURCES.md` and `../provenance/raw_input_manifest.csv`.

Before a public release, confirm that redistribution of every derived field and the combined Parquet table is allowed, and resolve the two source-provenance gaps identified in `../docs/RELEASE_CHECKLIST.md`.
