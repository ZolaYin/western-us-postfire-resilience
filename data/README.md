# Data layout

`processed/westernus_model_table.parquet` is the final analysis-ready table. Its field-level schema and compact summary are stored beside it.

`splits/westernus_split_assignments.parquet` stores the deterministic random and 100 km spatial-block assignments keyed by `pixel_id`; `split_metadata.json` records the parameters used to create them.

Download options:

- use the files directly from this GitHub directory; or
- use the [public Google Drive data release](https://drive.google.com/drive/folders/1C1kPp0hS7RW5zTaVD0c7O88LxmNuJ3wk), which also contains larger derived artifacts and checksum documentation.

Raw third-party rasters are intentionally not bundled. Official access links, exact Earth Engine collection IDs, project transformations, attribution, and reuse notes are documented in [`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md) and [`../provenance/raw_input_manifest.csv`](../provenance/raw_input_manifest.csv).
