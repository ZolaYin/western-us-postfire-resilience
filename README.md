# Western U.S. Post-fire Forest Resilience

Reproducibility package for a 1 km analysis of post-fire forest resilience across 11 western U.S. states. The publication workflow links three complementary response dimensions (Resistance, integrated recovery, and post-fire stability) to environmental and human drivers using random-forest transferability tests, OLS spatial diagnostics, GWR/MGWR, and EPA Level III ecoregion management zoning.

This repository is a **private GitHub release draft**. It has been uploaded for review but has not been made public.

## Repository contents

```text
config/                 Final predictor definitions
data/processed/         Final model table and schema
data/splits/            Reproducible random and 100 km block assignments
docs/                   Data sources, dictionary, workflow, and HPRC inventory
src/preprocessing/      Response/driver/table preparation
src/splits/             Split-generation code
src/models/             RF, OLS, GWR/MGWR code
src/zoning/             EPA Level III zoning code
hpc/                    Portable Slurm templates
results/                Small, publication-relevant result tables
archive/                Historical model-code snapshot, not the recommended entry point
provenance/             File manifests and checksums
```

## Final analysis table

`data/processed/westernus_model_table.parquet` is the immutable table used to assemble the public workflow:

- 133,409 fire-affected forest pixels
- 83 columns
- 1 km projected coordinates in NAD83 / CONUS Albers (EPSG:5070)
- fire years 2005–2022
- 11 western states: Arizona, California, Colorado, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, and Wyoming

The table contains pixel identifiers, fire timing/severity, resilience responses, raw predictors, selected standardized predictors, WGS84 coordinates, and five broad reporting regions. See `docs/DATA_DICTIONARY.md` and `data/processed/westernus_model_table_schema.csv`.

## Main analysis sequence

1. Rebuild response metrics from annual RESI and MTBS-aligned inputs.
2. Join topography/soil, forest structure/type, climate, human-footprint, road, and trail drivers.
3. Generate deterministic random and 100 km spatial-block partitions.
4. Fit RF M1/M2/M3 forest-type representations and quantify the transferability gap.
5. Fit the global OLS reference and test residual spatial autocorrelation.
6. Calibrate GWR/MGWR on the shared 12,000-point sample; optionally run complete-sample MGWR.
7. Translate local MGWR effects to EPA Level III ecoregion management zones.

Exact commands are in `docs/REPRODUCIBILITY.md`.

## Publication status and release blockers

Before publishing this repository, resolve the items below:

- add the final manuscript title, author list, DOI, and citation;
- choose a code license and a separate derived-data license;
- add the public archive/DOI for the annual RESI input, or document how it can be requested;
- recover the exact Google Earth Engine collection/asset ID used for annual TCC exports;
- confirm redistribution terms for every derived column and for the bundled final table;
- recover or formally document the rule that created the five broad `region` labels;
- replace draft wording and remove the `archive/` directory if the journal package should expose only the final pipeline.

## Reproducibility boundary

The final table, saved partitions, final model code, and key result tables can be distributed directly from this draft. Raw rasters are not bundled because of size and source-specific distribution terms; `docs/DATA_SOURCES.md` records the authoritative access points and the local processing role of each input.

## Safety note

