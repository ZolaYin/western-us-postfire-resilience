# Western U.S. Post-fire Forest Resilience

Reproducibility package for a 1 km analysis of post-fire forest resilience across 11 western U.S. states. The workflow links three response dimensions—Resistance, integrated recovery, and post-fire stability—to environmental and human drivers using random-forest transferability tests, OLS spatial diagnostics, GWR/MGWR, and EPA Level III ecoregion management zoning.

## Data and reproducibility access

- [Analysis-ready data in this repository](data/): final model table, data dictionary, schema, and deterministic split assignments.
- [Public Google Drive data release](https://drive.google.com/drive/folders/1C1kPp0hS7RW5zTaVD0c7O88LxmNuJ3wk): downloadable copies of the analysis-ready files, larger derived artifacts, and checksum documentation.
- [Reproduction instructions](docs/REPRODUCIBILITY.md): environment setup and commands for preprocessing, splitting, modeling, and zoning.
- [Complete data provenance](docs/DATA_SOURCES.md): product identifiers, variables, transformations, access links, and reuse notes.

The raw third-party rasters are not repackaged. They remain available from the official providers linked below; project-created tables and results are distributed directly through GitHub and Google Drive.

## Repository contents

```text
config/                 Final predictor definitions
data/processed/         Final model table, schema, and summary
data/splits/            Reproducible random and 100 km block assignments
docs/                   Sources, dictionary, data access, and workflow
src/preprocessing/      RESI/TCC export and table-preparation code
src/splits/             Split-generation code
src/models/             RF, OLS, GWR/MGWR code
src/zoning/             EPA Level III zoning code
hpc/                    Portable Slurm templates
results/                Publication-relevant output tables and spatial layer
archive/                Analysis-history code retained for provenance
provenance/             File manifests and checksums
```

## Final analysis table

[`data/processed/westernus_model_table.parquet`](data/processed/westernus_model_table.parquet) is the analysis-ready table used by the released workflow:

- 133,409 fire-affected forest pixels and 83 columns;
- 1 km projected coordinates in NAD83 / CONUS Albers (EPSG:5070);
- fire years 2005–2022;
- Arizona, California, Colorado, Idaho, Montana, Nevada, New Mexico, Oregon, Utah, Washington, and Wyoming;
- SHA-256: `2675c4b1b24846bdc52f72b2f15f89cd086626cdfca6a654bf28bf6d9897ec7a`.

The table contains identifiers, fire timing and severity, resilience responses, raw predictors, selected standardized predictors, WGS84 coordinates, and five broad reporting regions. Field definitions are in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) and [`data/processed/westernus_model_table_schema.csv`](data/processed/westernus_model_table_schema.csv).

## Complete upstream data sources

| Input | Official source | Use in this project |
|---|---|---|
| MTBS fire history and severity | [MTBS Direct Download](https://www.mtbs.gov/direct-download) | Fire year, severity, and burned-pixel mask |
| Landsat 5 Collection 2 Level-2 | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2) | Project-derived annual RESI |
| Landsat 7 Collection 2 Level-2 | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2) | Project-derived annual RESI |
| Landsat 8 Collection 2 Level-2 | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2) | Project-derived annual RESI |
| Landsat 9 Collection 2 Level-2 | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2) | Project-derived annual RESI |
| TIGER/Line state boundaries | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/TIGER_2018_States) | Western 11-state RESI/TCC export boundary |
| LANDFIRE EVT and canopy base height | [LANDFIRE Data Distribution](https://landfire.gov/data) | Forest type/mask and near-fire-year canopy structure |
| USFS Tree Canopy Cover v2023-5 | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2023_REL_TCC_v2023-5) | Annual `NLCD_Percent_Tree_Canopy_Cover` sampled at fire year |
| gridMET | [Climatology Lab](https://www.climatologylab.org/gridmet.html) | Pre/post-fire climate and water-balance predictors |
| USGS 3DEP | [The National Map data delivery](https://www.usgs.gov/the-national-map-data-delivery) | Elevation and terrain derivatives |
| SoilGrids 250 m v2 | [ISRIC documentation](https://docs.isric.org/globaldata/soilgrids/) | 0–30 cm soil organic carbon |
| NLCD / Annual NLCD | [MRLC data access](https://www.mrlc.gov/data) | Near-fire-year imperviousness and forest masking |
| VIIRS annual nighttime lights | [Earth Observation Group](https://eogdata.mines.edu/products/vnl/) | Near-fire-year nighttime-light intensity |
| GPWv4 population density | [NASA SEDAC](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4) | Population-pressure predictor |
| OpenStreetMap regional extracts | [Geofabrik U.S. downloads](https://download.geofabrik.de/north-america/us.html) | 5 km road and 10 km trail density |
| EPA Level III ecoregions | [EPA download page](https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states) | Management-zone aggregation and boundaries |

Annual RESI is not a missing external dataset: it is produced by [`src/preprocessing/export_annual_resi_gee.js`](src/preprocessing/export_annual_resi_gee.js) from May–September Landsat surface-reflectance composites. The exact TCC collection and export code are recorded in [`src/preprocessing/export_annual_tcc_gee.js`](src/preprocessing/export_annual_tcc_gee.js).

## Broad reporting regions

The `region` field is a deterministic longitude/latitude partition, not an external boundary dataset. The verified rule is:

- west of 118°W: `PNW` at or north of 44°N, otherwise `CA_med`;
- at or east of 118°W: `N_Rockies` at or north of 44°N, `S_Rockies` from 37°N to 44°N, otherwise `SW_dry`.

[`src/preprocessing/assign_reporting_regions.py`](src/preprocessing/assign_reporting_regions.py) reproduces and validates the labels; it matches all 133,409 released rows.

## Main analysis sequence

1. Build annual RESI and fire-aligned response metrics.
2. Join topography/soil, forest structure/type, climate, human-footprint, road, and trail drivers.
3. Generate deterministic random and 100 km spatial-block partitions.
4. Fit RF M1/M2/M3 forest-type representations and quantify the transferability gap.
5. Fit the global OLS reference and evaluate residual spatial autocorrelation.
6. Calibrate GWR/MGWR on the shared 12,000-point sample; optionally run complete-sample MGWR.
7. Translate local MGWR effects to EPA Level III ecoregion management zones.

Exact commands are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation and licenses

Until a journal/archival DOI is assigned, cite the repository as:

> Yin, Q. (2026). *Western U.S. Post-fire Forest Resilience: reproducibility package* (Version 0.1.0). GitHub. https://github.com/ZolaYin/western-us-postfire-resilience

Machine-readable metadata are provided in [`CITATION.cff`](CITATION.cff). Project code is released under the [MIT License](LICENSE). Project-created data tables and results are released under [CC BY 4.0](LICENSE-DATA.md), subject to the upstream attribution and license notes documented in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md). OpenStreetMap-derived fields retain applicable ODbL obligations.




