# Data sources and provenance

The released Parquet table contains project-derived 1 km attributes, not copies of the source rasters. Raw inputs can be obtained from the official access points below. The public [Google Drive data release](https://drive.google.com/drive/folders/1C1kPp0hS7RW5zTaVD0c7O88LxmNuJ3wk) mirrors the analysis-ready data and provides larger derived artifacts.

## Source inventory

| Data group | Product and identifier | Official access | Project use | Reuse/attribution note |
|---|---|---|---|---|
| Fire history and severity | Monitoring Trends in Burn Severity (MTBS) | [MTBS Direct Download](https://www.mtbs.gov/direct-download) | Fire year `t0`, severity at `t0`, and burned-pixel mask aligned to the 1 km grid | U.S. federal data; credit MTBS/USGS/USFS and cite the downloaded product |
| Annual vegetation response | Project-derived annual RESI, 2000–2023 | Landsat [LT05](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2), [LE07](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2), [LC08](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2), and [LC09](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2) Collection 2 Tier 1 Level-2 | Pre-fire baseline, Resistance, integrated recovery, stability, and recovery-time metrics | Landsat data are U.S. public-domain data; cite USGS. The project-derived annual layer is reproducible from the released GEE code |
| Export boundary | TIGER/2018/States | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/TIGER_2018_States) | Dissolved boundary of the 11 western states | U.S. Census Bureau data |
| Existing vegetation type | LANDFIRE EVT 2022 | [LANDFIRE Data Distribution](https://landfire.gov/data) | Forest mask/type and broad EVT groups | U.S. federal product; cite the LANDFIRE version |
| Canopy base height | LANDFIRE CBH for 2008, 2010, 2012, 2014, 2016, and 2022 | [LANDFIRE Data Distribution](https://landfire.gov/data) | Nearest-year canopy structure, with the earlier year winning ties | U.S. federal product; cite the LANDFIRE versions |
| Tree canopy cover | USFS Tree Canopy Cover v2023-5; GEE collection `USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5`; band `NLCD_Percent_Tree_Canopy_Cover` | [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/USGS_NLCD_RELEASES_2023_REL_TCC_v2023-5) and [MRLC TCC page](https://www.mrlc.gov/data/type/nlcd-tree-canopy-cover) | Annual 30 m TCC matched to fire year and aggregated to the 1 km grid | Collected with U.S. Government funding; use is allowed without additional permission or fees; cite USDA Forest Service v2023.5 |
| Climate and water balance | gridMET, 2000–2023 | [Climatology Lab gridMET](https://www.climatologylab.org/gridmet.html) | Pre/post-fire precipitation, temperature, VPD, ETo, aridity, hot days, and variability | Cite gridMET and the variables/years used; raw NetCDF files are not redistributed here |
| Elevation | USGS 3D Elevation Program; retained aligned stack `TOPO_STATIC_WesternUS11_1km_5070_v2.tif` | [The National Map data delivery](https://www.usgs.gov/the-national-map-data-delivery) | Elevation, slope, northness/eastness, TWI, and roughness on the EPSG:5070 1 km grid | U.S. federal data; cite USGS 3DEP. The original DEM resolution, tile IDs, and download date were not retained in the historical metadata |
| Soil | SoilGrids 250 m v2, 0–30 cm soil organic carbon | [ISRIC SoilGrids documentation](https://docs.isric.org/globaldata/soilgrids/) | 0–30 cm soil organic carbon aggregated to the 1 km grid | [CC BY 4.0](https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_04.html); cite SoilGrids 2.0 |
| Land cover and imperviousness | NLCD 2019 release forest masks and an eight-epoch impervious stack | [MRLC data access](https://www.mrlc.gov/data) and GEE collection `USGS/NLCD_RELEASES/2019_REL/NLCD` | Forest classes 41/42/43 for 2006, 2011, 2016, and 2019; nearest-`t0` imperviousness for 2001, 2004, 2006, 2008, 2011, 2013, 2016, and 2019 | U.S. federal products; cite the NLCD 2019 release. The exact upstream collection used to build the retained impervious stack was not preserved |
| Nighttime lights | EOG VIIRS annual nighttime lights; retained aligned stack `HUMAN_VIIRS_AnnualMean_WesternUS11_1km_5070_v1.tif` | [EOG VIIRS Nighttime Lights](https://eogdata.mines.edu/products/vnl/) | Annual-mean bands for 2013–2024, with the nearest available year matched to `t0` | EOG lists VNL among products available under CC BY 4.0. The exact VNL product generation/version and download date were not retained, so the original upstream pixels cannot be reconstructed byte-for-byte from this repository alone |
| Population | Gridded Population of the World v4 | [NASA SEDAC GPWv4](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4) | Near-`t0` or windowed population-pressure predictor | Cite GPWv4 and follow the SEDAC data-use conditions; raw GPW grids are not redistributed here |
| Roads and trails | OSM-derived state/regional shapefile snapshots dated 2026-04-05; retained names follow `*-260405-free.shp/gis_osm_roads_free_1.shp` | [OpenStreetMap copyright and download information](https://www.openstreetmap.org/copyright); [Geofabrik U.S. downloads](https://download.geofabrik.de/north-america/us.html) are a current retrieval route | Exact line length followed by 5 km road and 10 km trail moving-window density | © OpenStreetMap contributors; source database is licensed under [ODbL 1.0](https://www.openstreetmap.org/copyright). Historical metadata did not preserve the download host, so Geofabrik is not asserted as the exact archived provider |
| Management boundaries | EPA Level III ecoregions | [EPA Level III/IV ecoregions](https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states) | Aggregation and final management-zone boundaries | U.S. federal data; cite EPA and the downloaded boundary release |

## Annual RESI derivation

Annual RESI is generated by [`../src/preprocessing/export_annual_resi_gee.js`](../src/preprocessing/export_annual_resi_gee.js), rather than downloaded as an independent product:

1. merge Landsat 5/7/8/9 Collection 2 Tier 1 Level-2 scenes;
2. mask cloud shadow, snow, cloud, and cirrus with `QA_PIXEL`;
3. apply the Collection 2 surface-reflectance scale and offset;
4. form a May–September median composite for each year from 2000 through 2023;
5. compute `NDVI` and `NDMI`, followed by `RESI = 0.5 × (NDVI + NDMI)`;
6. scale `[-0.2, 0.8]` to `[0, 1]`, clamp, multiply by 10,000, round, and export as UInt16;
7. export in EPSG:5070 at 1,000 m over the 11-state boundary.

## Tree canopy cover derivation

[`../src/preprocessing/export_annual_tcc_gee.js`](../src/preprocessing/export_annual_tcc_gee.js) records the exact source collection and band used for the annual exports:

- collection: `USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5`;
- band: `NLCD_Percent_Tree_Canopy_Cover`;
- available source years: 1985–2023;
- project export years: 2000–2023;
- native resolution: 30 m;
- project analysis value: exact fire year `t0`, aggregated to the 1 km grid.

## Recovered source-snapshot details and remaining limits

The release records all identifiers that can be recovered from the retained code,
file names, band descriptions, and workflow audits. They establish the following
details beyond the provider-level links:

- the legacy NLCD forest mask came from GEE collection
  `USGS/NLCD_RELEASES/2019_REL/NLCD`, using years 2006, 2011, 2016, and 2019,
  forest classes 41/42/43, and nearest-neighbor sampling to the 1 km grid;
- the aligned impervious stack contains 2001, 2004, 2006, 2008, 2011, 2013,
  2016, and 2019 bands, and the candidate-table builder selects the nearest
  available year to each pixel's `t0`;
- the aligned VIIRS stack contains 12 annual-mean bands for 2013–2024 in
  EPSG:5070 at 1 km, with nearest-year matching to `t0`;
- the aligned topographic stack is EPSG:5070 at 1 km and contains elevation
  (m), slope (degrees), aspect (degrees), northness, eastness, TWI, and
  roughness;
- the OSM-derived input set consists of 12 state/regional snapshots (Arizona,
  Colorado, Idaho, Montana, Nevada, New Mexico, northern California, Oregon,
  southern California, Utah, Washington, and Wyoming) dated 2026-04-05.

Four byte-level upstream details could not be recovered from the historical
metadata: the 3DEP DEM resolution/tile inventory, the exact upstream NLCD
collection used for the retained impervious stack, the exact EOG VNL product
generation used for the VIIRS stack, and the download host for the OSM-derived
shapefiles. These are documented limits rather than silently guessed versions.
They do not prevent exact reuse of the released analysis-ready table, whose
bytes and field-level derivations are fixed by the repository manifests and data
dictionary; they do prevent a claim that every third-party raw pixel can be
re-downloaded byte-for-byte.

## Harmonization rules

- Target CRS: NAD83 / CONUS Albers, EPSG:5070.
- Target resolution: 1,000 m.
- Dynamic inputs are anchored to fire year `t0`.
- Continuous rasters use mean aggregation or bilinear resampling as appropriate.
- Categorical rasters use nearest-neighbor or modal aggregation.
- CBH uses the nearest available year, with the earlier year winning ties.
- TCC uses the exact fire year for the retained 2005–2022 pixels.
- Road/trail density units are km of line per km² within circular neighborhoods.

## Distribution boundary

GitHub and Google Drive distribute the project-created model table, split assignments, code, and derived results. Source rasters remain with their official providers, which keeps downloads current and preserves source-specific attribution. The project-created compilation is licensed under [`../LICENSE-DATA.md`](../LICENSE-DATA.md); upstream terms remain applicable to source-derived fields, especially the OpenStreetMap-derived road and trail densities.
