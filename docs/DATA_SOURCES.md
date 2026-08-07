# Data sources and provenance

Raw rasters are not committed. The links below are the authoritative access points for the upstream products. Product versions, acquisition dates, local filenames, checksums, and processing scripts must be frozen in `provenance/raw_input_manifest.csv` before public release.

| Data group | Product used in the project | Authoritative access | Project use | Release status |
|---|---|---|---|---|
| Fire history and severity | Monitoring Trends in Burn Severity (MTBS) | https://www.mtbs.gov/direct-download | Fire year `t0`, severity at `t0`, burned-pixel mask; aligned to the 1 km reference grid | Link verified; exact source bundle/checksum still needed |
| Annual vegetation response | Annual RESI, 2000–2023 | **Public archive/DOI still required** | Pre-fire baseline, Resistance, IRI, stability, and recovery-time metrics | Blocking item for public release |
| Existing vegetation type and canopy base height | LANDFIRE EVT and CBH, including 2008/2010/2012/2014/2016/2022 CBH inputs | https://landfire.gov/data | Forest mask/type, broad EVT groups, near-`t0` CBH | Link verified; freeze exact LANDFIRE versions |
| Tree canopy cover | Annual 30 m GEE exports, matched to `t0` | **Exact GEE collection/asset ID still required** | `FS_TCC_t0` | Blocking provenance item |
| Climate and water balance | gridMET | https://www.climatologylab.org/gridmet.html | Pre/post-fire precipitation, temperature, VPD, ETo, aridity, hot days, and variability | Link verified; freeze NetCDF names/checksums |
| Elevation | USGS 3D Elevation Program / The National Map | https://www.usgs.gov/the-national-map-data-delivery | Elevation; slope and other terrain derivatives | Link verified |
| Soil | SoilGrids 250 m v2 | https://docs.isric.org/globaldata/soilgrids/ | 0–30 cm soil organic carbon | Link verified; document layer/depth statistic |
| Imperviousness / forest mask | NLCD / Annual NLCD through MRLC | https://www.mrlc.gov/viewer/ | Near-`t0` impervious surface and legacy NLCD forest mask | Link verified; freeze collection/version |
| Nighttime lights | VIIRS annual nighttime lights, Earth Observation Group | https://eogdata.mines.edu/products/vnl/ | Near-`t0` nighttime light | Link verified; freeze VNL version and band |
| Population | GPWv4 population density | https://sedac.ciesin.columbia.edu/data/collection/gpw-v4 | Near-`t0`/windowed population pressure | Confirm exact release and redistribution terms |
| Roads and trails | OpenStreetMap-derived line data, downloaded through Geofabrik | https://download.geofabrik.de/north-america/us.html | Exact line length followed by 5 km road and 10 km trail moving-window density | Document download dates and ODbL attribution |
| Management boundaries | EPA Level III ecoregions | https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states | Aggregation and final management-zone boundaries | Link verified; freeze shapefile checksum |

## Harmonization rules

- Target CRS: NAD83 / CONUS Albers, EPSG:5070.
- Target resolution: 1,000 m.
- Dynamic inputs are anchored to fire year `t0`.
- Continuous rasters use mean aggregation or bilinear resampling as appropriate.
- Categorical rasters use nearest-neighbor/mode logic.
- CBH uses nearest available year, with earlier year winning ties.
- TCC uses the exact fire year for the retained 2005–2022 pixels.
- Road/trail density units are km of line per km² within circular neighborhoods.

## Redistribution review required

The presence of a public download link does not automatically authorize repackaging the raw raster. Before release, record the license/terms for each product and decide whether the repository should contain (a) only links and scripts, (b) derived variables, or (c) selected source rasters. The current draft contains only the derived model table.
