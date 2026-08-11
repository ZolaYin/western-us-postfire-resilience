# Data dictionary

The machine-readable schema is `data/processed/westernus_model_table_schema.csv`. Column prefixes follow these conventions:

| Prefix or suffix | Meaning |
|---|---|
| `TS_` | Topography and soil |
| `FS_` | Forest structure/type |
| `HUM_` | Human footprint and access |
| `CLIM_` | Climate and water balance |
| `_pre` / `_post` | Fire-relative pre-fire or post-fire window |
| `_z` | Full-table z-score |
| `_log_z` | `log1p` transform followed by z-score |

## Identifiers and geometry

- `pixel_id`: stable row identifier in the released model table.
- `row`, `col`: reference-grid indices.
- `x`, `y`: EPSG:5070 pixel-center coordinates in metres.
- `lon_wgs84`, `lat_wgs84`: WGS84 coordinates for reporting.
- `region`: five deterministic geographic reporting strata. West of 118°W, pixels are `PNW` at or north of 44°N and `CA_med` to the south. At or east of 118°W, pixels are `N_Rockies` at or north of 44°N, `S_Rockies` from 37°N to 44°N, and `SW_dry` south of 37°N. The executable rule is in `src/preprocessing/assign_reporting_regions.py`.

## Fire and response variables

- `t0_year`: first retained fire year for the pixel.
- `sev`: MTBS-derived severity class/value at `t0`.
- `Bi`: five-year pre-fire RESI baseline.
- `Rmin`: minimum RESI at `t0` or `t0+1`.
- `Resistance`: resistance relative to the pre-fire baseline.
- `T50`, `T80`, `T90`, `T95`: years to reach the named fraction of the pre-fire baseline; paired `_reached` fields indicate whether the threshold was observed.
- `IRI_gap`, `IRI_good`, `IRI_good_pow2`: integrated recovery summaries.
- `STAB_CV`, `STAB_good`, `STAB_good_pow2`: post-fire temporal-stability summaries.
- `AUC_deficit`: post-fire cumulative deficit.

## Final manuscript responses

The main inferential chain retains `Resistance` as the primary response and `IRI_good_pow2` plus `STAB_good_pow2` as complementary response dimensions. T50/T80 are preserved in the table and historical model archive but are not part of the current main three-response claim chain.
