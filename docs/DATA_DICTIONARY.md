# Data dictionary

The canonical field-level dictionary is
[`data/processed/westernus_model_table_dictionary.csv`](../data/processed/westernus_model_table_dictionary.csv).
It contains one entry for each of the 83 columns in the released model table,
in table order, with these fields:

- `column`: exact Parquet column name;
- `dtype`: released physical data type;
- `unit`: measurement or encoded unit;
- `temporal_window`: reference year or aggregation window;
- `description`: plain-language meaning;
- `derivation`: transformation or aggregation rule;
- `source`: upstream product or project-derived source.

The physical schema is also available as
[`data/processed/westernus_model_table_schema.csv`](../data/processed/westernus_model_table_schema.csv),
and summary statistics and missingness are in
[`data/processed/westernus_model_table_summary.json`](../data/processed/westernus_model_table_summary.json).

## Naming conventions

- `_pre`, `_during`, and `_post` identify the pre-fire, fire, and post-fire windows defined in the CSV.
- `_near_t0` identifies the value nearest the fire year; `_t0agg` identifies the fire-year aggregation used by the released pipeline.
- `_z` is a standardized predictor used by the fitted models.
- `_log` is the documented log transform before standardization.
- `Resistance`, `IRI_good_pow2`, and `STAB_good_pow2` are the three modeled resilience responses; their exact equations and input fields are listed in the CSV.

## Important unit notes

- Climate temperature fields are degrees Celsius; precipitation and climatic water deficit fields are millimetres over the window specified for each row.
- `FS_CBH_t0agg` retains the source product's encoded 0.1 m unit; the dictionary records both the stored and physical interpretation.
- `TS_SOC_0_30cm` contains 15 inherited `-9999` source-stack nodata sentinels. They are documented rather than silently rewritten so that the released model inputs and results remain byte-reproducible.
- Road and trail density fields are kilometres per square kilometre within the radius encoded in the name.
- Coordinate fields `x` and `y` are metres in EPSG:5070.

## Reporting regions

`region` is a project-derived five-class reporting label. Its deterministic rule
is implemented in `src/preprocessing/assign_reporting_regions.py` and documented
in `docs/REGION_RULE.md`; it is not an external administrative boundary.

Blank values are preserved where an upstream observation or a valid temporal
window was unavailable. Modeling scripts apply response-specific eligibility
flags from the released split table rather than silently imputing those rows.
