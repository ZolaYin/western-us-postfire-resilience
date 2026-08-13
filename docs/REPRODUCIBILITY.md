# Reproducibility guide

Run commands from the repository root. Runtime and memory estimates below are approximate and are meant to distinguish ordinary laptop tasks from long-running jobs.

## Resource guide

| Step | Task | Expected runtime | Approximate peak memory | Reference hardware / note |
|---|---|---:|---:|---|
| 1 | Create Conda environment | 5–20 min | 2–4 GB | Network and package-cache dependent |
| 2 | Earth Engine exports | Provider dependent | Browser task | Exports run asynchronously in Earth Engine |
| 3 | Table check and region validation | <2 min | <2 GB | Modern laptop |
| 4 | Deterministic split rebuild | <2 min | <2 GB | Modern laptop |
| 5 | 18 RF fits with 300 trees | about 10–20 min | 4–8 GB | Observed on a 10-core laptop; the script prints progress for every fit |
| 5a | RF block-size sensitivity (12 fits) | about 10–20 min | 4–8 GB | Three responses; 50, 100, and 200 km blocks; progress is printed for every fit |
| 6 | Build the 12,000-row MGWR sample | <1 min | <2 GB | Modern laptop |
| 7 | OLS and residual spatial diagnostics | about 2–10 min | 4–8 GB | Modern laptop |
| 8 | 12,000-point MGWR calibration | hours | 16–32 GB | Prefer 16 CPU cores; use `hpc/submit_mgwr.sbatch` |
| 8a | Complete-sample MGWR | days | 128–192 GB | HPC only; Slurm templates request up to 21 days |
| 9 | Multiresponse zoning and constraint maps | about 1–5 min | 4–8 GB | Modern laptop; observed under 1 min on a 10-core laptop |

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate western-us-postfire-resilience
```

## 2. Rebuild annual remote-sensing inputs when needed

The exact Google Earth Engine exports used for annual RESI and tree canopy cover are recorded in:

- `src/preprocessing/export_annual_resi_gee.js`
- `src/preprocessing/export_annual_tcc_gee.js`

Paste either script into the Earth Engine Code Editor, adjust only the destination Drive folder if needed, and start the generated export tasks. Official collection links and derivations are documented in `docs/DATA_SOURCES.md`.

## 3. Verify the bundled table and reporting regions

```bash
python src/preprocessing/describe_model_table.py \
  --input data/processed/westernus_model_table.parquet \
  --schema-output output/westernus_model_table_schema.csv
```

Compare its SHA-256 checksum with `provenance/file_manifest.csv`. Writing the generated schema under `output/` avoids modifying a tracked release file during a check.

```bash
python src/preprocessing/assign_reporting_regions.py \
  data/processed/westernus_model_table.parquet
```

The released table should report `mismatches=0` for all 133,409 rows.

## 4. Rebuild deterministic partitions

```bash
mkdir -p output/splits
python src/splits/build_split_assignments.py \
  --input data/processed/westernus_model_table.parquet \
  --output output/splits/westernus_split_assignments.parquet \
  --block-km 100 --test-size 0.2 --random-state 42
```

Compare the rebuilt file with `data/splits/westernus_split_assignments.parquet`. The saved table contains a global 100 km block ID and response-specific eligibility, random split, and block split columns. Split labels are joined by `pixel_id`; row order is never used as an external key.

## 5. Run the RF transferability comparison

```bash
python src/models/rf/run_foresttype_comparison.py \
  --input data/processed/westernus_model_table.parquet \
  --splits data/splits/westernus_split_assignments.parquet \
  --output-dir output/rf_foresttype \
  --responses Resistance IRI_good_pow2 STAB_good_pow2 \
  --trees 300 --random-state 42
```

The command runs 18 fits: three responses × three forest-type representations × two validation schemes. Progress, elapsed time, R², and RMSE are printed after each fit. Small cross-platform differences are expected from parallel random-forest reduction; the independent release check found maximum differences of 0.0017 for R² and 0.0002 for RMSE while preserving all model rankings and transferability conclusions.

To reproduce the regularized RF sensitivity check reported in the manuscript,
reuse the same saved random and 100 km block assignments and set the two tree
constraints explicitly:

```bash
python src/models/rf/run_foresttype_comparison.py \
  --input data/processed/westernus_model_table.parquet \
  --splits data/splits/westernus_split_assignments.parquet \
  --output-dir output/rf_foresttype_regularized \
  --responses Resistance IRI_good_pow2 STAB_good_pow2 \
  --trees 300 --random-state 42 \
  --max-depth 10 --min-samples-leaf 20
```

The defaults (`--max-depth None` internally and `--min-samples-leaf 1`) retain
the primary unconstrained-RF behavior. The selected values are written to
`run_metadata.json` for every run.

To reproduce Supplementary Table S10 with the final 27-predictor RF-base
specification, run the dedicated block-size sensitivity entry point:

```bash
python src/models/rf/run_block_size_sensitivity.py \
  --input data/processed/westernus_model_table.parquet \
  --splits data/splits/westernus_split_assignments.parquet \
  --output-dir output/rf_block_size_sensitivity \
  --responses Resistance IRI_good_pow2 STAB_good_pow2 \
  --block-km 50 100 200 \
  --trees 300 --random-state 42
```

The random 80/20 split is fitted once per response and reused across block
sizes. The released 100 km labels are reused exactly; 50 and 200 km block
labels are generated deterministically by flooring EPSG:5070 coordinates by
the requested block width and applying `GroupShuffleSplit`. The script writes
the performance table, row and block counts, split provenance, predictor list,
RF settings, and runtime metadata. The released reference output is
`results/rf/rf_block_size_sensitivity.csv`.

## 6. Prepare the shared MGWR sample

```bash
python src/preprocessing/prepare_mgwr_samples.py \
  --input data/processed/westernus_model_table.parquet \
  --output-dir output/mgwr_samples \
  --sample-n 12000 --random-state 42
```

## 7. Run OLS residual spatial diagnostics

```bash
python src/models/ols/run_pre_mgwr_diagnostics.py \
  --input output/mgwr_samples/sample_n12000_seed42.parquet \
  --predictors-file config/mgwr_predictors.txt \
  --response Resistance \
  --output-dir output/ols_diagnostics
```

## 8. Run MGWR

```bash
python src/models/mgwr/run_mgwr.py \
  --input output/mgwr_samples/sample_n12000_seed42.parquet \
  --response Resistance \
  --predictors-file config/mgwr_predictors.txt \
  --output-dir output/mgwr_resistance
```

This is a long-running calibration even at 12,000 points. The complete-sample alternative is `src/models/mgwr/run_complete_sample_mgwr.py`; it is not a laptop task. Use the Slurm templates in `hpc/`, especially `submit_mgwr.sbatch`, `submit_full_fixed_bandwidth_mgwr.sbatch`, and `submit_complete_sample_mgwr_xlong.sbatch`.

## 9. Build management zones

The final zoning uses five downloadable inputs from the public Drive release:

1. [`spatialized_point_zones_full_candidate.parquet`](https://drive.google.com/file/d/1Drj3YzEIK_0CY-4EyGaashdeosp2rt99/view) (SHA-256 `7128b2c97fa2dee33e00cf20d768c4776c3c60db1e0a13d0dab3f5780bbcebd2`);
2. [`Resistance_mgwr_complete_coefficients.parquet`](https://drive.google.com/file/d/1r87WJvx7ufoo5aIbmpB7AzfTUmA6CH9w/view) (SHA-256 `301c05a2e408b36279ed6a664e2ba847c8519439b4600868dd177e56077decb7`);
3. [`IRI_good_pow2_mgwr_complete_coefficients.parquet`](https://drive.google.com/file/d/1QJXVhXZRu99sYZtijmjyF5NvIBmTU616/view) (SHA-256 `e9c027ce37205ae732be2864a4d1a30a34429a75fa180600060c0b758f174bf1`);
4. [`STAB_good_pow2_mgwr_complete_coefficients.parquet`](https://drive.google.com/file/d/14iQ5FASFuP7lwhUFi1expS9ZgAGIgkUo/view) (SHA-256 `39f4aef3ded18b9e1caf265c2d7eb9ce374e8f2e76380b03045b622a431ed36c`);
5. [`westernus_epa_l3_zoning_analysis_units.gpkg`](https://drive.google.com/file/d/1vze_NF-Oa8IaKI26510Zp0nSGs4FqA17/view) (SHA-256 `8d19b125e687757a2215054a7f0cebf22e11da115ccc19b1765a1600a8091bf9`).

Place the four Parquet files and the GeoPackage in one directory, for example `external/zoning/`. The coefficient file names above are recognized directly by the script. Then run:

```bash
python src/zoning/compute_multiresponse_zoning.py \
  --points external/zoning/spatialized_point_zones_full_candidate.parquet \
  --eco-l3 external/zoning/westernus_epa_l3_zoning_analysis_units.gpkg \
  --coef-dir external/zoning \
  --ecoregion-unit source-feature \
  --unmatched-policy nearest \
  --output-dir output/zoning_multiresponse

python src/zoning/build_constraint_zones.py \
  --input-gpkg output/zoning_multiresponse/epa_l3_multiresponse_management_zones.gpkg \
  --output-dir output/zoning_constraint
```

All paths are command-line arguments; the scripts contain no personal or Google Drive absolute paths. The first command produces the exact GeoPackage consumed by the second. In release validation, the regenerated thresholds, sensitivity tables, summaries, numeric zone fields, and final q75 classifications matched `results/zoning/` exactly after matching rows by EPA code and feature coordinates.

The analysis-unit GeoPackage is derived from the [official EPA Level III ecoregion layer](https://www.epa.gov/eco-research/level-iii-and-iv-ecoregions-continental-united-states). It records the exact 86 geometry units used for the published aggregation: EPA codes are dissolved except for the two disjoint source features carrying code 23, which remain separate. This removes ambiguity caused by later dissolving the source layer differently. For a new analysis rather than exact release reproduction, the official EPA shapefile can be supplied directly with either `--ecoregion-unit source-feature` or `--ecoregion-unit dissolved-code`.

The smaller 12,000-point coefficient tables under `results/mgwr/` remain useful for inspecting the sampled MGWR analysis, but they are not the coefficient inputs used to create the final complete-sample zoning release.

The optional `--resistance-only-reference` argument to `compute_multiresponse_zoning.py` adds historical comparison tables when the earlier resistance-only CSV is available. It is not required for the final multiresponse or constraint zoning outputs.

## Randomness and comparability

- Main seed: 42.
- RF estimators: 300 for the M1/M2/M3 comparison.
- RF estimators: 300 for the block-size and regularization sensitivity checks.
- Random validation test fraction: 0.2.
- Spatial validation: 100 km projected blocks, held out with `GroupShuffleSplit` at a 0.2 target fraction.
- Do not compare results generated from newly filtered rows to published values unless split eligibility and assignments match the saved partition table.
