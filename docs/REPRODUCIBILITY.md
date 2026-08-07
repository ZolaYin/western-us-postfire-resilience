# Reproducibility guide

Run commands from the repository root.

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate western-us-postfire-resilience
```

## 2. Verify the bundled table

```bash
python src/preprocessing/describe_model_table.py \
  --input data/processed/westernus_model_table.parquet \
  --schema-output data/processed/westernus_model_table_schema.csv
```

Compare its SHA-256 checksum with `provenance/file_manifest.csv`.

## 3. Rebuild deterministic partitions

```bash
python src/splits/build_split_assignments.py \
  --input data/processed/westernus_model_table.parquet \
  --output data/splits/westernus_split_assignments.parquet \
  --block-km 100 --test-size 0.2 --random-state 42
```

The saved table contains a global 100 km block ID and response-specific eligibility, random split, and block split columns. Split labels are joined by `pixel_id`; row order is never used as an external key.

## 4. Run the RF transferability comparison

```bash
python src/models/rf/run_foresttype_comparison.py \
  --input data/processed/westernus_model_table.parquet \
  --splits data/splits/westernus_split_assignments.parquet \
  --output-dir output/rf_foresttype \
  --responses Resistance IRI_good_pow2 STAB_good_pow2 \
  --trees 300 --random-state 42
```

## 5. Prepare the shared MGWR sample

```bash
python src/preprocessing/prepare_mgwr_samples.py \
  --input data/processed/westernus_model_table.parquet \
  --output-dir output/mgwr_samples \
  --sample-n 12000 --random-state 42
```

## 6. Run OLS residual spatial diagnostics

```bash
python src/models/ols/run_pre_mgwr_diagnostics.py \
  --input output/mgwr_samples/sample_n12000_seed42.parquet \
  --predictors-file config/mgwr_predictors.txt \
  --response Resistance \
  --output-dir output/ols_diagnostics
```

## 7. Run MGWR

```bash
python src/models/mgwr/run_mgwr.py \
  --input output/mgwr_samples/sample_n12000_seed42.parquet \
  --response Resistance \
  --predictors-file config/mgwr_predictors.txt \
  --output-dir output/mgwr_resistance
```

The complete-sample alternative is `src/models/mgwr/run_complete_sample_mgwr.py`; it requires substantially more compute and is intended for Slurm/HPRC.

## 8. Build management zones

The final zoning script expects the multiresponse MGWR effect table joined to EPA Level III geometries. The retained local coefficients and zoning summaries are included under `results/`. Run the multiresponse join first, then the constraint framework:

```bash
python src/zoning/compute_multiresponse_zoning.py
python src/zoning/build_constraint_zones.py
```

These two scripts currently use repository-relative input/output conventions preserved from the final analysis. Their inputs and expected filenames are documented at the top of each file.

## Randomness and comparability

- Main seed: 42.
- RF estimators: 300 for the M1/M2/M3 comparison.
- Random validation test fraction: 0.2.
- Spatial validation: 100 km projected blocks, held out with `GroupShuffleSplit` at a 0.2 target fraction.
- Do not compare results generated from newly filtered rows to published values unless split eligibility and assignments match the saved partition table.
