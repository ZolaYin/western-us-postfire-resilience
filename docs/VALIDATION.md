# Validation record

Validation date: 2026-08-13.

## Passed checks

- Final model table: 133,409 rows, 83 columns, unique `pixel_id`, EPSG:5070 projected coordinates, fire years 2005–2022.
- Final model-table SHA-256: `2675c4b1b24846bdc52f72b2f15f89cd086626cdfca6a654bf28bf6d9897ec7a`.
- The HPRC source table `westernus_current_candidate_table_plus_regions.parquet` has the same SHA-256, confirming byte-for-byte identity with the bundled table.
- Reporting-region labels regenerated for all 133,409 rows with zero mismatches.
- Deterministic partitions regenerated with seed 42, 20% target holdout, and 100 km spatial blocks; the rebuilt assignments match the released file exactly.
- All Python files in the canonical and archived package parse successfully.
- All 18 full RF fits were independently rerun. Relative to the released metrics, maximum differences were 0.0017 for R² and 0.0002 for RMSE, consistent with cross-platform parallel reduction. M2 > M3 > M1 and the random-versus-spatial transferability conclusions were unchanged.
- All 18 regularized RF sensitivity fits (`max_depth=10`, `min_samples_leaf=20`) were rerun with the released assignments after enforcing unique predictor columns. The 100 km block-validation R² advantage of M2 over M3 was 0.0323 for Resistance, 0.0673 for IRI, and 0.0989 for STAB, reproducing the manuscript's rounded values (0.032, 0.067, and 0.099).
- The RF model-matrix builder was checked for duplicate names and now enforces one column per declared predictor. The former work-table selection repeated `x` and `y` even though both already belonged to the 27-column baseline list. After de-duplication, the 100 km RF-base block R² differed from the released reference by no more than 0.000028 across the three responses, well inside the previously observed cross-platform RF tolerance; substantive rankings and conclusions were unchanged.
- The final 27-predictor RF-base block-size sensitivity was rerun end to end (12 fits; 12.5 min on a 10-core laptop). The random-minus-block R² gap remained positive for every response at 50, 100, and 200 km: Resistance 0.3458/0.4533/0.5356; IRI 0.4299/0.3691/0.3872; STAB 0.3544/0.4130/0.3989. The complete metrics and run metadata are retained in `results/rf/`.
- The shared MGWR sample and OLS residual diagnostics were rebuilt; published R², Moran's I, and p values matched to at least eight decimal places.
- The canonical MGWR calibration starts successfully. The full 12,000-point and complete-sample calibrations remain documented HPC jobs rather than laptop validation tasks.
- The final zoning was rebuilt from the newly published point table, three complete-sample coefficient tables, and exact EPA-derived analysis units. After matching rows by EPA code and feature coordinates, thresholds, sensitivity tables, summaries, numeric zone fields, and final q75 classifications matched the released outputs exactly.
- The two legacy non-q75 columns retained for provenance are not part of the final zoning definition. For Central Basin and Range, `mechanism_zone` and `management_zone` can switch between `Mixed-control transition zone` and `Structure-dominated resilience zone` because the unit lies at a floating-point decision boundary. This platform-sensitive label does not affect any numeric field or the final `mechanism_zone_q75` and `management_zone_q75` classifications.
- No personal local path, project email address, common private-key marker, GitHub personal-access-token pattern, or AWS access-key pattern was found in the text release files.
- No release file exceeds 90 MB.

## Scope note

The complete MGWR calibrations themselves were not repeated during this laptop-oriented release check; their retained coefficient tables were used to reproduce the downstream zoning. The repository documents expected resources and supplies Slurm entry points for those calibrations.
