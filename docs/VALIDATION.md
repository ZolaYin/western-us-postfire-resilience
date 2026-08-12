# Validation record

Validation date: 2026-08-11.

## Passed checks

- Final model table: 133,409 rows, 83 columns, unique `pixel_id`, EPSG:5070 projected coordinates, fire years 2005–2022.
- Final model-table SHA-256: `2675c4b1b24846bdc52f72b2f15f89cd086626cdfca6a654bf28bf6d9897ec7a`.
- The HPRC source table `westernus_current_candidate_table_plus_regions.parquet` has the same SHA-256, confirming byte-for-byte identity with the bundled table.
- Reporting-region labels regenerated for all 133,409 rows with zero mismatches.
- Deterministic partitions regenerated with seed 42, 20% target holdout, and 100 km spatial blocks; the rebuilt assignments match the released file exactly.
- All Python files in the canonical and archived package parse successfully.
- All 18 full RF fits were independently rerun. Relative to the released metrics, maximum differences were 0.0017 for R² and 0.0002 for RMSE, consistent with cross-platform parallel reduction. M2 > M3 > M1 and the random-versus-spatial transferability conclusions were unchanged.
- The shared MGWR sample and OLS residual diagnostics were rebuilt; published R², Moran's I, and p values matched to at least eight decimal places.
- The canonical MGWR calibration starts successfully. The full 12,000-point and complete-sample calibrations remain documented HPC jobs rather than laptop validation tasks.
- The final zoning was rebuilt from the newly published point table, three complete-sample coefficient tables, and exact EPA-derived analysis units. After matching rows by EPA code and feature coordinates, thresholds, sensitivity tables, summaries, numeric zone fields, and final q75 classifications matched the released outputs exactly.
- The two legacy non-q75 columns retained for provenance are not part of the final zoning definition. For Central Basin and Range, `mechanism_zone` and `management_zone` can switch between `Mixed-control transition zone` and `Structure-dominated resilience zone` because the unit lies at a floating-point decision boundary. This platform-sensitive label does not affect any numeric field or the final `mechanism_zone_q75` and `management_zone_q75` classifications.
- No personal local path, project email address, common private-key marker, GitHub personal-access-token pattern, or AWS access-key pattern was found in the text release files.
- No release file exceeds 90 MB.

## Scope note

The complete MGWR calibrations themselves were not repeated during this laptop-oriented release check; their retained coefficient tables were used to reproduce the downstream zoning. The repository documents expected resources and supplies Slurm entry points for those calibrations.
