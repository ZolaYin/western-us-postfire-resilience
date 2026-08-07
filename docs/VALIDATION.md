# Validation record

Validation date: 2026-08-07.

## Passed checks

- Final model table: 133,409 rows, 83 columns, unique `pixel_id`, EPSG:5070 projected coordinates, fire years 2005–2022.
- Final model-table SHA-256: `2675c4b1b24846bdc52f72b2f15f89cd086626cdfca6a654bf28bf6d9897ec7a`.
- The HPRC source table `westernus_current_candidate_table_plus_regions.parquet` has the same SHA-256, confirming byte-for-byte identity with the bundled table.
- Deterministic partitions regenerated with seed 42, 20% target holdout, and 100 km spatial blocks; regenerated counts match the saved split summary.
- All 199 Python files in the canonical and archived package parse successfully.
- A 1,000-row, two-tree RF smoke test completed for M1/M2/M3 under random and block splits.
- A 250-row shared MGWR sample was generated and passed through the OLS spatial-diagnostic entry point.
- The canonical MGWR entry point imports successfully and exposes its command-line interface.
- No personal local path, project email address, common private-key marker, GitHub personal-access-token pattern, or AWS access-key pattern was found in the text release files.
- No release file exceeds 90 MB.


## Deliberately not claimed

The full 300-tree RF matrix, complete 12,000-point MGWR calibration, complete-sample MGWR, and zoning rebuild were not rerun during packaging. Those compute-intensive release checks remain required before publication, along with comparison against the retained outputs.
