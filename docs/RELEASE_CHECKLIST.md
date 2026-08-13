# Maintainer release checklist

This checklist is for versioned repository releases.

- [x] Include the final analysis table, schema, deterministic splits, model code, zoning code, and retained results.
- [x] Record authoritative source links and the exact annual RESI and TCC Earth Engine collections.
- [x] Record every recoverable raw-snapshot detail and explicitly document the four unresolved historical identifiers instead of guessing versions.
- [x] Document and provide code for the five reporting-region labels.
- [x] Add separate code and derived-data licenses.
- [x] Add `CITATION.cff` and a repository citation.
- [x] Publish a read-only Google Drive data-release folder for downloadable and larger artifacts.
- [x] Regenerate `provenance/file_manifest.csv` for this release and verify SHA-256 values.
- [x] Run the retained laptop workflow, start the MGWR entry point, and compare regenerated RF, OLS, and zoning outputs with `results/`.
- [ ] Replace the repository citation with the final manuscript citation and DOI when they become available.
- [ ] Create a version tag and DOI-backed archive for the journal submission package.

The last two items depend on final manuscript metadata and the journal-release archive. They do not prevent collaborators from using or sharing the current repository.
