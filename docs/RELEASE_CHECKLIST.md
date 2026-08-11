# Maintainer release checklist

This checklist is for versioned repository releases.

- [x] Include the final analysis table, schema, deterministic splits, model code, zoning code, and retained results.
- [x] Record authoritative source links and the exact annual RESI and TCC Earth Engine collections.
- [x] Document and provide code for the five reporting-region labels.
- [x] Add separate code and derived-data licenses.
- [x] Add `CITATION.cff` and a repository citation.
- [x] Publish a read-only Google Drive data-release folder for downloadable and larger artifacts.

- [ ] For each future version, regenerate `provenance/file_manifest.csv` and verify SHA-256 values.
- [ ] Run the retained workflow and compare regenerated metrics with `results/`.
- [ ] Replace the repository citation with the final manuscript citation and DOI when they become available.
- [ ] Create a version tag and DOI-backed archive for the journal submission package.

The last four items are normal maintenance steps for a future tagged journal release; they do not prevent collaborators from using or sharing the current repository.
