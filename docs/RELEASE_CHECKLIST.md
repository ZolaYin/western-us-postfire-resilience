# Release checklist

- [ ] Confirm repository/project title and GitHub repository name.
- [ ] Confirm final author list and add `CITATION.cff`.
- [ ] Choose code license.
- [ ] Choose derived-data license and verify upstream redistribution terms.
- [ ] Add annual RESI archive/DOI or controlled-access instructions.
- [ ] Record exact TCC GEE collection/asset ID.
- [ ] Recover/document the five-region assignment rule.
- [ ] Freeze exact raw product versions, acquisition dates, filenames, and checksums.
- [x] Replace personal local/HPRC paths in the release snapshot with portable placeholders.
- [x] Scan for secrets, tokens, email addresses, and private filesystem information.
- [ ] Re-run final RF/OLS/MGWR/zoning workflow from the release directory.
- [ ] Compare regenerated metrics with `results/`.
- [ ] Create a tagged release and independent DOI archive.
- [ ] Verify the independent archive before considering HPRC cleanup.
- [ ] Obtain explicit approval for the exact HPRC deletion manifest.
