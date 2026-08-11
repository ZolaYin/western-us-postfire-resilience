# Legacy preprocessing snapshot

> **Archive status:** these scripts are provenance-only snapshots. They are not invoked by the numbered workflow in `docs/REPRODUCIBILITY.md`, are not required to reproduce the released table or results, and intentionally retain `/path/to/google-drive/...` placeholders in place of former machine-specific paths.

These files are the exact project preprocessing scripts, with personal absolute paths replaced by placeholders. They document the established processing logic for:

- annual RESI/MTBS response reconstruction;
- the base candidate table;
- LANDFIRE CBH aggregation and near-fire-year matching;
- annual TCC sampling;
- OpenStreetMap-derived road/trail density;
- appending corrected access-pressure variables; and
- the GEE RESI export.

They are retained for provenance because a full end-to-end rebuild depends on large raw rasters and source-specific folder layouts. Convert placeholder paths to command-line arguments before treating these files as reusable software. The cleaned downstream workflow in the parent directories has no personal absolute paths.
