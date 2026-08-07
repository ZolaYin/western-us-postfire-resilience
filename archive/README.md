# Historical code archive

This directory preserves modeling code that informed the project but is not the recommended publication entry point.

- `original_model_code/` contains the local snapshot selected by imports from scikit-learn, XGBoost, statsmodels, PyGAM, GWR, or MGWR. It covers RF, XGBoost, OLS, GWR/MGWR, SVC/GAM, latent-regime, spatial-transfer, multiscale/local RF, Soft-MoE, and response-sensitivity experiments.
- `hprc_code_snapshot/` is the read-only 2026-08-07 snapshot of Python, Slurm, predictor/configuration text, and Markdown files found in the confirmed HPRC project directories. Virtual/conda environments were excluded.

Absolute personal paths and the HPRC NetID were replaced with `/path/to/...`, `YOUR_EMAIL`, and `YOUR_NETID`. This archive records the analysis history; many files retain legacy filename conventions and are not guaranteed to run without adaptation.

Use `src/` for the cleaned, documented workflow. Before public release, decide whether journal transparency benefits from retaining this archive or whether it should instead be attached to a separate provenance release.
