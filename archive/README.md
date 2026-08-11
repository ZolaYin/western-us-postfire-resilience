# Historical code archive

This directory preserves modeling code that informed the project but is not the recommended entry point for reproducing the retained analysis.

- `original_model_code/` contains the modeling snapshot covering RF, XGBoost, OLS, GWR/MGWR, SVC/GAM, latent-regime, spatial-transfer, multiscale/local RF, Soft-MoE, and response-sensitivity experiments.
- `hprc_code_snapshot/` is the read-only 2026-08-07 snapshot of Python, Slurm, predictor/configuration text, and Markdown files found in the confirmed HPRC project directories. Virtual and conda environments were excluded.

Absolute personal paths and the HPRC NetID were replaced with `/path/to/...`, `YOUR_EMAIL`, and `YOUR_NETID`. Files retain historical names and may require path or environment adaptation.

Use `src/` for the cleaned, documented pipeline. The archive remains here to make model selection and analysis history transparent; it is not required for a standard reproduction run.
