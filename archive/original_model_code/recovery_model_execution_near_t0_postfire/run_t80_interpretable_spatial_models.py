from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_t80_iri_stability_near_t0_postfire.parquet"
PREDICTORS_PATH = BASE_DIR / "predictors_ols_recovery_baseline.txt"
OUT_DIR = BASE_DIR / "t80_interpretable_spatial_models_2026-03-31"
SPATIAL_TERMS = ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]
CLASSES = list(range(2, 11))


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def add_spatial_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x_sq"] = out["x"] ** 2
    out["y_sq"] = out["y"] ** 2
    out["xy"] = out["x"] * out["y"]
    for col in ["x_sq", "y_sq", "xy"]:
        std = out[col].std(ddof=0)
        mean = out[col].mean()
        out[f"{col}_z"] = 0.0 if std == 0 else (out[col] - mean) / std
    return out


def extract_linear_terms(model, predictors: list[str]) -> pd.DataFrame:
    rows = []
    for term in predictors:
        rows.append(
            {
                "term": term,
                "coef": float(model.params[term]),
                "pvalue": float(model.pvalues[term]),
                "stat_abs": float(abs(model.tvalues[term])),
                "direction": "positive" if model.params[term] > 0 else "negative",
                "is_spatial_term": term in SPATIAL_TERMS,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_spatial_term", "stat_abs"], ascending=[False, False])


def extract_ordered_terms(result, predictors: list[str]) -> pd.DataFrame:
    rows = []
    for term in predictors:
        rows.append(
            {
                "term": term,
                "coef": float(result.params[term]),
                "pvalue": float(result.pvalues[term]),
                "stat_abs": float(abs(result.tvalues[term])),
                "direction": "positive" if result.params[term] > 0 else "negative",
                "is_spatial_term": term in SPATIAL_TERMS,
            }
        )
    return pd.DataFrame(rows).sort_values(["is_spatial_term", "stat_abs"], ascending=[False, False])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictors = read_predictors(PREDICTORS_PATH) + SPATIAL_TERMS
    predictors = list(dict.fromkeys(predictors))

    df = add_spatial_terms(pd.read_parquet(INPUT_PATH))
    work = df[["T80_revised"] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X = sm.add_constant(work[predictors], has_constant="add")
    ols_model = sm.OLS(work["T80_revised"], X).fit()
    (OUT_DIR / "t80_continuous_ols_spatial_summary.txt").write_text(ols_model.summary().as_text())
    ols_terms = extract_linear_terms(ols_model, predictors)
    ols_terms.to_csv(OUT_DIR / "t80_continuous_ols_spatial_coefficients.csv", index=False)

    ordered_y = pd.Categorical(work["T80_revised"], categories=CLASSES, ordered=True)
    ordered_model = OrderedModel(ordered_y, work[predictors], distr="logit")
    ordered_result = ordered_model.fit(method="bfgs", maxiter=300, disp=False)
    (OUT_DIR / "t80_ordinal_logit_spatial_summary.txt").write_text(ordered_result.summary().as_text())
    ordered_terms = extract_ordered_terms(ordered_result, predictors)
    ordered_terms.to_csv(OUT_DIR / "t80_ordinal_logit_spatial_coefficients.csv", index=False)

    spatial_compare = pd.merge(
        ordered_terms.loc[ordered_terms["is_spatial_term"], ["term", "coef", "pvalue", "direction", "stat_abs"]].rename(
            columns={
                "coef": "ordinal_coef",
                "pvalue": "ordinal_pvalue",
                "direction": "ordinal_direction",
                "stat_abs": "ordinal_abs_z",
            }
        ),
        ols_terms.loc[ols_terms["is_spatial_term"], ["term", "coef", "pvalue", "direction", "stat_abs"]].rename(
            columns={
                "coef": "continuous_coef",
                "pvalue": "continuous_pvalue",
                "direction": "continuous_direction",
                "stat_abs": "continuous_abs_t",
            }
        ),
        on="term",
        how="outer",
    )
    spatial_compare.to_csv(OUT_DIR / "t80_spatial_terms_direction_significance.csv", index=False)

    metrics = pd.DataFrame(
        [
            {
                "model": "continuous_ols_spatial",
                "n_rows_used": int(len(work)),
                "r2": float(ols_model.rsquared),
                "adj_r2": float(ols_model.rsquared_adj),
                "aic": float(ols_model.aic),
                "bic": float(ols_model.bic),
            },
            {
                "model": "ordinal_logit_spatial",
                "n_rows_used": int(len(work)),
                "aic": float(ordered_result.aic),
                "bic": float(ordered_result.bic),
                "llf": float(ordered_result.llf),
            },
        ]
    )
    metrics.to_csv(OUT_DIR / "t80_interpretable_spatial_metrics.csv", index=False)

    notes = [
        "T80 interpretable spatial models",
        "Both models use the conservative OLS predictor set plus x, y, x_sq_z, y_sq_z, xy_z.",
        "Positive coefficient means association with slower / larger T80 category; negative means faster / smaller T80 category, holding others fixed.",
        "For the ordinal logit, coefficient signs describe movement toward higher T80 classes.",
    ]
    (OUT_DIR / "t80_interpretable_spatial_notes.txt").write_text("\n".join(notes))

    payload = {
        "predictors": predictors,
        "n_rows_used": int(len(work)),
    }
    (OUT_DIR / "t80_interpretable_spatial_run.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
