from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_t80_iri_stability_near_t0_postfire_with_mtbs_severity.parquet"
PREDICTORS_PATH = BASE_DIR / "predictors_rf_xgb_recovery_baseline.txt"
OUT_DIR = BASE_DIR / "fire_severity_categorical_all_rf_2026-03-31"
RANDOM_STATE = 42
MORAN_K = 8


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def compute_moran(coords_df: pd.DataFrame, residuals: np.ndarray) -> dict:
    weights = KNN.from_array(coords_df[["x", "y"]].to_numpy(), k=MORAN_K)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return {
        "k": MORAN_K,
        "n_obs": int(len(coords_df)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def probs_to_expected_year(probs: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    return probs @ model_classes.astype(float)


def add_severity_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    sev = out["FIRE_mtbs_sev_t0"].copy()
    sev = sev.where(sev.isin([1, 2, 3, 4, 5, 6]))
    sev_cat = pd.Categorical(sev, categories=[1, 2, 3, 4, 5, 6])
    dummies = pd.get_dummies(sev_cat, prefix="FIRE_mtbs_sev_class", dtype=int)
    out = pd.concat([out, dummies], axis=1)
    return out, list(dummies.columns)


def run_regression_variant(
    df: pd.DataFrame,
    response: str,
    family: str,
    variant_name: str,
    extra_predictors: list[str],
    severity_predictors: list[str],
) -> dict:
    predictors = read_predictors(PREDICTORS_PATH) + extra_predictors + severity_predictors
    predictors = list(dict.fromkeys(predictors))
    cols = list(dict.fromkeys([response] + predictors + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    X_train, X_test, y_train, y_test = train_test_split(
        work[predictors], work[response], test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    full_model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[predictors], work[response])
    full_pred = full_model.predict(work[predictors])
    residuals = work[response].to_numpy() - full_pred

    run_dir = OUT_DIR / family / response / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictor": predictors, "importance": full_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(run_dir / f"{variant_name}_importance.csv", index=False)

    metrics = {
        "family": family,
        "response": response,
        "variant": variant_name,
        "n_rows_used": int(len(work)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
    }
    (run_dir / f"{variant_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {
        "family": family,
        "response": response,
        "variant": variant_name,
        "n_rows_used": metrics["n_rows_used"],
        "test_r2": metrics["test_r2"],
        "test_rmse": metrics["test_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
    }


def run_ordinal_variant(
    df: pd.DataFrame,
    variant_name: str,
    extra_predictors: list[str],
    severity_predictors: list[str],
) -> dict:
    predictors = read_predictors(PREDICTORS_PATH) + extra_predictors + severity_predictors
    predictors = list(dict.fromkeys(predictors))
    cols = list(dict.fromkeys(["T80_revised"] + predictors + ["x", "y"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    train_idx, test_idx = train_test_split(
        work.index,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=work["T80_revised"],
    )
    train = work.loc[train_idx].copy()
    test = work.loc[test_idx].copy()

    model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train[predictors], train["T80_revised"])
    probs_test = model.predict_proba(test[predictors])
    pred_test = model.predict(test[predictors])
    expected_test = probs_to_expected_year(probs_test, model.classes_)

    full_model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[predictors], work["T80_revised"])
    probs_full = full_model.predict_proba(work[predictors])
    expected_full = probs_to_expected_year(probs_full, full_model.classes_)
    residuals = work["T80_revised"].to_numpy() - expected_full

    run_dir = OUT_DIR / "t80_ordinal_rf" / "T80_revised" / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictor": predictors, "importance": full_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(run_dir / f"{variant_name}_importance.csv", index=False)

    metrics = {
        "family": "t80_ordinal_rf",
        "response": "T80_revised",
        "variant": variant_name,
        "n_rows_used": int(len(work)),
        "test_accuracy": float(accuracy_score(test["T80_revised"], pred_test)),
        "test_macro_f1": float(f1_score(test["T80_revised"], pred_test, average="macro")),
        "test_expected_r2": float(r2_score(test["T80_revised"], expected_test)),
        "test_expected_rmse": float(np.sqrt(mean_squared_error(test["T80_revised"], expected_test))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
    }
    (run_dir / f"{variant_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {
        "family": "t80_ordinal_rf",
        "response": "T80_revised",
        "variant": variant_name,
        "n_rows_used": metrics["n_rows_used"],
        "test_accuracy": metrics["test_accuracy"],
        "test_macro_f1": metrics["test_macro_f1"],
        "test_expected_r2": metrics["test_expected_r2"],
        "test_expected_rmse": metrics["test_expected_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, severity_cols = add_severity_dummies(pd.read_parquet(INPUT_PATH))

    rows_reg = []
    rows_ord = []

    reg_variants = [
        ("rf_noxy", []),
        ("rf_plusxy", ["x", "y"]),
        ("rf_plusxy_poly", ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]),
    ]
    ord_variants = [
        ("ordinal_rf_noxy", []),
        ("ordinal_rf_plusxy", ["x", "y"]),
        ("ordinal_rf_plusxy_poly", ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]),
    ]

    for response in ["T80_revised", "IRI_good_10yr", "STAB_10yr"]:
        for base_name, extras in reg_variants:
            rows_reg.append(
                run_regression_variant(df, response, "recovery_rf", f"{base_name}_base", extras, [])
            )
            rows_reg.append(
                run_regression_variant(
                    df, response, "recovery_rf", f"{base_name}_sevcat", extras, severity_cols
                )
            )

    for base_name, extras in ord_variants:
        rows_ord.append(run_ordinal_variant(df, f"{base_name}_base", extras, []))
        rows_ord.append(run_ordinal_variant(df, f"{base_name}_sevcat", extras, severity_cols))

    reg_summary = pd.DataFrame(rows_reg)
    ord_summary = pd.DataFrame(rows_ord)
    reg_summary.to_csv(OUT_DIR / "recovery_rf_sevcat_summary.csv", index=False)
    ord_summary.to_csv(OUT_DIR / "t80_ordinal_rf_sevcat_summary.csv", index=False)

    lines = ["Categorical fire severity contrast across RF families", ""]
    lines.append("T80 ordinal RF")
    for _, row in ord_summary.iterrows():
        lines.append(
            f"{row['variant']}: expected_r2={row['test_expected_r2']:.4f}, "
            f"rmse={row['test_expected_rmse']:.4f}, accuracy={row['test_accuracy']:.4f}, "
            f"macro_f1={row['test_macro_f1']:.4f}, moran_i={row['moran_i']:.4f}, "
            f"n={int(row['n_rows_used'])}"
        )
    lines.append("")
    for response in ["T80_revised", "IRI_good_10yr", "STAB_10yr"]:
        lines.append(response)
        sub = reg_summary[reg_summary["response"] == response]
        for _, row in sub.iterrows():
            lines.append(
                f"{row['variant']}: r2={row['test_r2']:.4f}, rmse={row['test_rmse']:.4f}, "
                f"moran_i={row['moran_i']:.4f}, n={int(row['n_rows_used'])}"
            )
        lines.append("")
    (OUT_DIR / "categorical_fire_severity_contrast_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
