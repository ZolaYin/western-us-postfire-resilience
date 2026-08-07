from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_t80_iri_stability_near_t0_postfire.parquet"
PREDICTORS_PATH = BASE_DIR / "predictors_rf_xgb_recovery_baseline.txt"
OUT_DIR = BASE_DIR / "ordinal_rf_spatial_terms_2026-03-31"
RANDOM_STATE = 42
MORAN_K = 8
CLASSES = list(range(2, 11))


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


def probs_to_expected_year(probs: np.ndarray) -> np.ndarray:
    return probs @ np.array(CLASSES, dtype=float)


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


def run_variant(df: pd.DataFrame, variant_name: str, extra_predictors: list[str]) -> dict:
    predictors = read_predictors(PREDICTORS_PATH) + ["x", "y"] + extra_predictors
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
    expected_test = probs_to_expected_year(probs_test)

    full_model = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    full_model.fit(work[predictors], work["T80_revised"])
    probs_full = full_model.predict_proba(work[predictors])
    expected_full = probs_to_expected_year(probs_full)
    residuals = work["T80_revised"].to_numpy() - expected_full

    run_dir = OUT_DIR / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictor": predictors, "importance": full_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(run_dir / f"{variant_name}_importance.csv", index=False)

    metrics = {
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
        "variant": variant_name,
        "test_accuracy": metrics["test_accuracy"],
        "test_macro_f1": metrics["test_macro_f1"],
        "test_expected_r2": metrics["test_expected_r2"],
        "test_expected_rmse": metrics["test_expected_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT_PATH)
    df = add_spatial_terms(df)

    rows = [
        run_variant(df, "ordinal_rf_plusxy", []),
        run_variant(df, "ordinal_rf_plusxy_poly", ["x_sq_z", "y_sq_z", "xy_z"]),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "ordinal_rf_spatial_terms_summary.csv", index=False)
    lines = ["Ordinal RF spatial-terms comparison"]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['variant']}: expected_r2={row['test_expected_r2']:.4f}, "
            f"rmse={row['test_expected_rmse']:.4f}, accuracy={row['test_accuracy']:.4f}, "
            f"macro_f1={row['test_macro_f1']:.4f}, moran_i={row['moran_i']:.4f}"
        )
    (OUT_DIR / "ordinal_rf_spatial_terms_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
