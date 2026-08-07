from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_increment_recovery_near_t0_postfire.parquet"
PREDICTORS_PATH = BASE_DIR / "predictors_rf_xgb_recovery_baseline.txt"
OUT_DIR = BASE_DIR / "increment_recovery_rf_spatial_terms_2026-03-31"
RESPONSES = ["INC_end_rel_10obs", "INC_cum_rel_10obs"]
RANDOM_STATE = 42
MORAN_K = 8


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


def run_variant(df: pd.DataFrame, response: str, variant_name: str, extra_predictors: list[str]) -> dict:
    predictors = read_predictors(PREDICTORS_PATH) + extra_predictors
    predictors = list(dict.fromkeys(predictors))
    cols = list(dict.fromkeys([response] + predictors + ["x", "y", "t0_year"]))
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

    run_dir = OUT_DIR / response / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"predictor": predictors, "importance": full_model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(run_dir / f"{variant_name}_importance.csv", index=False)

    metrics = {
        "response": response,
        "variant": variant_name,
        "n_rows_used": int(len(work)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "full_fit_residual_moran": compute_moran(work[["x", "y"]], residuals),
        "predictors": predictors,
        "t0_year_min": int(work["t0_year"].min()),
        "t0_year_max": int(work["t0_year"].max()),
    }
    (run_dir / f"{variant_name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return {
        "response": response,
        "variant": variant_name,
        "n_rows_used": metrics["n_rows_used"],
        "test_r2": metrics["test_r2"],
        "test_rmse": metrics["test_rmse"],
        "moran_i": metrics["full_fit_residual_moran"]["moran_i"],
        "t0_year_min": metrics["t0_year_min"],
        "t0_year_max": metrics["t0_year_max"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = add_spatial_terms(pd.read_parquet(INPUT_PATH))
    rows = []
    for response in RESPONSES:
        rows.append(run_variant(df, response, "rf_noxy", []))
        rows.append(run_variant(df, response, "rf_plusxy", ["x", "y"]))
        rows.append(run_variant(df, response, "rf_plusxy_poly", ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "increment_recovery_rf_spatial_terms_summary.csv", index=False)
    lines = ["RF spatial-term comparison for increment recovery y"]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['response']} | {row['variant']}: r2={row['test_r2']:.4f}, "
            f"rmse={row['test_rmse']:.4f}, moran_i={row['moran_i']:.4f}, "
            f"n={int(row['n_rows_used'])}, years={int(row['t0_year_min'])}-{int(row['t0_year_max'])}"
        )
    (OUT_DIR / "increment_recovery_rf_spatial_terms_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
