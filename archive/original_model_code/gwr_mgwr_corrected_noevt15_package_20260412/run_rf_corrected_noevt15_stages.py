#!/usr/bin/env python3
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


PACKAGE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/gwr_mgwr_corrected_noevt15_package_20260412"
)
INPUT = PACKAGE_DIR / "GWR_MGWR_ready_table_corrected_noevt15.parquet"
OUT_DIR = PACKAGE_DIR / "rf_stage_comparison_20260414"

ID_COLS = ["pixel_id", "row", "col", "x", "y", "t0_year"]
RESPONSE = "Resistance"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500
MORAN_K = 8

STAGES = [
    ("stage1_topo_soil_4", PACKAGE_DIR / "predictors_stage1_topo_soil_4.txt"),
    ("stage2_topo_soil_forest_6", PACKAGE_DIR / "predictors_stage2_topo_soil_forest_6.txt"),
    ("stage3_plus_human_core_9", PACKAGE_DIR / "predictors_stage3_plus_human_core_9.txt"),
    ("stage4_plus_access_11", PACKAGE_DIR / "predictors_stage4_plus_access_11.txt"),
    ("stage5_plus_climate_15", PACKAGE_DIR / "predictors_stage5_plus_climate_15.txt"),
]


def read_predictors(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def compute_moran(coords: np.ndarray, residuals: np.ndarray, k: int = MORAN_K) -> dict:
    weights = KNN.from_array(coords, k=k)
    weights.transform = "R"
    moran = Moran(residuals.astype(float), weights, permutations=0)
    return {
        "k": int(k),
        "n_obs": int(len(coords)),
        "moran_i": float(moran.I),
        "z_norm": float(moran.z_norm),
        "p_norm": float(moran.p_norm),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT)

    summary_rows: list[dict] = []
    moran_rows: list[dict] = []

    for stage_name, predictor_file in STAGES:
        predictors = read_predictors(predictor_file)
        cols = ID_COLS + [RESPONSE] + predictors
        work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        X = work[predictors]
        y = work[RESPONSE]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        eval_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        eval_model.fit(X_train, y_train)
        pred_test = eval_model.predict(X_test)

        full_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        full_model.fit(X, y)
        pred_full = full_model.predict(X)
        residuals = y.to_numpy() - pred_full

        moran = compute_moran(work[["x", "y"]].to_numpy(dtype=float), residuals, MORAN_K)
        moran["stage"] = stage_name
        moran_rows.append(moran)

        metrics = {
            "stage": stage_name,
            "predictor_file": str(predictor_file),
            "rows_used": int(len(work)),
            "predictor_count": int(len(predictors)),
            "predictors": predictors,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_r2": float(r2_score(y_test, pred_test)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "full_r2": float(r2_score(y, pred_full)),
            "full_rmse": float(np.sqrt(mean_squared_error(y, pred_full))),
            "moran_i": float(moran["moran_i"]),
        }
        summary_rows.append(metrics)

        stage_dir = OUT_DIR / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        importance_df = pd.DataFrame(
            {"variable": predictors, "importance": full_model.feature_importances_}
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(stage_dir / "rf_feature_importance.csv", index=False)

        residual_df = work[ID_COLS + [RESPONSE]].copy()
        residual_df["prediction"] = pred_full
        residual_df["residual"] = residuals
        residual_df.to_parquet(stage_dir / "rf_residuals.parquet", index=False)

        (stage_dir / "rf_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "rf_stage_metrics.csv", index=False)

    moran_df = pd.DataFrame(moran_rows)[["stage", "k", "n_obs", "moran_i", "z_norm", "p_norm"]]
    moran_df.to_csv(OUT_DIR / "rf_stage_moran.csv", index=False)

    lines = [
        "# RF Stage Comparison (Exact Stage1-5)",
        "",
        f"- Input table: `{INPUT}`",
        f"- Response: `{RESPONSE}`",
        f"- Random state: `{RANDOM_STATE}`",
        f"- Test size: `{TEST_SIZE}`",
        f"- Trees per model: `{N_ESTIMATORS}`",
        "",
        "## Stage metrics",
        summary_df.to_csv(index=False).strip(),
        "",
        "## Residual Moran's I",
        moran_df.to_csv(index=False).strip(),
    ]
    (OUT_DIR / "rf_stage_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "metrics_csv": str(OUT_DIR / "rf_stage_metrics.csv"),
                "moran_csv": str(OUT_DIR / "rf_stage_moran.csv"),
                "report_md": str(OUT_DIR / "rf_stage_report.md"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
