import json
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
FULL_SAF_DIR = ROOT / "resistance_full_saf_models_2026-03-30"
OUT = ROOT / "current_models_full_saf_summary_2026-03-30"
INPUT = ROOT / "resistance_model_execution_near_t0_aggregated" / "MGWR_ready_table_near_t0_aggregated.parquet"

ID_COLS = ["pixel_id", "row", "col", "x", "y", "t0_year"]
RESPONSE = "Resistance"
BASE_PREDICTORS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_northness_z",
    "TS_eastness_z",
    "TS_twi_z",
    "TS_roughness_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z",
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_eto_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_tmmx_std_pre_z",
]


def build_full_saf(df: pd.DataFrame, codes: list[int]) -> pd.DataFrame:
    out = df.copy()
    saf = pd.to_numeric(out["FS_EVT_t0agg_SAF_code"], errors="coerce").astype("Int64")
    for code in codes:
        out[f"FS_EVT_t0agg_SAF_{code}"] = (saf == code).fillna(False).astype(int)
    return out


def get_full_saf_predictors() -> tuple[list[int], list[str]]:
    meta = json.loads((FULL_SAF_DIR / "full_saf_metadata.json").read_text())
    codes = meta["full_saf_codes"]
    predictors = BASE_PREDICTORS + meta["full_saf_indicator_columns"]
    return codes, predictors


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=float), index=vals.index)
    return (vals - vals.mean()) / std


def add_spatial_terms(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x_sq"] = out["x"] ** 2
    out["y_sq"] = out["y"] ** 2
    out["xy"] = out["x"] * out["y"]
    out["x_sq_z"] = zscore(out["x_sq"])
    out["y_sq_z"] = zscore(out["y_sq"])
    out["xy_z"] = zscore(out["xy"])
    return out


def moran_row(label: str, values: np.ndarray, w) -> dict:
    m = Moran(values.astype(float), w, permutations=999)
    return {
        "variable": label,
        "moran_I": float(m.I),
        "p_value": float(m.p_sim),
        "z_score": float(m.z_sim),
    }


def run_variant(work: pd.DataFrame, predictors: list[str], model_name: str, out_dir: Path) -> tuple[dict, dict]:
    X = work[predictors]
    y = work[RESPONSE]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_eval = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf_eval.fit(X_train, y_train)
    pred_test = rf_eval.predict(X_test)

    rf_full = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf_full.fit(X, y)
    pred_full = rf_full.predict(X)
    resid_full = (y - pred_full).to_numpy()

    coords = work[["x", "y"]].to_numpy()
    w = KNN.from_array(coords, k=8)
    w.transform = "R"
    moran = moran_row(f"{model_name}_residual", resid_full, w)

    id_cols_unique = []
    for col in ID_COLS:
        if col not in id_cols_unique:
            id_cols_unique.append(col)
    metrics = {
        "model_name": model_name,
        "input_dataset": str(INPUT),
        "response": RESPONSE,
        "predictor_count": len(predictors),
        "predictors": predictors,
        "success": True,
        "r2": float(r2_score(y_test, pred_test)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "residual_moran_i": moran["moran_I"],
        "residual_moran_p": moran["p_value"],
        "residual_moran_z": moran["z_score"],
    }
    (out_dir / f"{model_name}_metrics.json").write_text(json.dumps(metrics, indent=2))

    coef_like = work[id_cols_unique + [RESPONSE]].copy()
    coef_like["fitted"] = pred_full
    coef_like["residual"] = resid_full
    coef_like.to_parquet(out_dir / f"{model_name}_predictions.parquet", index=False)
    coef_like.to_csv(out_dir / f"{model_name}_predictions.csv", index=False)
    return metrics, moran


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT / "rf_full_saf_xy_variants_2026-04-01"
    out_dir.mkdir(parents=True, exist_ok=True)

    codes, predictors_full_saf = get_full_saf_predictors()
    df = pd.read_parquet(INPUT)
    df = build_full_saf(df, codes)
    df = add_spatial_terms(df)

    needed = []
    for col in ID_COLS + [RESPONSE] + predictors_full_saf + ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]:
        if col not in needed:
            needed.append(col)
    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()

    variants = {
        "RF_full_SAF_plusxy": predictors_full_saf + ["x", "y"],
        "RF_full_SAF_plusxy_poly": predictors_full_saf + ["x", "y", "x_sq_z", "y_sq_z", "xy_z"],
    }

    metric_rows = []
    moran_rows = [moran_row("Resistance", work[RESPONSE].to_numpy(), KNN.from_array(work[["x", "y"]].to_numpy(), k=8))]
    moran_rows[0]["p_value"] = float(moran_rows[0]["p_value"])

    for model_name, predictors in variants.items():
        metrics, moran = run_variant(work, predictors, model_name, out_dir)
        metric_rows.append(metrics)
        moran_rows.append(moran)

    pd.DataFrame(metric_rows).to_csv(out_dir / "rf_full_saf_xy_variants_summary.csv", index=False)
    pd.DataFrame(moran_rows).to_csv(out_dir / "rf_full_saf_xy_variants_moran.csv", index=False)

    notes = [
        "RF full SAF spatial-term variants",
        "",
        f"n_rows_used={len(work)}",
    ]
    for row in metric_rows:
        notes.append(
            f"{row['model_name']}: R2={row['r2']:.6f}, RMSE={row['rmse']:.6f}, MoranI={row['residual_moran_i']:.6f}"
        )
    (out_dir / "rf_full_saf_xy_variants_notes.txt").write_text("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
