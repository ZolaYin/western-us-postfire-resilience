import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


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
    df = df.copy()
    saf = pd.to_numeric(df["FS_EVT_t0agg_SAF_code"], errors="coerce").astype("Int64")
    for code in codes:
        df[f"FS_EVT_t0agg_SAF_{code}"] = (saf == code).fillna(False).astype(int)
    return df


def get_full_saf_predictors() -> tuple[list[int], list[str]]:
    meta = json.loads((FULL_SAF_DIR / "full_saf_metadata.json").read_text())
    codes = meta["full_saf_codes"]
    predictors = BASE_PREDICTORS + meta["full_saf_indicator_columns"]
    return codes, predictors


def moran_row(label: str, values: np.ndarray, w) -> dict:
    m = Moran(values.astype(float), w, permutations=999)
    return {
        "variable": label,
        "moran_I": float(m.I),
        "p_value": float(m.p_sim),
        "z_score": float(m.z_sim),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    codes, predictors = get_full_saf_predictors()
    df = pd.read_parquet(INPUT)
    df = build_full_saf(df, codes)
    work = df[ID_COLS + [RESPONSE] + predictors].replace([np.inf, -np.inf], np.nan).dropna().copy()

    coords = work[["x", "y"]].to_numpy()
    w = KNN.from_array(coords, k=8)
    w.transform = "R"

    summary_rows = []
    residual_rows = [moran_row("Resistance", work[RESPONSE].to_numpy(), w)]

    # OLS
    X_ols = sm.add_constant(work[predictors], has_constant="add")
    y = work[RESPONSE]
    ols = sm.OLS(y, X_ols).fit()
    pred_ols = ols.predict(X_ols)
    resid_ols = (y - pred_ols).to_numpy()
    residual_rows.append(moran_row("OLS_full_SAF_residual", resid_ols, w))
    summary_rows.append({
        "model_name": "OLS_full_SAF",
        "input_dataset": str(INPUT),
        "response": RESPONSE,
        "predictor_count": len(predictors),
        "x_variables": ", ".join(predictors),
        "y_variable": RESPONSE,
        "success": True,
        "r2": float(ols.rsquared),
        "adj_r2": float(ols.rsquared_adj),
        "rmse": float(np.sqrt(np.mean((y - pred_ols) ** 2))),
        "aic": float(ols.aic),
        "bic": float(ols.bic),
        "evt_representation": "full_SAF",
    })

    # RF
    X = work[predictors]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_eval = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf_eval.fit(X_train, y_train)
    pred_test_rf = rf_eval.predict(X_test)
    rf_full = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    rf_full.fit(X, y)
    pred_rf = rf_full.predict(X)
    resid_rf = (y - pred_rf).to_numpy()
    residual_rows.append(moran_row("RF_full_SAF_residual", resid_rf, w))
    summary_rows.append({
        "model_name": "RF_full_SAF",
        "input_dataset": str(INPUT),
        "response": RESPONSE,
        "predictor_count": len(predictors),
        "x_variables": ", ".join(predictors),
        "y_variable": RESPONSE,
        "success": True,
        "r2": float(r2_score(y_test, pred_test_rf)),
        "adj_r2": np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test_rf))),
        "aic": np.nan,
        "bic": np.nan,
        "evt_representation": "full_SAF",
    })

    # XGBoost
    xgb_eval = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    xgb_eval.fit(X_train, y_train)
    pred_test_xgb = xgb_eval.predict(X_test)
    xgb_full = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    xgb_full.fit(X, y)
    pred_xgb = xgb_full.predict(X)
    resid_xgb = (y - pred_xgb).to_numpy()
    residual_rows.append(moran_row("XGBoost_full_SAF_residual", resid_xgb, w))
    summary_rows.append({
        "model_name": "XGBoost_full_SAF",
        "input_dataset": str(INPUT),
        "response": RESPONSE,
        "predictor_count": len(predictors),
        "x_variables": ", ".join(predictors),
        "y_variable": RESPONSE,
        "success": True,
        "r2": float(r2_score(y_test, pred_test_xgb)),
        "adj_r2": np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test_xgb))),
        "aic": np.nan,
        "bic": np.nan,
        "evt_representation": "full_SAF",
    })

    pd.DataFrame(summary_rows).to_csv(OUT / "current_full_saf_model_summary.csv", index=False)
    pd.DataFrame(residual_rows).to_csv(OUT / "current_full_saf_residual_spatial_summary.csv", index=False)

    notes = [
        "Current full-SAF model summary and residual spatial diagnostics",
        "",
        "Setup",
        "- Input dataset: MGWR_ready_table_near_t0_aggregated.parquet",
        "- Response variable: Resistance",
        "- X variables: static topo/soil + near-t0 forest/human + pre-fire climate + all observed SAF-code indicators",
        "- Number of observed SAF codes expanded: 32",
        "",
        "Model performance",
    ]
    for row in summary_rows:
        notes.append(
            f"- {row['model_name']}: predictor_count={row['predictor_count']}, "
            f"R2={row['r2']:.6f}, RMSE={row['rmse']:.6f}"
        )
    notes.extend([
        "",
        "Residual Moran's I",
    ])
    for row in residual_rows:
        notes.append(
            f"- {row['variable']}: I={row['moran_I']:.6f}, p={row['p_value']:.3f}, z={row['z_score']:.3f}"
        )
    notes.extend([
        "",
        "Interpretation",
        "- OLS_full_SAF is the most interpretable because each SAF code enters explicitly as a coefficient, but it still leaves strong spatial structure.",
        "- RF_full_SAF has the best predictive performance among the three current full-SAF models and removes the most spatial structure, but residual Moran's I remains clearly above zero.",
        "- XGBoost_full_SAF is nonlinear and flexible, but still leaves more spatial structure than RF.",
        "- Because all three residual sets remain spatially autocorrelated, introducing spatial modeling is scientifically justified.",
        "- Because earlier all-in GWR attempts were singular, the next spatial step should be grouped/staged GWR rather than immediate full MGWR.",
    ])
    (OUT / "current_full_saf_model_notes.txt").write_text("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
