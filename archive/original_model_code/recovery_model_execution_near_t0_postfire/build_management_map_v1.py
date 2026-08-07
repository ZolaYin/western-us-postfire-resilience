from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


ROOT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
RECOVERY_DIR = ROOT / "recovery_model_execution_near_t0_postfire"
RESISTANCE_DIR = ROOT / "resistance_model_execution_near_t0_aggregated"
OUT_DIR = RECOVERY_DIR / "management_map_v1_2026-03-31"
RANDOM_STATE = 42
T80_CLASSES = list(range(2, 11))
N_ESTIMATORS = 200


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


def percentile_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def classify_priority(values: pd.Series) -> pd.Series:
    bins = values.quantile([0.2, 0.4, 0.6, 0.8]).to_list()
    labels = ["Very Low", "Low", "Moderate", "High", "Very High"]
    return pd.cut(values, bins=[-np.inf] + bins + [np.inf], labels=labels, include_lowest=True)


def group_map_response(response: str) -> dict[str, list[str]]:
    common = {
        "topo_soil": [
            "TS_elev_m_z",
            "TS_slope_deg_z",
            "TS_northness_z",
            "TS_eastness_z",
            "TS_twi_z",
            "TS_roughness_z",
            "TS_SOC_0_30cm_z",
        ],
        "forest": [
            "FS_TCC_t0_z",
            "FS_CBH_t0agg_z",
            "FS_EVT_t0agg_resistance_proxy",
        ],
        "human": [
            "HUM_popdens_win10km_log_z",
            "HUM_roaddens_r5km_z",
            "HUM_traildens_r10km_z",
            "HUM_imperv_near_t0_z",
            "HUM_viirs_near_t0_log_z",
        ],
    }
    if response == "Resistance":
        common["climate"] = [
            "CLIM_pr_sum_pre_z",
            "CLIM_eto_sum_pre_z",
            "CLIM_tmmn_mean_pre_z",
            "CLIM_hot_days_35C_pre_z",
            "CLIM_aridity_pre_z",
            "CLIM_tmmx_std_pre_z",
        ]
        common["space"] = ["x", "y"]
    else:
        common["climate"] = [
            "CLIM_pr_sum_post_z",
            "CLIM_eto_sum_post_z",
            "CLIM_tmmn_mean_post_z",
            "CLIM_hot_days_35C_post_z",
            "CLIM_aridity_post_z",
            "CLIM_tmmx_std_post_z",
        ]
        common["space"] = ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]
    return common


def make_recommendation(priority: str, dominant_group: str) -> str:
    if priority in {"Very Low", "Low"}:
        return f"{priority} priority"
    mapping = {
        "climate": "Climate adaptation priority",
        "forest": "Forest composition/structure priority",
        "human": "Human pressure / ignition priority",
        "topo_soil": "Site limitation / refugia priority",
        "space": "Regional context priority",
    }
    return f"{priority} - {mapping.get(dominant_group, dominant_group)}"


def group_ablation_contributions(model, X_full: pd.DataFrame, group_map: dict[str, list[str]], expected_year: bool = False) -> pd.DataFrame:
    base_pred = predict_model(model, X_full, expected_year=expected_year)
    ref = X_full.median(numeric_only=True)
    out = pd.DataFrame(index=X_full.index)
    for group, cols in group_map.items():
        X_alt = X_full.copy()
        for col in cols:
            if col in X_alt.columns:
                X_alt[col] = ref[col]
        alt_pred = predict_model(model, X_alt, expected_year=expected_year)
        out[group] = base_pred - alt_pred
    return out


def predict_model(model, X: pd.DataFrame, expected_year: bool = False) -> np.ndarray:
    if expected_year:
        probs = model.predict_proba(X)
        return probs @ np.array(T80_CLASSES, dtype=float)
    return model.predict(X)


def fit_regressor(df: pd.DataFrame, response: str, predictors: list[str]) -> tuple[RandomForestRegressor, pd.DataFrame]:
    cols = list(dict.fromkeys([response] + predictors + ["pixel_id", "x", "y", "t0_year"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(work[predictors], work[response])
    return model, work


def fit_t80_classifier(df: pd.DataFrame, predictors: list[str]) -> tuple[RandomForestClassifier, pd.DataFrame]:
    cols = list(dict.fromkeys(["T80_revised"] + predictors + ["pixel_id", "x", "y", "t0_year"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(work[predictors], work["T80_revised"])
    return model, work


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    resistance_df = pd.read_parquet(RESISTANCE_DIR / "MGWR_ready_table_near_t0_aggregated.parquet")
    recovery_df = add_spatial_terms(pd.read_parquet(RECOVERY_DIR / "MGWR_ready_table_t80_iri_stability_near_t0_postfire.parquet"))

    resistance_predictors = [
        "TS_elev_m_z",
        "TS_slope_deg_z",
        "TS_northness_z",
        "TS_eastness_z",
        "TS_twi_z",
        "TS_roughness_z",
        "TS_SOC_0_30cm_z",
        "FS_TCC_t0_z",
        "FS_CBH_t0agg_z",
        "FS_EVT_t0agg_resistance_proxy",
        "HUM_popdens_win10km_log_z",
        "HUM_roaddens_r5km_z",
        "HUM_traildens_r10km_z",
        "HUM_imperv_near_t0_z",
        "HUM_viirs_near_t0_log_z",
        "CLIM_pr_sum_pre_z",
        "CLIM_eto_sum_pre_z",
        "CLIM_tmmn_mean_pre_z",
        "CLIM_hot_days_35C_pre_z",
        "CLIM_aridity_pre_z",
        "CLIM_tmmx_std_pre_z",
        "x",
        "y",
    ]
    recovery_base_predictors = [
        "TS_elev_m_z",
        "TS_slope_deg_z",
        "TS_northness_z",
        "TS_eastness_z",
        "TS_twi_z",
        "TS_roughness_z",
        "TS_SOC_0_30cm_z",
        "FS_TCC_t0_z",
        "FS_CBH_t0agg_z",
        "FS_EVT_t0agg_resistance_proxy",
        "HUM_popdens_win10km_log_z",
        "HUM_roaddens_r5km_z",
        "HUM_traildens_r10km_z",
        "HUM_imperv_near_t0_z",
        "HUM_viirs_near_t0_log_z",
        "CLIM_pr_sum_post_z",
        "CLIM_eto_sum_post_z",
        "CLIM_tmmn_mean_post_z",
        "CLIM_hot_days_35C_post_z",
        "CLIM_aridity_post_z",
        "CLIM_tmmx_std_post_z",
    ]
    t80_predictors = recovery_base_predictors + ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]
    iri_stab_predictors = recovery_base_predictors + ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]

    resistance_model, resistance_work = fit_regressor(resistance_df, "Resistance", resistance_predictors)
    t80_model, t80_work = fit_t80_classifier(recovery_df, t80_predictors)
    iri_model, iri_work = fit_regressor(recovery_df, "IRI_good_10yr", iri_stab_predictors)
    stab_model, stab_work = fit_regressor(recovery_df, "STAB_10yr", iri_stab_predictors)

    base = recovery_df[["pixel_id", "x", "y", "t0_year"]].copy()
    management = base.copy()

    resistance_full = resistance_df[list(dict.fromkeys(["pixel_id"] + resistance_predictors))].replace([np.inf, -np.inf], np.nan).dropna().copy()
    resistance_full["pred_Resistance"] = predict_model(resistance_model, resistance_full[resistance_predictors])
    resistance_contrib = group_ablation_contributions(
        resistance_model, resistance_full[resistance_predictors], group_map_response("Resistance"), expected_year=False
    )
    resistance_out = pd.concat([resistance_full[["pixel_id", "pred_Resistance"]], resistance_contrib.add_prefix("Resistance__")], axis=1)

    t80_full = recovery_df[list(dict.fromkeys(["pixel_id"] + t80_predictors))].replace([np.inf, -np.inf], np.nan).dropna().copy()
    t80_full["pred_T80"] = predict_model(t80_model, t80_full[t80_predictors], expected_year=True)
    t80_contrib = group_ablation_contributions(
        t80_model, t80_full[t80_predictors], group_map_response("T80"), expected_year=True
    )
    t80_out = pd.concat([t80_full[["pixel_id", "pred_T80"]], t80_contrib.add_prefix("T80__")], axis=1)

    iri_full = recovery_df[list(dict.fromkeys(["pixel_id"] + iri_stab_predictors))].replace([np.inf, -np.inf], np.nan).dropna().copy()
    iri_full["pred_IRI_good_10yr"] = predict_model(iri_model, iri_full[i_stab_predictors] if False else iri_full[iri_stab_predictors])
    iri_contrib = group_ablation_contributions(
        iri_model, iri_full[iri_stab_predictors], group_map_response("IRI"), expected_year=False
    )
    iri_out = pd.concat([iri_full[["pixel_id", "pred_IRI_good_10yr"]], iri_contrib.add_prefix("IRI__")], axis=1)

    stab_full = recovery_df[list(dict.fromkeys(["pixel_id"] + iri_stab_predictors))].replace([np.inf, -np.inf], np.nan).dropna().copy()
    stab_full["pred_STAB_10yr"] = predict_model(stab_model, stab_full[iri_stab_predictors])
    stab_contrib = group_ablation_contributions(
        stab_model, stab_full[i_stab_predictors] if False else stab_full[iri_stab_predictors], group_map_response("STAB"), expected_year=False
    )
    stab_out = pd.concat([stab_full[["pixel_id", "pred_STAB_10yr"]], stab_contrib.add_prefix("STAB__")], axis=1)

    for piece in [resistance_out, t80_out, iri_out, stab_out]:
        management = management.merge(piece, on="pixel_id", how="left")

    for prefix in ["Resistance", "T80", "IRI", "STAB"]:
        contrib_cols = [c for c in management.columns if c.startswith(f"{prefix}__")]
        if not contrib_cols:
            continue
        abs_df = management[contrib_cols].abs().copy()
        abs_df.columns = [c.split("__", 1)[1] for c in abs_df.columns]
        management[f"{prefix}_dominant_group_all"] = abs_df.idxmax(axis=1)
        actionable = abs_df.drop(columns=["space"], errors="ignore")
        management[f"{prefix}_dominant_group_actionable"] = actionable.idxmax(axis=1)

    management["risk_resistance"] = 1.0 - percentile_rank(management["pred_Resistance"])
    management["risk_t80"] = percentile_rank(management["pred_T80"])
    management["risk_iri"] = 1.0 - percentile_rank(management["pred_IRI_good_10yr"])
    management["risk_stab"] = 1.0 - percentile_rank(management["pred_STAB_10yr"])
    management["management_need_index"] = management[["risk_resistance", "risk_t80", "risk_iri", "risk_stab"]].mean(axis=1)
    management["management_priority_class"] = classify_priority(management["management_need_index"]).astype(str)

    actionable_groups = ["climate", "forest", "human", "topo_soil"]
    integrated = pd.DataFrame(index=management.index)
    for group in actionable_groups:
        cols = []
        for prefix in ["Resistance", "T80", "IRI", "STAB"]:
            col = f"{prefix}__{group}"
            if col in management.columns:
                cols.append(col)
        integrated[group] = management[cols].abs().mean(axis=1)
    management["integrated_dominant_actionable_group"] = integrated.idxmax(axis=1)
    management["management_recommendation_class"] = [
        make_recommendation(priority, group)
        for priority, group in zip(
            management["management_priority_class"], management["integrated_dominant_actionable_group"]
        )
    ]

    management.to_parquet(OUT_DIR / "management_map_v1_predictions_and_recommendations.parquet", index=False)
    management.head(2000).to_csv(OUT_DIR / "management_map_v1_predictions_and_recommendations_sample.csv", index=False)

    global_rows = []
    for y, prefix in [
        ("Resistance", "Resistance"),
        ("T80_revised", "T80"),
        ("IRI_good_10yr", "IRI"),
        ("STAB_10yr", "STAB"),
    ]:
        for group in ["topo_soil", "forest", "human", "climate", "space"]:
            col = f"{prefix}__{group}"
            if col in management.columns:
                global_rows.append(
                    {
                        "y": y,
                        "group": group,
                        "mean_abs_ablation_effect": float(management[col].abs().mean()),
                    }
                )
    pd.DataFrame(global_rows).sort_values(["y", "mean_abs_ablation_effect"], ascending=[True, False]).to_csv(
        OUT_DIR / "management_map_v1_global_group_effects.csv", index=False
    )

    pd.DataFrame(
        {
            "management_priority_class": management["management_priority_class"],
            "integrated_dominant_actionable_group": management["integrated_dominant_actionable_group"],
            "management_recommendation_class": management["management_recommendation_class"],
        }
    ).value_counts().reset_index(name="count").to_csv(
        OUT_DIR / "management_map_v1_recommendation_counts.csv", index=False
    )

    notes = [
        "Management map v1",
        "This product uses the current best models for the four y variables:",
        "- Resistance: RF + x + y",
        "- T80: ordinal RF + x + y + x_sq_z + y_sq_z + xy_z",
        "- IRI_good_10yr: RF + x + y + x_sq_z + y_sq_z + xy_z",
        "- STAB_10yr: RF + x + y + x_sq_z + y_sq_z + xy_z",
        "",
        "Local driver interpretation uses grouped ablation, not linear coefficients.",
        "For each pixel, each variable group was replaced by a reference median and the prediction change was recorded.",
        "The largest absolute grouped ablation effect defines the local dominant driver group.",
        "",
        "Actionable groups: climate, forest, human, topo_soil.",
        "Space is retained for diagnosis but excluded from the integrated actionable group map.",
        "",
        "Management need index = mean of four normalized risk components:",
        "- low predicted Resistance",
        "- high predicted T80",
        "- low predicted IRI_good_10yr",
        "- low predicted STAB_10yr",
        "",
        "Important caveat: recovery models were trained on pixels with observed 10-year windows and then predicted across the full table.",
    ]
    (OUT_DIR / "management_map_v1_notes.txt").write_text("\n".join(notes))

    run_info = {
        "n_pixels_output": int(len(management)),
        "columns_output": int(len(management.columns)),
        "actionable_groups": actionable_groups,
    }
    (OUT_DIR / "management_map_v1_run_info.json").write_text(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()
