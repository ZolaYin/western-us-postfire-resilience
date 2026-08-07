from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/"
    "recovery_model_execution_near_t0_postfire"
)
INPUT_PATH = BASE_DIR / "MGWR_ready_table_increment_recovery_near_t0_postfire.parquet"
OUT_DIR = BASE_DIR / "increment_grouped_ablation_2026-03-31"
RANDOM_STATE = 42
N_ESTIMATORS = 200


def group_map() -> dict[str, list[str]]:
    return {
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
        "climate": [
            "CLIM_pr_sum_post_z",
            "CLIM_eto_sum_post_z",
            "CLIM_tmmn_mean_post_z",
            "CLIM_hot_days_35C_post_z",
            "CLIM_aridity_post_z",
            "CLIM_tmmx_std_post_z",
        ],
        "space": ["x", "y", "x_sq_z", "y_sq_z", "xy_z"],
    }


def predictors() -> list[str]:
    return (
        group_map()["topo_soil"]
        + group_map()["forest"]
        + group_map()["human"]
        + group_map()["climate"]
        + group_map()["space"]
    )


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def group_ablation_contributions(model, X_full: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    base_pred = predict_model(model, X_full)
    ref = X_full.median(numeric_only=True)
    out = pd.DataFrame(index=X_full.index)
    for group, cols in groups.items():
        X_alt = X_full.copy()
        for col in cols:
            if col in X_alt.columns:
                X_alt[col] = ref[col]
        alt_pred = predict_model(model, X_alt)
        out[group] = base_pred - alt_pred
    return out


def fit_regressor(df: pd.DataFrame, response: str, preds: list[str]) -> tuple[RandomForestRegressor, pd.DataFrame]:
    cols = list(dict.fromkeys([response] + preds + ["pixel_id", "x", "y", "t0_year"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(work[preds], work[response])
    return model, work


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT_PATH)
    preds = predictors()
    groups = group_map()

    all_global_rows = []
    all_count_rows = []

    for response in ["INC_end_rel_10obs", "INC_cum_rel_10obs"]:
        model, work = fit_regressor(df, response, preds)
        contrib = group_ablation_contributions(model, work[preds], groups)
        pred = predict_model(model, work[preds])
        out = pd.concat([work[["pixel_id", "x", "y", "t0_year", response]].copy(), pd.Series(pred, index=work.index, name=f"pred_{response}"), contrib.add_prefix(f"{response}__")], axis=1)

        abs_df = contrib.abs().copy()
        out["dominant_group_all"] = abs_df.idxmax(axis=1)
        actionable = abs_df.drop(columns=["space"], errors="ignore")
        out["dominant_group_actionable"] = actionable.idxmax(axis=1)

        response_dir = OUT_DIR / response
        response_dir.mkdir(parents=True, exist_ok=True)
        out.to_parquet(response_dir / f"{response}_grouped_ablation.parquet", index=False)

        global_effects = abs_df.mean().reset_index()
        global_effects.columns = ["group", "mean_abs_effect"]
        global_effects["response"] = response
        global_effects.to_csv(response_dir / f"{response}_global_group_effects.csv", index=False)
        all_global_rows.append(global_effects)

        counts = out["dominant_group_actionable"].value_counts(dropna=False).reset_index()
        counts.columns = ["dominant_group_actionable", "n"]
        counts["response"] = response
        counts.to_csv(response_dir / f"{response}_dominant_group_counts.csv", index=False)
        all_count_rows.append(counts)

        notes = {
            "response": response,
            "n_rows_used": int(len(work)),
            "predictors": preds,
            "global_group_effects_ordered": global_effects.sort_values("mean_abs_effect", ascending=False).to_dict(orient="records"),
        }
        (response_dir / f"{response}_grouped_ablation_notes.json").write_text(json.dumps(notes, indent=2))

    pd.concat(all_global_rows, ignore_index=True).to_csv(OUT_DIR / "increment_grouped_ablation_global_effects_all.csv", index=False)
    pd.concat(all_count_rows, ignore_index=True).to_csv(OUT_DIR / "increment_grouped_ablation_counts_all.csv", index=False)

    lines = ["Increment recovery grouped ablation summary"]
    for response in ["INC_end_rel_10obs", "INC_cum_rel_10obs"]:
        sub = pd.concat(all_global_rows, ignore_index=True)
        sub = sub[sub["response"] == response].sort_values("mean_abs_effect", ascending=False)
        top = ", ".join(f"{r.group}={r.mean_abs_effect:.4f}" for r in sub.itertuples(index=False))
        lines.append(f"{response}: {top}")
    (OUT_DIR / "increment_grouped_ablation_summary.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
