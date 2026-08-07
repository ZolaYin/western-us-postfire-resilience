from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split


ROOT = Path(
    "/path/to/google-drive/"
    "我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
RECOVERY_DIR = ROOT / "recovery_model_execution_near_t0_postfire"
RESISTANCE_DIR = ROOT / "resistance_model_execution_near_t0_aggregated"
OUT_DIR = RECOVERY_DIR / "management_stability_validation_2026-04-01"

T80_CLASSES = list(range(2, 11))
N_ESTIMATORS = 200

RUN_CONFIGS = [
    {"run_id": "run01", "seed": 11, "test_size": 0.20, "ablation_sample_n": 2000},
    {"run_id": "run02", "seed": 42, "test_size": 0.25, "ablation_sample_n": 3000},
    {"run_id": "run03", "seed": 101, "test_size": 0.30, "ablation_sample_n": 4000},
    {"run_id": "run04", "seed": 202, "test_size": 0.20, "ablation_sample_n": 5000},
    {"run_id": "run05", "seed": 303, "test_size": 0.30, "ablation_sample_n": 6000},
]

RESISTANCE_PREDICTORS = [
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

RECOVERY_BASE_PREDICTORS = [
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
T80_PREDICTORS = RECOVERY_BASE_PREDICTORS + ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]
IRI_STAB_PREDICTORS = RECOVERY_BASE_PREDICTORS + ["x", "y", "x_sq_z", "y_sq_z", "xy_z"]


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


def predict_model(model, X: pd.DataFrame, expected_year: bool = False) -> np.ndarray:
    if expected_year:
        probs = model.predict_proba(X)
        return probs @ np.array(T80_CLASSES, dtype=float)
    return model.predict(X)


def group_ablation_contributions(
    model,
    X_full: pd.DataFrame,
    group_map: dict[str, list[str]],
    expected_year: bool,
    ref_seed: int,
    ref_sample_n: int,
) -> pd.DataFrame:
    base_pred = predict_model(model, X_full, expected_year=expected_year)
    ref_df = X_full.sample(n=min(ref_sample_n, len(X_full)), random_state=ref_seed)
    ref = ref_df.median(numeric_only=True)
    out = pd.DataFrame(index=X_full.index)
    for group, cols in group_map.items():
        X_alt = X_full.copy()
        for col in cols:
            if col in X_alt.columns:
                X_alt[col] = ref[col]
        alt_pred = predict_model(model, X_alt, expected_year=expected_year)
        out[group] = base_pred - alt_pred
    return out


def fit_regressor(df: pd.DataFrame, response: str, predictors: list[str], seed: int, test_size: float):
    cols = list(dict.fromkeys([response] + predictors + ["pixel_id", "x", "y", "t0_year"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, _ = train_test_split(work.index, test_size=test_size, random_state=seed)
    train = work.loc[train_idx].copy()
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
    model.fit(train[predictors], train[response])
    return model, work


def fit_t80_classifier(df: pd.DataFrame, predictors: list[str], seed: int, test_size: float):
    cols = list(dict.fromkeys(["T80_revised"] + predictors + ["pixel_id", "x", "y", "t0_year"]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    train_idx, _ = train_test_split(
        work.index,
        test_size=test_size,
        random_state=seed,
        stratify=work["T80_revised"],
    )
    train = work.loc[train_idx].copy()
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=seed, n_jobs=-1)
    model.fit(train[predictors], train["T80_revised"])
    return model, work


def build_management_for_run(
    resistance_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    run_id: str,
    seed: int,
    test_size: float,
    ablation_sample_n: int,
) -> pd.DataFrame:
    resistance_model, resistance_work = fit_regressor(resistance_df, "Resistance", RESISTANCE_PREDICTORS, seed, test_size)
    t80_model, t80_work = fit_t80_classifier(recovery_df, T80_PREDICTORS, seed, test_size)
    iri_model, iri_work = fit_regressor(recovery_df, "IRI_good_10yr", IRI_STAB_PREDICTORS, seed, test_size)
    stab_model, stab_work = fit_regressor(recovery_df, "STAB_10yr", IRI_STAB_PREDICTORS, seed, test_size)

    base = recovery_df[["pixel_id", "x", "y", "t0_year"]].copy()
    management = base.copy()

    resistance_full = resistance_work.copy()
    resistance_full["pred_Resistance"] = predict_model(resistance_model, resistance_full[RESISTANCE_PREDICTORS])
    resistance_contrib = group_ablation_contributions(
        resistance_model,
        resistance_full[RESISTANCE_PREDICTORS],
        group_map_response("Resistance"),
        expected_year=False,
        ref_seed=seed,
        ref_sample_n=ablation_sample_n,
    )
    resistance_out = pd.concat(
        [resistance_full[["pixel_id", "pred_Resistance"]], resistance_contrib.add_prefix("Resistance__")], axis=1
    )

    t80_full = t80_work.copy()
    t80_full["pred_T80"] = predict_model(t80_model, t80_full[T80_PREDICTORS], expected_year=True)
    t80_contrib = group_ablation_contributions(
        t80_model,
        t80_full[T80_PREDICTORS],
        group_map_response("T80"),
        expected_year=True,
        ref_seed=seed,
        ref_sample_n=ablation_sample_n,
    )
    t80_out = pd.concat([t80_full[["pixel_id", "pred_T80"]], t80_contrib.add_prefix("T80__")], axis=1)

    iri_full = iri_work.copy()
    iri_full["pred_IRI_good_10yr"] = predict_model(iri_model, iri_full[IRI_STAB_PREDICTORS])
    iri_contrib = group_ablation_contributions(
        iri_model,
        iri_full[IRI_STAB_PREDICTORS],
        group_map_response("IRI"),
        expected_year=False,
        ref_seed=seed,
        ref_sample_n=ablation_sample_n,
    )
    iri_out = pd.concat([iri_full[["pixel_id", "pred_IRI_good_10yr"]], iri_contrib.add_prefix("IRI__")], axis=1)

    stab_full = stab_work.copy()
    stab_full["pred_STAB_10yr"] = predict_model(stab_model, stab_full[IRI_STAB_PREDICTORS])
    stab_contrib = group_ablation_contributions(
        stab_model,
        stab_full[IRI_STAB_PREDICTORS],
        group_map_response("STAB"),
        expected_year=False,
        ref_seed=seed,
        ref_sample_n=ablation_sample_n,
    )
    stab_out = pd.concat([stab_full[["pixel_id", "pred_STAB_10yr"]], stab_contrib.add_prefix("STAB__")], axis=1)

    for piece in [resistance_out, t80_out, iri_out, stab_out]:
        management = management.merge(piece, on="pixel_id", how="left")

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
    management["risk_resistance"] = 1.0 - percentile_rank(management["pred_Resistance"])
    management["risk_t80"] = percentile_rank(management["pred_T80"])
    management["risk_iri"] = 1.0 - percentile_rank(management["pred_IRI_good_10yr"])
    management["risk_stab"] = 1.0 - percentile_rank(management["pred_STAB_10yr"])
    management["management_need_index"] = management[["risk_resistance", "risk_t80", "risk_iri", "risk_stab"]].mean(axis=1)
    management["management_priority_class"] = classify_priority(management["management_need_index"]).astype(str)
    management["is_high_priority"] = management["management_priority_class"].isin(["High", "Very High"])
    management["run_id"] = run_id
    management["seed"] = seed
    management["test_size"] = test_size
    management["ablation_sample_n"] = ablation_sample_n

    return management[
        [
            "pixel_id",
            "x",
            "y",
            "t0_year",
            "run_id",
            "seed",
            "test_size",
            "ablation_sample_n",
            "management_need_index",
            "management_priority_class",
            "is_high_priority",
            "integrated_dominant_actionable_group",
        ]
    ].copy()


def pairwise_agreement(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    runs = sorted(df["run_id"].unique())
    rows = []
    for a, b in combinations(runs, 2):
        da = df[df["run_id"] == a][["pixel_id", value_col]].rename(columns={value_col: "a"})
        db = df[df["run_id"] == b][["pixel_id", value_col]].rename(columns={value_col: "b"})
        m = da.merge(db, on="pixel_id", how="inner").dropna()
        if len(m) == 0:
            continue
        agree = (m["a"] == m["b"]).mean()
        rows.append({"run_a": a, "run_b": b, "value_col": value_col, "n_common": int(len(m)), "agreement": float(agree)})
    return pd.DataFrame(rows)


def pairwise_jaccard_high_priority(df: pd.DataFrame) -> pd.DataFrame:
    runs = sorted(df["run_id"].unique())
    rows = []
    for a, b in combinations(runs, 2):
        da = df[df["run_id"] == a][["pixel_id", "is_high_priority"]].rename(columns={"is_high_priority": "a"})
        db = df[df["run_id"] == b][["pixel_id", "is_high_priority"]].rename(columns={"is_high_priority": "b"})
        m = da.merge(db, on="pixel_id", how="inner").dropna()
        inter = ((m["a"]) & (m["b"])).sum()
        union = ((m["a"]) | (m["b"])).sum()
        jaccard = float(inter / union) if union > 0 else np.nan
        rows.append({"run_a": a, "run_b": b, "n_common": int(len(m)), "jaccard_high_priority": jaccard})
    return pd.DataFrame(rows)


def per_pixel_stability(df: pd.DataFrame) -> pd.DataFrame:
    priority_pivot = df.pivot(index="pixel_id", columns="run_id", values="management_priority_class")
    high_pivot = df.pivot(index="pixel_id", columns="run_id", values="is_high_priority")
    dom_pivot = df.pivot(index="pixel_id", columns="run_id", values="integrated_dominant_actionable_group")

    rows = []
    for pid in priority_pivot.index:
        pvals = priority_pivot.loc[pid].dropna().astype(str)
        hvals = high_pivot.loc[pid].dropna().astype(bool)
        dvals = dom_pivot.loc[pid].dropna().astype(str)
        if len(pvals) == 0:
            continue
        p_mode = pvals.value_counts().idxmax()
        p_freq = float(pvals.value_counts().max() / len(pvals))
        h_rate = float(hvals.mean()) if len(hvals) else np.nan
        if len(dvals):
            d_mode = dvals.value_counts().idxmax()
            d_freq = float(dvals.value_counts().max() / len(dvals))
        else:
            d_mode = None
            d_freq = np.nan
        rows.append(
            {
                "pixel_id": pid,
                "priority_mode": p_mode,
                "priority_mode_freq": p_freq,
                "high_priority_rate": h_rate,
                "dominant_group_mode": d_mode,
                "dominant_group_mode_freq": d_freq,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resistance_df = pd.read_parquet(RESISTANCE_DIR / "MGWR_ready_table_near_t0_aggregated.parquet")
    recovery_df = add_spatial_terms(pd.read_parquet(RECOVERY_DIR / "MGWR_ready_table_t80_iri_stability_near_t0_postfire.parquet"))

    all_runs = []
    for cfg in RUN_CONFIGS:
        print(f"Running {cfg['run_id']} seed={cfg['seed']} test_size={cfg['test_size']} ablation_n={cfg['ablation_sample_n']}", flush=True)
        out = build_management_for_run(
            resistance_df,
            recovery_df,
            run_id=cfg["run_id"],
            seed=cfg["seed"],
            test_size=cfg["test_size"],
            ablation_sample_n=cfg["ablation_sample_n"],
        )
        out.to_parquet(OUT_DIR / f"{cfg['run_id']}_management_predictions.parquet", index=False)
        all_runs.append(out)

    all_df = pd.concat(all_runs, ignore_index=True)
    all_df.to_parquet(OUT_DIR / "management_stability_all_runs.parquet", index=False)

    priority_agree = pairwise_agreement(all_df, "management_priority_class")
    priority_agree.to_csv(OUT_DIR / "pairwise_priority_agreement.csv", index=False)
    dom_agree = pairwise_agreement(all_df, "integrated_dominant_actionable_group")
    dom_agree.to_csv(OUT_DIR / "pairwise_dominant_group_agreement.csv", index=False)
    high_jaccard = pairwise_jaccard_high_priority(all_df)
    high_jaccard.to_csv(OUT_DIR / "pairwise_high_priority_jaccard.csv", index=False)

    pixel_stab = per_pixel_stability(all_df)
    pixel_stab.to_csv(OUT_DIR / "per_pixel_stability_summary.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "mean_pairwise_priority_agreement",
                "value": float(priority_agree["agreement"].mean()),
            },
            {
                "metric": "mean_pairwise_dominant_group_agreement",
                "value": float(dom_agree["agreement"].mean()),
            },
            {
                "metric": "mean_pairwise_high_priority_jaccard",
                "value": float(high_jaccard["jaccard_high_priority"].mean()),
            },
            {
                "metric": "share_pixels_priority_mode_freq_ge_0.8",
                "value": float((pixel_stab["priority_mode_freq"] >= 0.8).mean()),
            },
            {
                "metric": "share_pixels_high_priority_rate_ge_0.8",
                "value": float((pixel_stab["high_priority_rate"] >= 0.8).mean()),
            },
            {
                "metric": "share_pixels_dominant_group_mode_freq_ge_0.8",
                "value": float((pixel_stab["dominant_group_mode_freq"] >= 0.8).mean()),
            },
        ]
    )
    summary.to_csv(OUT_DIR / "management_stability_summary.csv", index=False)

    notes = {
        "run_configs": RUN_CONFIGS,
        "interpretation": {
            "priority_agreement": "五级管理优先级在不同重跑之间的一致性",
            "high_priority_jaccard": "High+Very High 管理区的空间重叠度",
            "dominant_group_agreement": "主导 actionable group 在不同重跑之间的一致性",
            "mode_freq": "单个像元在多次重跑中被分到相同类别的频率",
        },
    }
    (OUT_DIR / "management_stability_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["Management stability validation", ""]
    for _, row in summary.iterrows():
        lines.append(f"{row['metric']}: {row['value']:.4f}")
    (OUT_DIR / "management_stability_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
