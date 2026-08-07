#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = (
    ROOT
    / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
TODAY = date.today().strftime("%Y-%m-%d")
OUT_DIR = ROOT / f"spatial_block_validation_best_nosev_variants_{TODAY}"
OUT_SUMMARY = OUT_DIR / "best_nosev_block_validation_summary.csv"
OUT_JSON = OUT_DIR / "best_nosev_block_validation_summary.json"
OUT_REPORT = OUT_DIR / "best_nosev_block_validation_report.md"

RESPONSES = ["Resistance", "T80", "IRI_good_pow2", "STAB_good_pow2"]
RECOVERY_RESPONSES = {"T80", "IRI_good_pow2", "STAB_good_pow2"}
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
BLOCK_KM = 100

FIRE_REGIME_MAP = {
    7053: "surface_fire", 7054: "surface_fire", 7031: "surface_fire",
    7017: "surface_fire", 7019: "surface_fire", 7020: "surface_fire",
    7022: "surface_fire", 7035: "surface_fire", 7036: "surface_fire",
    7063: "surface_fire",
    7050: "stand_replacing", 7055: "stand_replacing", 7056: "stand_replacing",
    7046: "stand_replacing", 7041: "stand_replacing", 7044: "stand_replacing",
    7032: "stand_replacing", 7033: "stand_replacing", 7058: "stand_replacing",
    7061: "stand_replacing", 7062: "stand_replacing", 7113: "stand_replacing",
    7114: "stand_replacing", 7118: "stand_replacing",
    7045: "mixed_severity", 7047: "mixed_severity", 7166: "mixed_severity",
    7051: "mixed_severity", 7018: "mixed_severity", 7027: "mixed_severity",
    7028: "mixed_severity", 7172: "mixed_severity", 7030: "mixed_severity",
    7265: "mixed_severity",
    7037: "coastal_mixed", 7039: "coastal_mixed", 7042: "coastal_mixed",
    7174: "coastal_mixed", 7043: "coastal_mixed", 7014: "coastal_mixed",
    7015: "coastal_mixed",
    7011: "hardwood", 7029: "hardwood", 7010: "hardwood",
    7008: "hardwood", 7021: "hardwood",
}

BASE_PREDS = [
    "TS_elev_m_z", "TS_slope_deg_z", "TS_northness_z", "TS_eastness_z",
    "TS_twi_z", "TS_roughness_z", "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z", "FS_CBH_t0agg_z",
    "HUM_popdens_win10km_log_z", "HUM_roaddens_r5km_z", "HUM_traildens_r10km_z",
    "HUM_imperv_near_t0_z", "HUM_viirs_near_t0_log_z",
    "CLIM_pr_sum_pre_z", "CLIM_eto_sum_pre_z", "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z", "CLIM_aridity_pre_z", "CLIM_tmmx_std_pre_z",
    "CLIM_vpd_mean_pre_z", "CLIM_vpd_std_pre_z",
    "x", "y", "x_sq_z", "y_sq_z", "xy_z",
]
POST_PREDS = [
    "CLIM_pr_sum_post_z",
    "CLIM_tmmn_mean_post_z",
    "CLIM_aridity_post_z",
    "CLIM_tmmx_std_post_z",
]
EVT_PREDS = ["FS_EVT_resistance_proxy_z", "FS_EVT_regeneration_proxy_z"]
SEV_PREDS = ["sev_z"]

BASE_TO_Z = {
    "TS_elev_m_z": "TS_elev_m",
    "TS_slope_deg_z": "TS_slope_deg",
    "TS_northness_z": "TS_northness",
    "TS_eastness_z": "TS_eastness",
    "TS_twi_z": "TS_twi",
    "TS_roughness_z": "TS_roughness",
    "TS_SOC_0_30cm_z": "TS_SOC_0_30cm",
    "FS_TCC_t0_z": "FS_TCC_t0",
    "FS_CBH_t0agg_z": "FS_CBH_t0agg",
    "HUM_roaddens_r5km_z": "HUM_roaddens_r5km",
    "HUM_traildens_r10km_z": "HUM_traildens_r10km",
    "HUM_imperv_near_t0_z": "HUM_imperv_near_t0",
    "CLIM_pr_sum_pre_z": "CLIM_pr_sum_pre",
    "CLIM_eto_sum_pre_z": "CLIM_eto_sum_pre",
    "CLIM_tmmn_mean_pre_z": "CLIM_tmmn_mean_pre",
    "CLIM_hot_days_35C_pre_z": "CLIM_hot_days_35C_pre",
    "CLIM_aridity_pre_z": "CLIM_aridity_pre",
    "CLIM_tmmx_std_pre_z": "CLIM_tmmx_std_pre",
    "CLIM_vpd_mean_pre_z": "CLIM_vpd_mean_pre",
    "CLIM_vpd_std_pre_z": "CLIM_vpd_std_pre",
    "CLIM_pr_sum_post_z": "CLIM_pr_sum_post",
    "CLIM_tmmn_mean_post_z": "CLIM_tmmn_mean_post",
    "CLIM_aridity_post_z": "CLIM_aridity_post",
    "CLIM_tmmx_std_post_z": "CLIM_tmmx_std_post",
    "FS_EVT_resistance_proxy_z": "FS_EVT_resistance_proxy",
    "FS_EVT_regeneration_proxy_z": "FS_EVT_regeneration_proxy",
    "sev_z": "sev",
}


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def log1p_z(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").astype(float).clip(lower=0)
    return zscore(np.log1p(clean))


def ensure_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    for z_col, raw_col in BASE_TO_Z.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    for col, expr in [
        ("x_sq_z", out["x"] ** 2),
        ("y_sq_z", out["y"] ** 2),
        ("xy_z", out["x"] * out["y"]),
    ]:
        if col not in out.columns:
            out[col] = zscore(expr)

    regime = out["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    dummies = pd.get_dummies(regime, prefix="EVT_regime").astype(np.float32)
    if "EVT_regime_other" in dummies.columns:
        dummies = dummies.drop(columns=["EVT_regime_other"])
    regime_cols = list(dummies.columns)
    out = pd.concat([out, dummies], axis=1)
    return out, regime_cols


def fit_predict(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def block_groups(df: pd.DataFrame) -> pd.Series:
    block_m = BLOCK_KM * 1000.0
    return pd.Series(
        [
            f"{int(np.floor(x / block_m))}_{int(np.floor(y / block_m))}"
            for x, y in zip(df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float))
        ],
        index=df.index,
    )


def build_variants(regime_cols: list[str], response: str) -> OrderedDict[str, dict]:
    is_recovery = response in RECOVERY_RESPONSES

    def maybe_post(preds: list[str], add_post: bool) -> list[str]:
        out = list(preds)
        if is_recovery and add_post:
            out += POST_PREDS
        return out

    return OrderedDict(
        [
            (
                "A_baseline_nosev",
                {
                    "label": "A: baseline + VPD + poly (no sev, no EVT)",
                    "role": "clean_baseline",
                    "predictors": maybe_post(BASE_PREDS, False),
                },
            ),
            (
                "C_EVTproxy_postclim",
                {
                    "label": "C: A + EVT proxy + postclim",
                    "role": "clean_candidate",
                    "predictors": maybe_post(BASE_PREDS + EVT_PREDS, True),
                },
            ),
            (
                "D_fireregime_postclim",
                {
                    "label": "D: A + fire-regime dummies + postclim",
                    "role": "clean_candidate",
                    "predictors": maybe_post(BASE_PREDS + regime_cols, True),
                },
            ),
            (
                "E_EVT_fireregime_postclim",
                {
                    "label": "E: A + EVT proxy + fire-regime dummies + postclim",
                    "role": "clean_candidate",
                    "predictors": maybe_post(BASE_PREDS + EVT_PREDS + regime_cols, True),
                },
            ),
            (
                "sev_ceiling",
                {
                    "label": "Ceiling: A + sev + EVT proxy (+postclim for recovery)",
                    "role": "predictive_ceiling",
                    "predictors": maybe_post(BASE_PREDS + SEV_PREDS + EVT_PREDS, True),
                },
            ),
        ]
    )


def evaluate_variant(df: pd.DataFrame, response: str, predictors: list[str]) -> dict:
    cols = list(dict.fromkeys([response, "x", "y", *predictors]))
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    X = work[predictors]
    y = work[response]
    groups = block_groups(work)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    random_pred = fit_predict(X_train, y_train, X_test)
    random_r2 = float(r2_score(y_test, random_pred))
    random_rmse = float(np.sqrt(mean_squared_error(y_test, random_pred)))

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    hold_pred = fit_predict(X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx])
    hold_r2 = float(r2_score(y.iloc[test_idx], hold_pred))
    hold_rmse = float(np.sqrt(mean_squared_error(y.iloc[test_idx], hold_pred)))

    return {
        "rows_used": int(len(work)),
        "n_predictors": int(len(predictors)),
        "n_groups": int(pd.Series(groups).nunique()),
        "random_r2": random_r2,
        "random_rmse": random_rmse,
        "block_holdout_r2": hold_r2,
        "block_holdout_rmse": hold_rmse,
        "block_holdout_delta_vs_random": hold_r2 - random_r2,
    }


def write_outputs(rows: list[dict], summary_json: dict[str, object]) -> pd.DataFrame:
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df.to_csv(OUT_SUMMARY, index=False)
    OUT_JSON.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    return summary_df


def build_report(summary_df: pd.DataFrame) -> str:
    lines = [
        f"# Best No-Sev Spatial Block Validation ({TODAY})",
        "",
        f"- Input table: `{INPUT}`",
        f"- Responses: `{', '.join(RESPONSES)}`",
        f"- Block size: `{BLOCK_KM} km`",
        f"- Trees per RF: `{N_ESTIMATORS}`",
        "- Goal: compare clean variants across multiple response variables under matched random vs block validation.",
        "",
    ]
    for response in RESPONSES:
        sub = summary_df[summary_df["response"] == response].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("block_holdout_r2", ascending=False).reset_index(drop=True)
        baseline = sub.loc[sub["variant"] == "A_baseline_nosev"].iloc[0]
        lines += [
            f"## {response}",
            "",
            "| Variant | Role | Rows | Random R2 | 100 km holdout R2 | Holdout delta vs random | Delta vs A holdout |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in sub.iterrows():
            delta_vs_a = row["block_holdout_r2"] - baseline["block_holdout_r2"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["label"],
                        row["role"],
                        f"{int(row['rows_used']):,}",
                        f"{row['random_r2']:.4f}",
                        f"{row['block_holdout_r2']:.4f}",
                        f"{row['block_holdout_delta_vs_random']:+.4f}",
                        f"{delta_vs_a:+.4f}",
                    ]
                )
                + " |"
            )
        clean_sub = sub[sub["role"] == "clean_candidate"]
        if not clean_sub.empty:
            best_clean = clean_sub.iloc[0]
            lines += [
                "",
                f"- Best clean variant for `{response}`: `{best_clean['variant']}` "
                f"with block holdout R²=`{best_clean['block_holdout_r2']:.4f}`.",
                "",
            ]

    lines += [
        "## Interpretation notes",
        "",
        "- This is a matched fast-screen block comparison using 100-tree RF for all variants.",
        "- `sev_ceiling` is a predictive upper bound only; it should not be interpreted as the clean mechanism model.",
        "- Row counts differ where EVT proxy columns are missing; compare variants with that sample-size shift in mind.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(INPUT).copy()
    if "Forest_at_t0" in raw.columns:
        raw = raw[raw["Forest_at_t0"] == 1].reset_index(drop=True)
    df, regime_cols = ensure_columns(raw)

    rows: list[dict] = []
    summary_json: dict[str, object] = {
        "input_table": str(INPUT),
        "responses": RESPONSES,
        "block_km": BLOCK_KM,
        "n_estimators": N_ESTIMATORS,
        "results": {},
    }

    total = sum(len(build_variants(regime_cols, response)) for response in RESPONSES)
    done = 0
    for response in RESPONSES:
        variants = build_variants(regime_cols, response)
        for variant_name, cfg in variants.items():
            done += 1
            print(f"[{done}/{total}] {response} | {variant_name} ...", flush=True)
            metrics = evaluate_variant(df, response, cfg["predictors"])
            row = {
                "response": response,
                "variant": variant_name,
                "label": cfg["label"],
                "role": cfg["role"],
                **metrics,
            }
            rows.append(row)
            summary_json["results"].setdefault(response, {})[variant_name] = row
            write_outputs(rows, summary_json)
            print(
                f"    rows={metrics['rows_used']:,} | "
                f"random_R2={metrics['random_r2']:.4f} | "
                f"block_holdout_R2={metrics['block_holdout_r2']:.4f}",
                flush=True,
            )

    summary_df = write_outputs(rows, summary_json)
    report = build_report(summary_df)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
