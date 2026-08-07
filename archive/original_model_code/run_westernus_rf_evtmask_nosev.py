#!/usr/bin/env python3
"""
WestUS RF enrichment — EVT forest mask + no sev.

Two changes from previous enrichment screen:
  1. Forest mask: keep only EVT-forest pixels
     (FS_EVT_group_class in {'conifer', 'mixed', 'deciduous'})
     This replaces the NLCD-based Forest_at_t0 which was all-1 and did nothing.
  2. Remove 'sev' (fire severity) from all variants — endogenous predictor.

Variant structure (no sev):
  baseline       : current 22 preds, EVT-forest pixels only
  +EVT           : + EVT proxy (resist+regen)
  +VPD           : + vpd_mean_pre + vpd_std_pre
  +postclim      : + post-fire climate (recovery vars only)
  +poly          : + polynomial x,y
  +EVT+VPD       : combined
  +EVT+VPD+poly  : combined + poly
  full_nosev     : EVT+VPD+poly + postclim for recovery

Output: westernus_rf_evtmask_nosev_<date>/
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
OUT_DIR = ROOT / f"westernus_rf_evtmask_nosev_{TODAY}"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500

# EVT forest mask: keep only these group classes
EVT_FOREST_CLASSES = {"conifer", "mixed", "deciduous"}

# ── Baseline predictor set (current, no sev) ─────────────────────────────────
BASELINE_PREDS = [
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

EVT_PREDS   = ["FS_EVT_resistance_proxy_z", "FS_EVT_regeneration_proxy_z"]
VPD_PREDS   = ["CLIM_vpd_mean_pre_z", "CLIM_vpd_std_pre_z"]
POST_PREDS  = ["CLIM_pr_sum_post_z", "CLIM_tmmn_mean_post_z",
               "CLIM_aridity_post_z", "CLIM_tmmx_std_post_z"]
POLY_PREDS  = ["x_sq_z", "y_sq_z", "xy_z"]

ALL_RESPONSES      = ["Resistance", "T80", "IRI_good_pow2", "STAB_good_pow2"]
RECOVERY_RESPONSES = {"T80", "IRI_good_pow2", "STAB_good_pow2"}

VARIANTS: dict[str, dict] = {
    "baseline": {
        "extra": [],
        "add_post_for_recovery": False,
        "label": "Baseline (22 preds, no sev, EVT-forest mask)",
    },
    "+EVT": {
        "extra": EVT_PREDS,
        "add_post_for_recovery": False,
        "label": "+EVT proxy",
    },
    "+VPD": {
        "extra": VPD_PREDS,
        "add_post_for_recovery": False,
        "label": "+VPD pre-fire",
    },
    "+postclim": {
        "extra": [],
        "add_post_for_recovery": True,
        "label": "+post-fire climate (recovery only)",
    },
    "+poly": {
        "extra": POLY_PREDS,
        "add_post_for_recovery": False,
        "label": "+polynomial x,y",
    },
    "+EVT+VPD": {
        "extra": EVT_PREDS + VPD_PREDS,
        "add_post_for_recovery": False,
        "label": "+EVT+VPD",
    },
    "+EVT+VPD+poly": {
        "extra": EVT_PREDS + VPD_PREDS + POLY_PREDS,
        "add_post_for_recovery": False,
        "label": "+EVT+VPD+poly",
    },
    "full_nosev": {
        "extra": EVT_PREDS + VPD_PREDS + POLY_PREDS,
        "add_post_for_recovery": True,
        "label": "Full (no sev): EVT+VPD+poly+postclim",
    },
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


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z_map = {
        "TS_elev_m_z":            "TS_elev_m",
        "TS_slope_deg_z":         "TS_slope_deg",
        "TS_northness_z":         "TS_northness",
        "TS_eastness_z":          "TS_eastness",
        "TS_twi_z":               "TS_twi",
        "TS_roughness_z":         "TS_roughness",
        "TS_SOC_0_30cm_z":        "TS_SOC_0_30cm",
        "FS_TCC_t0_z":            "FS_TCC_t0",
        "FS_CBH_t0agg_z":         "FS_CBH_t0agg",
        "HUM_roaddens_r5km_z":    "HUM_roaddens_r5km",
        "HUM_traildens_r10km_z":  "HUM_traildens_r10km",
        "HUM_imperv_near_t0_z":   "HUM_imperv_near_t0",
        "CLIM_pr_sum_pre_z":      "CLIM_pr_sum_pre",
        "CLIM_eto_sum_pre_z":     "CLIM_eto_sum_pre",
        "CLIM_tmmn_mean_pre_z":   "CLIM_tmmn_mean_pre",
        "CLIM_hot_days_35C_pre_z":"CLIM_hot_days_35C_pre",
        "CLIM_aridity_pre_z":     "CLIM_aridity_pre",
        "CLIM_tmmx_std_pre_z":    "CLIM_tmmx_std_pre",
        # EVT proxies
        "FS_EVT_resistance_proxy_z":  "FS_EVT_resistance_proxy",
        "FS_EVT_regeneration_proxy_z":"FS_EVT_regeneration_proxy",
        # VPD
        "CLIM_vpd_mean_pre_z":    "CLIM_vpd_mean_pre",
        "CLIM_vpd_std_pre_z":     "CLIM_vpd_std_pre",
        # Post-fire climate
        "CLIM_pr_sum_post_z":     "CLIM_pr_sum_post",
        "CLIM_tmmn_mean_post_z":  "CLIM_tmmn_mean_post",
        "CLIM_aridity_post_z":    "CLIM_aridity_post",
        "CLIM_tmmx_std_post_z":   "CLIM_tmmx_std_post",
    }
    for z_col, raw_col in z_map.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])

    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])

    if "x_sq_z" not in out.columns:
        out["x_sq_z"] = zscore(out["x"] ** 2)
    if "y_sq_z" not in out.columns:
        out["y_sq_z"] = zscore(out["y"] ** 2)
    if "xy_z" not in out.columns:
        out["xy_z"] = zscore(out["x"] * out["y"])

    return out


def apply_evt_forest_mask(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["FS_EVT_group_class"].isin(EVT_FOREST_CLASSES)
    filtered = df[mask].copy()
    return filtered


def build_predictor_list(vcfg: dict, is_recovery: bool) -> list[str]:
    preds = list(BASELINE_PREDS)
    preds += vcfg["extra"]
    if is_recovery and vcfg["add_post_for_recovery"]:
        preds += POST_PREDS
    return preds


def run_rf(df: pd.DataFrame, response: str, predictors: list[str]) -> dict:
    needed = [response] + predictors
    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 200:
        return {"rows_used": len(work), "test_r2": float("nan"),
                "test_rmse": float("nan"), "feature_importances": None}

    X = work[predictors]
    y = work[response]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "rows_used": int(len(work)),
        "test_r2": float(r2_score(y_te, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "feature_importances": model.feature_importances_.tolist(),
        "predictors": list(predictors),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT.name} ...")
    raw = pd.read_parquet(INPUT)
    print(f"  Raw rows: {len(raw):,}")

    # --- EVT forest mask ---
    forest_df = apply_evt_forest_mask(raw)
    print(f"  After EVT forest mask: {len(forest_df):,} rows")
    print(f"  EVT class counts in masked data:")
    for cls, cnt in forest_df["FS_EVT_group_class"].value_counts().items():
        print(f"    {cls}: {cnt:,}")

    df = ensure_columns(forest_df)

    # Response availability after mask
    print("\n  Response availability after EVT mask:")
    for resp in ALL_RESPONSES:
        n = df[resp].notna().sum()
        print(f"    {resp}: {n:,}")

    rows = []
    total = len(VARIANTS) * len(ALL_RESPONSES)
    done = 0

    for variant_name, vcfg in VARIANTS.items():
        for response in ALL_RESPONSES:
            is_recovery = response in RECOVERY_RESPONSES
            preds = build_predictor_list(vcfg, is_recovery)

            result = run_rf(df, response, preds)
            done += 1

            row = {
                "variant": variant_name,
                "variant_label": vcfg["label"],
                "response": response,
                "n_predictors": len(preds),
                "rows_used": result["rows_used"],
                "test_r2": result["test_r2"],
                "test_rmse": result["test_rmse"],
            }
            rows.append(row)

            # Save feature importance for key variants
            if result["feature_importances"] is not None and \
               variant_name in ("+EVT+VPD+poly", "full_nosev"):
                imp_df = pd.DataFrame({
                    "predictor": result["predictors"],
                    "importance": result["feature_importances"],
                }).sort_values("importance", ascending=False).reset_index(drop=True)
                tag = variant_name.lstrip("+").replace("+", "_")
                imp_df.to_csv(
                    OUT_DIR / f"importance_{tag}__{response}.csv", index=False
                )

            print(
                f"  [{done}/{total}] {variant_name} | {response} | "
                f"rows={result['rows_used']:,} | test_R²={result['test_r2']:.4f}"
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "evtmask_nosev_results.csv", index=False)

    # Build pivot table
    pivot_r2 = results_df.pivot_table(
        index=["variant", "variant_label"], columns="response", values="test_r2"
    ).reset_index()
    pivot_rmse = results_df.pivot_table(
        index=["variant", "variant_label"], columns="response", values="test_rmse"
    ).reset_index()

    resp_order = [r for r in ALL_RESPONSES if r in pivot_r2.columns]
    baseline_r2 = (
        results_df[results_df["variant"] == "baseline"]
        .set_index("response")["test_r2"]
        .to_dict()
    )

    # Sort by Resistance R²
    pivot_r2 = pivot_r2.sort_values("Resistance", ascending=False)

    lines = [
        f"# WestUS RF: EVT Forest Mask + No Sev ({TODAY})",
        "",
        f"- n_estimators: {N_ESTIMATORS}",
        f"- **Forest mask**: EVT group class in {{conifer, mixed, deciduous}}",
        f"- **sev removed**: endogenous predictor excluded from all variants",
        f"- EVT-forest rows (Resistance): {forest_df['Resistance'].notna().sum():,}",
        f"- EVT-forest rows (T80/IRI/STAB): {forest_df['T80'].notna().sum():,}",
        "",
        "## EVT class breakdown (after mask)",
        "",
    ]
    for cls, cnt in forest_df["FS_EVT_group_class"].value_counts().items():
        lines.append(f"- `{cls}`: {cnt:,}")

    lines += [
        "",
        "## Baseline reference (EVT-masked, no sev)",
        "",
    ]
    for resp in ALL_RESPONSES:
        b = results_df[(results_df["variant"] == "baseline") &
                       (results_df["response"] == resp)].iloc[0]
        lines.append(
            f"- `{resp}`: test R²=`{b['test_r2']:.4f}`, "
            f"RMSE=`{b['test_rmse']:.4f}`, n=`{b['rows_used']:,}`"
        )

    lines += [
        "",
        "## Test R² by variant (sorted by Resistance)",
        "",
        "| Variant | " + " | ".join(resp_order) + " |",
        "|" + "---|" * (len(resp_order) + 1),
    ]
    for _, r in pivot_r2.iterrows():
        cells = [r["variant_label"]]
        for resp in resp_order:
            val = r.get(resp, float("nan"))
            delta = val - baseline_r2.get(resp, float("nan"))
            if np.isnan(val):
                cells.append("—")
            else:
                sign = "+" if delta >= 0 else ""
                cells.append(f"{val:.4f} ({sign}{delta:.4f})")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Best variant per response",
        "",
    ]
    for resp in ALL_RESPONSES:
        sub = results_df[results_df["response"] == resp].sort_values(
            "test_r2", ascending=False
        )
        best = sub.iloc[0]
        delta = best["test_r2"] - baseline_r2.get(resp, float("nan"))
        lines.append(
            f"- **{resp}**: `{best['variant']}` — "
            f"test R²=`{best['test_r2']:.4f}` (+`{delta:.4f}` vs EVT-masked baseline), "
            f"n_preds={best['n_predictors']}, rows={best['rows_used']:,}"
        )

    report = "\n".join(lines) + "\n"
    (OUT_DIR / "evtmask_nosev_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
