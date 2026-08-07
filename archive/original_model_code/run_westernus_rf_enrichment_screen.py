#!/usr/bin/env python3
"""
WestUS RF enrichment screen — improve test R² across all 4 response variables.

Tests systematic additions to the current baseline predictor set:
  1. Fire severity (sev)
  2. EVT proxy (resistance + regeneration)
  3. VPD pre-fire (vpd_mean_pre, vpd_std_pre)
  4. Post-fire climate (for recovery variables T80/IRI/STAB)
  5. Polynomial x,y terms (x², y², xy)
  6. Combined best enrichment

Output: westernus_rf_enrichment_screen_<date>/
  enrichment_screen_results.csv
  enrichment_screen_report.md
  feature_importance_<variant>_<response>.csv
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
OUT_DIR = ROOT / f"westernus_rf_enrichment_screen_{TODAY}"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 300  # fast but reliable; bump to 500 for final run

# ── Baseline predictor set (current best WestUS model) ──────────────────────
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

# ── New predictor groups ──────────────────────────────────────────────────────
SEV_PREDS = ["sev_z"]
EVT_PREDS = ["FS_EVT_resistance_proxy_z", "FS_EVT_regeneration_proxy_z"]
VPD_PREDS = ["CLIM_vpd_mean_pre_z", "CLIM_vpd_std_pre_z"]
POST_PREDS = [
    "CLIM_pr_sum_post_z",
    "CLIM_tmmn_mean_post_z",
    "CLIM_aridity_post_z",
    "CLIM_tmmx_std_post_z",
]
POLY_PREDS = ["x_sq_z", "y_sq_z", "xy_z"]

# ── Response variables ────────────────────────────────────────────────────────
# Resistance uses pre-fire only; T80/IRI/STAB can use post-fire climate
ALL_RESPONSES = ["Resistance", "T80", "IRI_good_pow2", "STAB_good_pow2"]
RECOVERY_RESPONSES = {"T80", "IRI_good_pow2", "STAB_good_pow2"}


# ── Variant definitions ───────────────────────────────────────────────────────
# Each variant adds a tuple of predictor groups on top of the baseline.
# For recovery responses the post-fire group is also included where noted.
VARIANTS: dict[str, dict] = {
    "baseline": {
        "extra": [],
        "add_post_for_recovery": False,
        "label": "Baseline (current 22 preds)",
    },
    "+sev": {
        "extra": SEV_PREDS,
        "add_post_for_recovery": False,
        "label": "+fire severity",
    },
    "+EVT": {
        "extra": EVT_PREDS,
        "add_post_for_recovery": False,
        "label": "+EVT proxy (resist+regen)",
    },
    "+VPD": {
        "extra": VPD_PREDS,
        "add_post_for_recovery": False,
        "label": "+VPD pre-fire",
    },
    "+postclim": {
        "extra": [],           # empty: POST_PREDS added only for recovery via flag below
        "add_post_for_recovery": True,
        "label": "+post-fire climate (recovery only)",
    },
    "+poly": {
        "extra": POLY_PREDS,
        "add_post_for_recovery": False,
        "label": "+polynomial x,y",
    },
    "+sev+EVT+VPD": {
        "extra": SEV_PREDS + EVT_PREDS + VPD_PREDS,
        "add_post_for_recovery": False,
        "label": "+sev+EVT+VPD",
    },
    "+sev+EVT+VPD+poly": {
        "extra": SEV_PREDS + EVT_PREDS + VPD_PREDS + POLY_PREDS,
        "add_post_for_recovery": False,
        "label": "+sev+EVT+VPD+poly",
    },
    "full_enrichment": {
        "extra": SEV_PREDS + EVT_PREDS + VPD_PREDS + POLY_PREDS,
        "add_post_for_recovery": True,
        "label": "Full enrichment (all new preds)",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    """Add z-scored / transformed columns that are not already in the table."""
    out = df.copy()

    # Standard z-scores from raw columns
    z_map = {
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
        # New additions
        "sev_z": "sev",
        "FS_EVT_resistance_proxy_z": "FS_EVT_resistance_proxy",
        "FS_EVT_regeneration_proxy_z": "FS_EVT_regeneration_proxy",
        "CLIM_vpd_mean_pre_z": "CLIM_vpd_mean_pre",
        "CLIM_vpd_std_pre_z": "CLIM_vpd_std_pre",
        "CLIM_pr_sum_post_z": "CLIM_pr_sum_post",
        "CLIM_tmmn_mean_post_z": "CLIM_tmmn_mean_post",
        "CLIM_aridity_post_z": "CLIM_aridity_post",
        "CLIM_tmmx_std_post_z": "CLIM_tmmx_std_post",
    }
    for z_col, raw_col in z_map.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])

    # Log-z transforms
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])

    # Polynomial spatial terms (from raw x,y — standardise after squaring)
    if "x_sq_z" not in out.columns:
        out["x_sq_z"] = zscore(out["x"] ** 2)
    if "y_sq_z" not in out.columns:
        out["y_sq_z"] = zscore(out["y"] ** 2)
    if "xy_z" not in out.columns:
        out["xy_z"] = zscore(out["x"] * out["y"])

    return out


def run_rf(
    df: pd.DataFrame,
    response: str,
    predictors: list[str],
) -> dict:
    """Fit RF with train/test split; return metrics dict."""
    needed = [response] + predictors
    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 200:
        return {"rows_used": len(work), "test_r2": float("nan"), "test_rmse": float("nan")}

    X = work[predictors]
    y = work[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "rows_used": int(len(work)),
        "test_r2": float(r2_score(y_test, pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "model": model,
        "predictors": predictors,
    }


def build_predictor_list(variant_cfg: dict, is_recovery: bool) -> list[str]:
    preds = list(BASELINE_PREDS)
    preds += variant_cfg["extra"]
    if is_recovery and variant_cfg["add_post_for_recovery"]:
        preds += POST_PREDS
    return preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT} ...")
    df = ensure_columns(pd.read_parquet(INPUT).copy())
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Verify new columns exist
    for col in SEV_PREDS + EVT_PREDS + VPD_PREDS + POST_PREDS + POLY_PREDS:
        n_valid = df[col].notna().sum() if col in df.columns else 0
        print(f"  {col}: {n_valid:,} valid rows")

    rows = []
    importance_dfs = {}

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

            if "model" in result and not np.isnan(result["test_r2"]):
                imp_df = (
                    pd.DataFrame({
                        "predictor": result["predictors"],
                        "importance": result["model"].feature_importances_,
                    })
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True)
                )
                key = f"{variant_name}__{response}"
                importance_dfs[key] = imp_df
                # Save importance for the best-looking variants
                if variant_name in ("+sev+EVT+VPD+poly", "full_enrichment"):
                    imp_path = OUT_DIR / f"importance_{variant_name.lstrip('+').replace('+','_')}__{response}.csv"
                    imp_df.to_csv(imp_path, index=False)

            print(
                f"  [{done}/{total}] {variant_name} | {response} | "
                f"rows={result['rows_used']:,} | test_R²={result['test_r2']:.4f}"
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "enrichment_screen_results.csv", index=False)

    # ── Build report ─────────────────────────────────────────────────────────
    # Pivot: rows = variants, cols = responses, values = test R²
    pivot = results_df.pivot_table(
        index=["variant", "variant_label"],
        columns="response",
        values="test_r2",
    ).reset_index()

    resp_order = [r for r in ALL_RESPONSES if r in pivot.columns]
    pivot = pivot.sort_values("Resistance", ascending=False)

    # Baseline reference values
    baseline_row = results_df[results_df["variant"] == "baseline"]
    baseline_r2 = baseline_row.set_index("response")["test_r2"].to_dict()

    lines = [
        f"# WestUS RF Enrichment Screen ({TODAY})",
        "",
        f"- n_estimators: {N_ESTIMATORS}",
        f"- Input: `westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet`",
        "",
        "## Baseline reference (current model)",
        "",
    ]
    for resp in ALL_RESPONSES:
        row = baseline_row[baseline_row["response"] == resp].iloc[0]
        lines.append(
            f"- `{resp}`: test R²=`{row['test_r2']:.4f}`, RMSE=`{row['test_rmse']:.4f}`, n=`{row['rows_used']:,}`"
        )

    lines += [
        "",
        "## Test R² by variant and response",
        "",
        "| Variant | " + " | ".join(resp_order) + " |",
        "|" + "---|" * (len(resp_order) + 1),
    ]
    for _, r in pivot.iterrows():
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
        "## RMSE by variant and response",
        "",
        "| Variant | " + " | ".join(resp_order) + " |",
        "|" + "---|" * (len(resp_order) + 1),
    ]
    rmse_pivot = results_df.pivot_table(
        index=["variant", "variant_label"],
        columns="response",
        values="test_rmse",
    ).reset_index()
    rmse_pivot = rmse_pivot.loc[rmse_pivot["variant"].isin(pivot["variant"])].set_index("variant").loc[pivot["variant"]].reset_index()

    for _, r in rmse_pivot.iterrows():
        cells = [results_df.loc[results_df["variant"] == r["variant"], "variant_label"].iloc[0]]
        for resp in resp_order:
            val = r.get(resp, float("nan"))
            cells.append(f"{val:.4f}" if not np.isnan(val) else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Predictor counts per variant",
        "",
        "| Variant | n_preds (Resistance) | n_preds (T80/IRI/STAB) |",
        "|---|---|---|",
    ]
    for variant_name, vcfg in VARIANTS.items():
        n_resi = len(build_predictor_list(vcfg, False))
        n_recov = len(build_predictor_list(vcfg, True))
        lines.append(f"| {vcfg['label']} | {n_resi} | {n_recov} |")

    lines += [
        "",
        "## Best variant per response",
        "",
    ]
    for resp in ALL_RESPONSES:
        sub = results_df[results_df["response"] == resp].sort_values("test_r2", ascending=False)
        best = sub.iloc[0]
        lines.append(
            f"- **{resp}**: `{best['variant']}` — test R²=`{best['test_r2']:.4f}` "
            f"(+`{best['test_r2'] - baseline_r2[resp]:.4f}` vs baseline), "
            f"n_preds={best['n_predictors']}, rows={best['rows_used']:,}"
        )

    lines += ["", "## Rows available per response", ""]
    for resp in ALL_RESPONSES:
        n = df[resp].notna().sum()
        lines.append(f"- `{resp}`: `{n:,}` non-null rows")

    report_text = "\n".join(lines) + "\n"
    (OUT_DIR / "enrichment_screen_report.md").write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Results CSV: {OUT_DIR / 'enrichment_screen_results.csv'}")


if __name__ == "__main__":
    main()
