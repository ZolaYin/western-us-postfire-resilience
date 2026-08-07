#!/usr/bin/env python3
"""
WestUS RF — EVT-based fire-regime group dummies.

Key idea (user request):
  "用原始 EVT，因为有些树种就是会比较抗火"

Approach:
  - Apply EVT forest mask (conifer / mixed / deciduous only)
  - Map raw EVT2022 codes to fire-regime groups based on species fire traits
  - Create dummy variables for fire-regime groups
  - Compare three encoding strategies:
      A. fire_regime_dummies  (5-class ecological grouping)
      B. top_evt_dummies      (one-hot for top 12 EVT codes + "other")
      C. combined             (fire_regime + top_evt as predictors)
  - Baseline: no EVT at all (current best no-sev, EVT-masked)
  - All variants: no sev, +VPD+poly+postclim for recovery

Output: westernus_rf_evt_fireregime_<date>/
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
OUT_DIR = ROOT / f"westernus_rf_evt_fireregime_{TODAY}"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 500

EVT_FOREST_CLASSES = {"conifer", "mixed", "deciduous"}

# ── Fire-regime classification of EVT2022 codes ───────────────────────────────
# Based on LANDFIRE EVT names, SAF associations, and known fire ecology literature.
#
# surface_fire:   thick-barked, frequent low-severity, high intrinsic resistance
# mixed_severity: intermediate bark, variable fire severity, moderate resistance
# stand_replacing:thin-barked or high-elevation, crown/stand-replacing fire regime
# coastal_mixed:  maritime Pacific, wetter, mixed severity
# hardwood:       deciduous / oak / aspen, resprouting
# other:          all remaining codes

FIRE_REGIME_MAP = {
    # ── surface_fire (ponderosa / jeffrey / juniper / pinyon) ─────────────────
    7053: "surface_fire",   # N Rocky Mtn Ponderosa Pine Woodland & Savanna
    7054: "surface_fire",   # S Rocky Mtn Ponderosa Pine Woodland
    7031: "surface_fire",   # CA Montane Jeffrey Pine-(Ponderosa Pine) Woodland
    7017: "surface_fire",   # Columbia Plateau Western Juniper Woodland
    7019: "surface_fire",   # Great Basin Pinyon-Juniper Woodland
    7020: "surface_fire",   # Intermtn Basin Juniper Savanna (sparse)
    7022: "surface_fire",   # Southern Rockies Pinyon-Juniper Woodland
    7035: "surface_fire",   # N Rockies Ponderosa Pine-Douglas-fir (open)
    7036: "surface_fire",   # S Rockies Ponderosa Pine Savanna
    7063: "surface_fire",   # CA Interior Cypress Forest (fire-adapted)

    # ── stand_replacing (lodgepole / spruce-fir / subalpine / high-elev) ──────
    7050: "stand_replacing",  # Rocky Mtn Lodgepole Pine Forest
    7055: "stand_replacing",  # RM Subalpine Dry-Mesic Spruce-Fir Forest
    7056: "stand_replacing",  # RM Subalpine Mesic-Wet Spruce-Fir Forest
    7046: "stand_replacing",  # N Rocky Mtn Subalpine Woodland (whitebark pine)
    7041: "stand_replacing",  # N Pacific Mountain Hemlock Forest
    7044: "stand_replacing",  # N CA Mesic Subalpine Woodland (mountain hemlock)
    7032: "stand_replacing",  # Mediterranean CA Red Fir Forest
    7033: "stand_replacing",  # Mediterranean CA Subalpine Woodland
    7058: "stand_replacing",  # Sierra Nevada Subalpine Lodgepole Pine Forest
    7061: "stand_replacing",  # N Rocky Mtn Subalpine-Upper Montane Parkland
    7062: "stand_replacing",  # Rocky Mtn Alpine-Montane Wet Meadow/Parkland
    7113: "stand_replacing",  # S Rocky Mtn Subalpine-Montane Spruce-Fir
    7114: "stand_replacing",  # Columbia Plateau Ponderosa Pine Forest (but subalpine variant)
    7118: "stand_replacing",  # Rocky Mtn Lodgepole Pine-Engelmann Spruce-Subalpine Fir

    # ── mixed_severity (interior Douglas-fir / grand fir / sierra mixed) ──────
    7045: "mixed_severity",   # N Rocky Mtn Dry-Mesic Montane Mixed Conifer
    7047: "mixed_severity",   # N Rocky Mtn Mesic Montane Mixed Conifer (grand fir)
    7166: "mixed_severity",   # Middle Rocky Mtn Montane Douglas-fir
    7051: "mixed_severity",   # S Rocky Mtn Dry-Mesic Montane Mixed Conifer
    7018: "mixed_severity",   # E Cascades Mesic Montane Mixed Conifer (grand fir)
    7027: "mixed_severity",   # Mediterranean CA Dry-Mesic Mixed Conifer
    7028: "mixed_severity",   # Mediterranean CA Mesic Mixed Conifer (white fir)
    7172: "mixed_severity",   # Sierran-Intermontane Western White Pine-White Fir
    7030: "mixed_severity",   # Mediterranean CA Lower Montane Conifer
    7265: "mixed_severity",   # Northern Rocky Mtn Dry-Mesic Mixed Conifer (other)

    # ── coastal_mixed (Pacific maritime, wetter/closed canopy) ───────────────
    7037: "coastal_mixed",    # N Pacific Maritime Dry-Mesic Douglas-fir-W.Hemlock
    7039: "coastal_mixed",    # N Pacific Maritime Mesic-Wet Douglas-fir-W.Hemlock
    7042: "coastal_mixed",    # N Pacific Mesic Western Hemlock-Silver Fir
    7174: "coastal_mixed",    # N Pacific Dry-Mesic Silver Fir-Douglas-fir
    7043: "coastal_mixed",    # Mediterranean CA Mixed Evergreen Forest (tan oak/madrone)
    7014: "coastal_mixed",    # Central/S CA Mixed Evergreen Woodland
    7015: "coastal_mixed",    # CA Coastal Redwood Forest (very high resistance)

    # ── hardwood (deciduous / oak / aspen) ───────────────────────────────────
    7011: "hardwood",   # Rocky Mtn Aspen Forest and Woodland
    7029: "hardwood",   # Mediterranean CA Mixed Oak Woodland
    7010: "hardwood",   # Klamath-Siskiyou Lower Montane Serpentine Mixed Conifer-Hardwood
    7008: "hardwood",   # CA Lower Montane Blue Oak Woodland
    7021: "hardwood",   # Great Plains Foothill and Canyon Grassland (hardwood)
}

# Top EVT codes to encode as individual dummies (cover the bulk of pixels)
TOP_EVT_CODES = [7054, 7028, 7045, 7050, 7043, 7055, 7027, 7037, 7053, 7047,
                 7032, 7166]   # top 12, covering ~85% of forest pixels

# ── Baseline predictor set (no sev, no EVT) ─────────────────────────────────
BASELINE_PREDS = [
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
    "CLIM_pr_sum_post_z", "CLIM_tmmn_mean_post_z",
    "CLIM_aridity_post_z", "CLIM_tmmx_std_post_z",
]

ALL_RESPONSES      = ["Resistance", "T80", "IRI_good_pow2", "STAB_good_pow2"]
RECOVERY_RESPONSES = {"T80", "IRI_good_pow2", "STAB_good_pow2"}


# ── Transform helpers ─────────────────────────────────────────────────────────
def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def log1p_z(series: pd.Series) -> pd.Series:
    return zscore(np.log1p(pd.to_numeric(series, errors="coerce").astype(float).clip(lower=0)))


def ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z_map = {
        "TS_elev_m_z": "TS_elev_m", "TS_slope_deg_z": "TS_slope_deg",
        "TS_northness_z": "TS_northness", "TS_eastness_z": "TS_eastness",
        "TS_twi_z": "TS_twi", "TS_roughness_z": "TS_roughness",
        "TS_SOC_0_30cm_z": "TS_SOC_0_30cm",
        "FS_TCC_t0_z": "FS_TCC_t0", "FS_CBH_t0agg_z": "FS_CBH_t0agg",
        "HUM_roaddens_r5km_z": "HUM_roaddens_r5km",
        "HUM_traildens_r10km_z": "HUM_traildens_r10km",
        "HUM_imperv_near_t0_z": "HUM_imperv_near_t0",
        "CLIM_pr_sum_pre_z": "CLIM_pr_sum_pre", "CLIM_eto_sum_pre_z": "CLIM_eto_sum_pre",
        "CLIM_tmmn_mean_pre_z": "CLIM_tmmn_mean_pre",
        "CLIM_hot_days_35C_pre_z": "CLIM_hot_days_35C_pre",
        "CLIM_aridity_pre_z": "CLIM_aridity_pre", "CLIM_tmmx_std_pre_z": "CLIM_tmmx_std_pre",
        "CLIM_vpd_mean_pre_z": "CLIM_vpd_mean_pre", "CLIM_vpd_std_pre_z": "CLIM_vpd_std_pre",
        "CLIM_pr_sum_post_z": "CLIM_pr_sum_post", "CLIM_tmmn_mean_post_z": "CLIM_tmmn_mean_post",
        "CLIM_aridity_post_z": "CLIM_aridity_post", "CLIM_tmmx_std_post_z": "CLIM_tmmx_std_post",
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


def add_fire_regime_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map EVT codes → fire-regime groups → dummy columns."""
    regime = df["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    dummies = pd.get_dummies(regime, prefix="EVT_regime", drop_first=False).astype(np.float32)
    # Drop the "other" column (reference category)
    other_col = "EVT_regime_other"
    if other_col in dummies.columns:
        dummies = dummies.drop(columns=[other_col])
    dummy_cols = list(dummies.columns)
    out = pd.concat([df, dummies], axis=1)
    return out, dummy_cols


def add_top_evt_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode top EVT codes; everything else → EVT_other."""
    code = df["FS_EVT2022_code"].copy()
    dummies_list = []
    col_names = []
    for c in TOP_EVT_CODES:
        col = f"EVT_{c}"
        dummies_list.append((code == c).astype(np.float32).rename(col))
        col_names.append(col)
    out = pd.concat([df] + dummies_list, axis=1)
    return out, col_names


# ── RF runner ─────────────────────────────────────────────────────────────────
def run_rf(df: pd.DataFrame, response: str, predictors: list[str]) -> dict:
    needed = [response] + predictors
    work = df[[c for c in needed if c in df.columns]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()
    actual_preds = [p for p in predictors if p in work.columns]
    if len(work) < 200:
        return {"rows_used": len(work), "test_r2": float("nan"),
                "test_rmse": float("nan"), "importances": None, "predictors": actual_preds}
    X, y = work[actual_preds], work[response]
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
        "importances": model.feature_importances_.tolist(),
        "predictors": actual_preds,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data ...")
    raw = pd.read_parquet(INPUT)

    # EVT forest mask
    forest_mask = raw["FS_EVT_group_class"].isin(EVT_FOREST_CLASSES)
    df_forest = raw[forest_mask].copy()
    print(f"  After EVT forest mask: {len(df_forest):,} rows")

    # Add derived columns
    df_forest = ensure_base_columns(df_forest)
    df_forest, regime_cols = add_fire_regime_dummies(df_forest)
    df_forest, top_evt_cols = add_top_evt_dummies(df_forest)

    print(f"\n  Fire-regime dummy columns ({len(regime_cols)}): {regime_cols}")
    print(f"  Top-EVT dummy columns ({len(top_evt_cols)}): {top_evt_cols[:6]}...")

    # Fire-regime pixel counts
    print("\n  Fire-regime distribution:")
    regime_col = df_forest["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    for grp, cnt in regime_col.value_counts().items():
        print(f"    {grp}: {cnt:,}")

    # ── Define variants ────────────────────────────────────────────────────────
    VARIANTS = {
        "baseline_noevt": {
            "extra_preds": [],
            "label": "Baseline (VPD+poly, no EVT)",
        },
        "fire_regime_dummies": {
            "extra_preds": regime_cols,
            "label": "+fire-regime dummies (5 groups)",
        },
        "top_evt_dummies": {
            "extra_preds": top_evt_cols,
            "label": f"+top {len(top_evt_cols)} EVT dummies",
        },
        "regime_plus_top_evt": {
            "extra_preds": regime_cols + top_evt_cols,
            "label": "+regime dummies + top EVT dummies",
        },
    }

    rows = []
    total = len(VARIANTS) * len(ALL_RESPONSES)
    done = 0

    for variant_name, vcfg in VARIANTS.items():
        for response in ALL_RESPONSES:
            is_recovery = response in RECOVERY_RESPONSES
            preds = list(BASELINE_PREDS)
            if is_recovery:
                preds += POST_PREDS
            preds += vcfg["extra_preds"]

            result = run_rf(df_forest, response, preds)
            done += 1

            row = {
                "variant": variant_name,
                "variant_label": vcfg["label"],
                "response": response,
                "n_predictors": len(result["predictors"]),
                "rows_used": result["rows_used"],
                "test_r2": result["test_r2"],
                "test_rmse": result["test_rmse"],
            }
            rows.append(row)

            # Save feature importance
            if result["importances"] is not None:
                imp = pd.DataFrame({
                    "predictor": result["predictors"],
                    "importance": result["importances"],
                }).sort_values("importance", ascending=False).reset_index(drop=True)
                imp.to_csv(
                    OUT_DIR / f"importance_{variant_name}__{response}.csv", index=False
                )

            print(
                f"  [{done}/{total}] {variant_name} | {response} | "
                f"rows={result['rows_used']:,} | test_R²={result['test_r2']:.4f}"
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUT_DIR / "evt_fireregime_results.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    resp_order = [r for r in ALL_RESPONSES if r in results_df["response"].values]
    baseline_r2 = (
        results_df[results_df["variant"] == "baseline_noevt"]
        .set_index("response")["test_r2"]
        .to_dict()
    )
    pivot = results_df.pivot_table(
        index=["variant", "variant_label"], columns="response", values="test_r2"
    ).reset_index().sort_values("Resistance", ascending=False)

    lines = [
        f"# WestUS RF: EVT Fire-Regime Dummies ({TODAY})",
        "",
        f"- n_estimators: {N_ESTIMATORS}",
        "- Forest mask: EVT group class in {conifer, mixed, deciduous}",
        "- sev: removed (endogenous)",
        "- Baseline extra: VPD+poly+postclim (recovery vars)",
        "",
        "## Fire-regime classification",
        "",
        "| Group | Description | Key species | n pixels |",
        "|---|---|---|---|",
    ]
    for grp, cnt in regime_col.value_counts().items():
        desc_map = {
            "surface_fire": "Thick bark, frequent low-severity surface fire",
            "mixed_severity": "Interior Douglas-fir, Grand Fir, Sierra mixed conifer",
            "stand_replacing": "Lodgepole, Spruce-Fir, subalpine — crown/stand fire",
            "coastal_mixed": "Pacific maritime Douglas-fir, Hemlock, Redwood",
            "hardwood": "Aspen, Oak — resprouting",
            "other": "Remaining EVT codes not classified above",
        }
        sp_map = {
            "surface_fire": "Ponderosa pine, Jeffrey pine, Juniper",
            "mixed_severity": "Interior Douglas-fir, Grand Fir, White Fir",
            "stand_replacing": "Lodgepole pine, Engelmann spruce, Subalpine fir",
            "coastal_mixed": "Pacific Douglas-fir, W. Hemlock, Silver Fir, Redwood",
            "hardwood": "Aspen, California Black Oak",
            "other": "—",
        }
        lines.append(
            f"| {grp} | {desc_map.get(grp,'—')} | {sp_map.get(grp,'—')} | {cnt:,} |"
        )

    lines += [
        "",
        "## Test R² by variant",
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

    lines += ["", "## Best variant per response", ""]
    for resp in ALL_RESPONSES:
        sub = results_df[results_df["response"] == resp].sort_values(
            "test_r2", ascending=False
        )
        best = sub.iloc[0]
        delta = best["test_r2"] - baseline_r2.get(resp, float("nan"))
        lines.append(
            f"- **{resp}**: `{best['variant']}` — "
            f"test R²=`{best['test_r2']:.4f}` (+`{delta:.4f}` vs no-EVT baseline), "
            f"n_preds={best['n_predictors']}"
        )

    # Regime importance summary (from fire_regime_dummies variant)
    lines += ["", "## Fire-regime dummy importance in RF (Resistance model)", ""]
    imp_path = OUT_DIR / "importance_fire_regime_dummies__Resistance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        regime_imp = imp[imp["predictor"].str.startswith("EVT_regime")]
        if not regime_imp.empty:
            lines.append("| Regime group | Importance |")
            lines.append("|---|---|")
            for _, row in regime_imp.iterrows():
                lines.append(f"| {row['predictor']} | {row['importance']:.4f} |")

    report = "\n".join(lines) + "\n"
    (OUT_DIR / "evt_fireregime_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
