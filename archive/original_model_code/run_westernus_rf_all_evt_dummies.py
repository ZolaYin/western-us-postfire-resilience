#!/usr/bin/env python3
"""
WestUS RF — all EVT code dummies (with min-pixel threshold).

Tests one-hot encoding of ALL EVT codes above a pixel threshold,
rare codes merged into EVT_other.

Compares:
  fire_regime   : 5-group fire-regime dummies (from previous run, best was D: R²=0.669)
  evt_min200    : all EVT codes >= 200 pixels as dummies
  evt_min100    : all EVT codes >= 100 pixels as dummies
  evt_min50     : all EVT codes >= 50 pixels as dummies

All: NLCD mask, no sev, base+VPD+poly+postclim(recovery).
"""
from __future__ import annotations

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
    ROOT / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
TODAY = date.today().strftime("%Y-%m-%d")
OUT_DIR = ROOT / f"westernus_rf_all_evt_dummies_{TODAY}"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_ESTIMATORS = 500

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
    "CLIM_pr_sum_post_z", "CLIM_tmmn_mean_post_z",
    "CLIM_aridity_post_z", "CLIM_tmmx_std_post_z",
]

ALL_RESPONSES      = ["Resistance", "T80", "IRI_good_pow2", "STAB_good_pow2"]
RECOVERY_RESPONSES = {"T80", "IRI_good_pow2", "STAB_good_pow2"}


def zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    std = v.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(v), dtype=np.float32), index=v.index)
    return ((v - v.mean()) / std).astype(np.float32)


def log1p_z(s: pd.Series) -> pd.Series:
    return zscore(np.log1p(pd.to_numeric(s, errors="coerce").astype(float).clip(lower=0)))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
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
    }
    for z_col, raw_col in z_map.items():
        if z_col not in out.columns and raw_col in out.columns:
            out[z_col] = zscore(out[raw_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    for col, expr in [("x_sq_z", out["x"]**2), ("y_sq_z", out["y"]**2),
                      ("xy_z", out["x"]*out["y"])]:
        if col not in out.columns:
            out[col] = zscore(expr)
    return out


def make_evt_dummies(df: pd.DataFrame, min_pixels: int) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode EVT codes with >= min_pixels occurrences; rest → EVT_other."""
    counts = df["FS_EVT2022_code"].value_counts()
    common_codes = set(counts[counts >= min_pixels].index.tolist())

    # Assign label
    label = df["FS_EVT2022_code"].apply(
        lambda c: f"EVT_{int(c)}" if c in common_codes else "EVT_other"
    )
    dummies = pd.get_dummies(label, prefix="", prefix_sep="").astype(np.float32)
    # Drop "EVT_other" as reference category
    if "EVT_other" in dummies.columns:
        dummies = dummies.drop(columns=["EVT_other"])
    dummy_cols = list(dummies.columns)
    out = pd.concat([df, dummies], axis=1)
    print(f"  min_pixels={min_pixels}: {len(common_codes)} codes as dummies "
          f"({len(dummy_cols)} cols), "
          f"covering {counts[counts >= min_pixels].sum():,}/{len(df):,} rows "
          f"({counts[counts >= min_pixels].sum()/len(df)*100:.1f}%)")
    return out, dummy_cols


def make_fire_regime_dummies(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    regime = df["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    dummies = pd.get_dummies(regime, prefix="EVT_regime").astype(np.float32)
    if "EVT_regime_other" in dummies.columns:
        dummies = dummies.drop(columns=["EVT_regime_other"])
    cols = list(dummies.columns)
    out = pd.concat([df, dummies], axis=1)
    return out, cols


def run_rf(df: pd.DataFrame, response: str, predictors: list[str]) -> dict:
    work = df[[c for c in [response] + predictors if c in df.columns]].replace(
        [np.inf, -np.inf], np.nan).dropna().copy()
    actual = [p for p in predictors if p in work.columns]
    if len(work) < 200:
        return {"rows": len(work), "r2": float("nan"), "rmse": float("nan"),
                "imp": None, "preds": actual}
    X, y = work[actual], work[response]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE,
                                           random_state=RANDOM_STATE)
    m = RandomForestRegressor(n_estimators=N_ESTIMATORS,
                               random_state=RANDOM_STATE, n_jobs=-1)
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    return {"rows": int(len(work)), "r2": float(r2_score(yte, p)),
            "rmse": float(np.sqrt(mean_squared_error(yte, p))),
            "imp": m.feature_importances_.tolist(), "preds": actual}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(INPUT)
    print(f"Raw rows: {len(raw):,}, unique EVT codes: {raw['FS_EVT2022_code'].nunique()}")
    base_df = prepare(raw)

    # Build all EVT dummy sets — each from the clean base_df to avoid duplicate columns
    df_regime, regime_cols = make_fire_regime_dummies(base_df)
    df_evt50,  evt50_cols  = make_evt_dummies(base_df, min_pixels=50)
    df_evt100, evt100_cols = make_evt_dummies(base_df, min_pixels=100)
    df_evt200, evt200_cols = make_evt_dummies(base_df, min_pixels=200)

    VARIANTS = {
        "fire_regime": {"df": df_regime, "evt_cols": regime_cols,
                        "label": f"fire-regime dummies ({len(regime_cols)} groups)"},
        "evt_min200":  {"df": df_evt200, "evt_cols": evt200_cols,
                        "label": f"all EVT ≥200px ({len(evt200_cols)} dummies)"},
        "evt_min100":  {"df": df_evt100, "evt_cols": evt100_cols,
                        "label": f"all EVT ≥100px ({len(evt100_cols)} dummies)"},
        "evt_min50":   {"df": df_evt50,  "evt_cols": evt50_cols,
                        "label": f"all EVT ≥50px ({len(evt50_cols)} dummies)"},
    }

    print("\nStarting RF runs...")
    rows_out = []
    total = len(VARIANTS) * len(ALL_RESPONSES)
    done = 0

    for vname, vcfg in VARIANTS.items():
        vdf = vcfg["df"]
        for resp in ALL_RESPONSES:
            is_rec = resp in RECOVERY_RESPONSES
            preds  = list(BASE_PREDS) + vcfg["evt_cols"]
            if is_rec:
                preds += POST_PREDS

            res = run_rf(vdf, resp, preds)
            done += 1
            rows_out.append({
                "variant": vname, "label": vcfg["label"], "response": resp,
                "n_evt_dummies": len(vcfg["evt_cols"]),
                "n_preds": len(res["preds"]), "rows": res["rows"],
                "test_r2": res["r2"], "test_rmse": res["rmse"],
            })
            if res["imp"] is not None:
                imp_df = pd.DataFrame({"predictor": res["preds"],
                                       "importance": res["imp"]}
                                      ).sort_values("importance", ascending=False)
                imp_df.to_csv(OUT_DIR / f"imp_{vname}__{resp}.csv", index=False)

            print(f"  [{done}/{total}] {vname} | {resp} | "
                  f"rows={res['rows']:,} | R²={res['r2']:.4f} | "
                  f"n_dummies={len(vcfg['evt_cols'])}")

    results = pd.DataFrame(rows_out)
    results.to_csv(OUT_DIR / "all_evt_dummies_results.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    resp_order = ALL_RESPONSES
    pivot = (results.pivot_table(index=["variant","label","n_evt_dummies"],
                                  columns="response", values="test_r2")
             .reset_index().sort_values("Resistance", ascending=False))

    # Reference: fire_regime from best_nosev run
    ref = {"Resistance": 0.6689, "T80": 0.5286, "IRI_good_pow2": 0.6575, "STAB_good_pow2": 0.6269}

    lines = [
        f"# WestUS RF — All EVT Dummies vs Fire-Regime ({TODAY})",
        "",
        f"- n_estimators: {N_ESTIMATORS}  |  NLCD mask  |  no sev",
        "- Base predictors: 22 baseline + VPD + poly (+ postclim for recovery)",
        "- EVT dummy strategy: all codes ≥ threshold → individual dummies; rest → EVT_other (dropped)",
        "",
        "## EVT dummy counts",
        "",
        "| Variant | EVT codes encoded | % pixels covered |",
        "|---|---|---|",
    ]
    counts = raw["FS_EVT2022_code"].value_counts()
    for thresh, label in [(None, "fire-regime (5 groups)"),
                          (200, "≥200px"), (100, "≥100px"), (50, "≥50px")]:
        if thresh is None:
            n_codes = 5
            pct = 100.0
        else:
            n_codes = int((counts >= thresh).sum())
            pct = counts[counts >= thresh].sum() / len(raw) * 100
        lines.append(f"| {label} | {n_codes} | {pct:.1f}% |")

    lines += [
        "",
        "## Test R² by variant (sorted by Resistance R²)",
        "",
        "| Variant | n dummies | Resistance | T80 | IRI_good_pow2 | STAB_good_pow2 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in pivot.iterrows():
        cells = [r["label"], str(int(r["n_evt_dummies"]))]
        for resp in resp_order:
            val = r.get(resp, float("nan"))
            delta = val - ref.get(resp, float("nan"))
            if np.isnan(val):
                cells.append("—")
            else:
                sign = "+" if delta >= 0 else ""
                cells.append(f"{val:.4f} ({sign}{delta:.4f})")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Best per response (this run)", ""]
    for resp in resp_order:
        sub = results[results["response"] == resp].sort_values("test_r2", ascending=False)
        b = sub.iloc[0]
        lines.append(
            f"- **{resp}**: `{b['variant']}` ({b['n_evt_dummies']} dummies) — "
            f"R²=`{b['test_r2']:.4f}`, rows={b['rows']:,}"
        )

    # Top EVT codes by importance (evt_min50, Resistance)
    imp_path = OUT_DIR / "imp_evt_min50__Resistance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        evt_imp = imp[imp["predictor"].str.startswith("EVT_") &
                      ~imp["predictor"].str.startswith("EVT_regime")].head(15)
        if not evt_imp.empty:
            lines += ["", "## Top 15 individual EVT dummies by importance (Resistance, min50)", ""]
            lines += ["| EVT code | Importance |", "|---|---|"]
            for _, row in evt_imp.iterrows():
                lines.append(f"| {row['predictor']} | {row['importance']:.4f} |")

    report = "\n".join(lines) + "\n"
    (OUT_DIR / "all_evt_dummies_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
