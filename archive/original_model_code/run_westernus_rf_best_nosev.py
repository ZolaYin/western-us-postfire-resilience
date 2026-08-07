#!/usr/bin/env python3
"""
WestUS RF — best clean configuration (no sev, NLCD mask, enriched predictors).

Combines findings from three prior experiments:
  - NLCD mask (not EVT mask): keeps larger sample (133k Resistance, 56k recovery)
  - No sev (endogenous)
  - Enrichment: +VPD +poly +postclim (for recovery vars)
  - Tests whether EVT proxy and fire-regime dummies help with the larger NLCD sample

Variants:
  A. baseline_nosev       : 22 baseline + VPD + poly  (no EVT, no sev)
  B. +postclim            : A + postclim (recovery only)
  C. +EVTproxy+postclim   : B + EVT resistance/regen proxy
  D. +fireregime+postclim : B + fire-regime dummies (5 groups from EVT codes)
  E. +EVT+fireregime+post : B + EVT proxy + fire-regime dummies

All use NLCD forest mask (Forest_at_t0 == 1, which is all 133k rows).
EVT proxy columns will have NAs for ~11k rows → those rows dropped automatically.

Output: westernus_rf_best_nosev_<date>/
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
    ROOT
    / "westernus_roadtrail_append_legacy_nlcd_20260411"
    / "westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet"
)
TODAY = date.today().strftime("%Y-%m-%d")
OUT_DIR = ROOT / f"westernus_rf_best_nosev_{TODAY}"

RANDOM_STATE = 42
TEST_SIZE    = 0.2
N_ESTIMATORS = 500

# ── Fire-regime map (same as previous script) ─────────────────────────────────
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

# ── Predictor groups ──────────────────────────────────────────────────────────
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
POST_PREDS  = ["CLIM_pr_sum_post_z", "CLIM_tmmn_mean_post_z",
               "CLIM_aridity_post_z", "CLIM_tmmx_std_post_z"]
EVT_PREDS   = ["FS_EVT_resistance_proxy_z", "FS_EVT_regeneration_proxy_z"]

ALL_RESPONSES      = ["Resistance", "T80", "IRI_good_pow2", "STAB_good_pow2"]
RECOVERY_RESPONSES = {"T80", "IRI_good_pow2", "STAB_good_pow2"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def zscore(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype(float)
    std = v.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(v), dtype=np.float32), index=v.index)
    return ((v - v.mean()) / std).astype(np.float32)


def log1p_z(s: pd.Series) -> pd.Series:
    return zscore(np.log1p(pd.to_numeric(s, errors="coerce").astype(float).clip(lower=0)))


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
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

    # Fire-regime dummies
    regime = out["FS_EVT2022_code"].map(FIRE_REGIME_MAP).fillna("other")
    dummies = pd.get_dummies(regime, prefix="EVT_regime").astype(np.float32)
    other_col = "EVT_regime_other"
    if other_col in dummies.columns:
        dummies = dummies.drop(columns=[other_col])
    regime_cols = list(dummies.columns)
    out = pd.concat([out, dummies], axis=1)

    return out, regime_cols


def run_rf(df: pd.DataFrame, response: str, predictors: list[str]) -> dict:
    needed = [response] + predictors
    work = df[[c for c in needed if c in df.columns]].replace(
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
    return {"rows": int(len(work)),
            "r2": float(r2_score(yte, p)),
            "rmse": float(np.sqrt(mean_squared_error(yte, p))),
            "imp": m.feature_importances_.tolist(),
            "preds": actual}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(INPUT)
    print(f"Raw rows: {len(raw):,}")
    df, regime_cols = prepare(raw)
    print(f"Regime dummy cols: {regime_cols}")

    VARIANTS = {
        "A_baseline_nosev": {
            "extra": [],
            "post": False,
            "label": "A: baseline + VPD + poly (no sev, no EVT)",
        },
        "B_plus_postclim": {
            "extra": [],
            "post": True,
            "label": "B: A + postclim (recovery only)",
        },
        "C_EVTproxy_postclim": {
            "extra": EVT_PREDS,
            "post": True,
            "label": "C: B + EVT proxy (resist+regen)",
        },
        "D_fireregime_postclim": {
            "extra": regime_cols,
            "post": True,
            "label": "D: B + fire-regime dummies",
        },
        "E_EVT_fireregime_postclim": {
            "extra": EVT_PREDS + regime_cols,
            "post": True,
            "label": "E: B + EVT proxy + fire-regime dummies",
        },
    }

    rows_out = []
    total = len(VARIANTS) * len(ALL_RESPONSES)
    done = 0

    for vname, vcfg in VARIANTS.items():
        for resp in ALL_RESPONSES:
            is_rec = resp in RECOVERY_RESPONSES
            preds = list(BASE_PREDS) + vcfg["extra"]
            if is_rec and vcfg["post"]:
                preds += POST_PREDS

            res = run_rf(df, resp, preds)
            done += 1
            rows_out.append({
                "variant": vname, "label": vcfg["label"], "response": resp,
                "n_preds": len(res["preds"]), "rows": res["rows"],
                "test_r2": res["r2"], "test_rmse": res["rmse"],
            })
            if res["imp"] is not None:
                pd.DataFrame({"predictor": res["preds"], "importance": res["imp"]}
                             ).sort_values("importance", ascending=False
                             ).to_csv(OUT_DIR / f"imp_{vname}__{resp}.csv", index=False)
            print(f"  [{done}/{total}] {vname} | {resp} | "
                  f"rows={res['rows']:,} | R²={res['r2']:.4f}")

    results = pd.DataFrame(rows_out)
    results.to_csv(OUT_DIR / "best_nosev_results.csv", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    resp_order = ALL_RESPONSES
    base_r2 = (results[results["variant"] == "A_baseline_nosev"]
               .set_index("response")["test_r2"].to_dict())
    pivot = (results.pivot_table(index=["variant","label"], columns="response",
                                 values="test_r2")
             .reset_index().sort_values("Resistance", ascending=False))

    lines = [
        f"# WestUS RF Best (No Sev) — NLCD Mask ({TODAY})",
        "",
        f"- n_estimators: {N_ESTIMATORS}",
        "- Mask: NLCD forest (Forest_at_t0 == 1, all rows)",
        "- sev: excluded",
        "- Base predictors: 22 baseline + VPD + poly",
        "",
        "## Test R² comparison",
        "",
        "| Variant | " + " | ".join(resp_order) + " |",
        "|" + "---|" * (len(resp_order) + 1),
    ]
    for _, r in pivot.iterrows():
        cells = [r["label"]]
        for resp in resp_order:
            val = r.get(resp, float("nan"))
            d = val - base_r2.get(resp, float("nan"))
            if np.isnan(val):
                cells.append("—")
            else:
                sign = "+" if d >= 0 else ""
                cells.append(f"{val:.4f} ({sign}{d:.4f})")
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Row counts per response", ""]
    for resp in resp_order:
        n = raw[resp].notna().sum()
        lines.append(f"- `{resp}`: {n:,} rows (NLCD mask)")

    lines += ["", "## Best variant per response", ""]
    for resp in resp_order:
        sub = results[results["response"] == resp].sort_values("test_r2", ascending=False)
        b = sub.iloc[0]
        lines.append(
            f"- **{resp}**: `{b['variant']}` — R²=`{b['test_r2']:.4f}` "
            f"(+`{b['test_r2']-base_r2[resp]:.4f}` vs A), n_preds={b['n_preds']}, rows={b['rows']:,}"
        )

    # Cross-experiment summary
    lines += [
        "", "## Cross-experiment summary (all configs tried)",
        "",
        "| Config | Resistance | T80 | IRI | STAB | Notes |",
        "|---|---|---|---|---|---|",
        "| NLCD mask, baseline (no sev, no VPD/poly) | 0.635 | 0.492 | 0.619 | 0.574 | First WestUS result |",
        "| NLCD mask, +sev+EVT+VPD+poly | 0.693 | 0.533 | 0.656 | 0.648 | Sev is endogenous |",
        "| EVT forest mask, full_nosev | 0.661 | 0.396 | 0.558 | 0.515 | EVT mask too restrictive for recovery |",
        "| EVT mask + fire-regime dummies | 0.660 | 0.397 | 0.559 | 0.515 | Species info redundant within EVT-forest |",
    ]
    # Add this run's best
    for resp in resp_order:
        pass  # will add below
    b_resi = results[(results["response"]=="Resistance")].sort_values("test_r2",ascending=False).iloc[0]
    b_t80  = results[(results["response"]=="T80")].sort_values("test_r2",ascending=False).iloc[0]
    b_iri  = results[(results["response"]=="IRI_good_pow2")].sort_values("test_r2",ascending=False).iloc[0]
    b_stab = results[(results["response"]=="STAB_good_pow2")].sort_values("test_r2",ascending=False).iloc[0]
    lines.append(
        f"| **NLCD mask, no sev, best this run** | "
        f"**{b_resi['test_r2']:.3f}** | **{b_t80['test_r2']:.3f}** | "
        f"**{b_iri['test_r2']:.3f}** | **{b_stab['test_r2']:.3f}** | "
        f"← **current best (no sev)** |"
    )

    report = "\n".join(lines) + "\n"
    (OUT_DIR / "best_nosev_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
