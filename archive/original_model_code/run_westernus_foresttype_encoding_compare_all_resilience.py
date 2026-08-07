#!/usr/bin/env python3
from __future__ import annotations

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
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
TODAY = date.today().strftime("%Y-%m-%d")
OUT_DIR = ROOT / f"foresttype_encoding_compare_all_resilience_{TODAY}"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 300
BLOCK_KM = 100.0

RESPONSES = [
    "Resistance",
    "T50",
    "T80",
    "IRI_good_pow2",
    "STAB_good_pow2",
]

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
}


def zscore(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float)
    std = vals.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(vals), dtype=np.float32), index=vals.index)
    return ((vals - vals.mean()) / std).astype(np.float32)


def log1p_z(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").astype(float).clip(lower=0)
    return zscore(np.log1p(vals))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

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

    out["FS_EVT_group_class_clean"] = (
        out["FS_EVT_group_class"].astype("string").fillna("unknown").astype(str)
    )
    group_dummies = pd.get_dummies(
        out["FS_EVT_group_class_clean"], prefix="EVT_group", dtype=np.float32
    )
    out = pd.concat([out, group_dummies], axis=1)

    code_str = out["FS_EVT2022_code"].astype("Int64").astype(str).fillna("missing")
    code_dummies = pd.get_dummies(code_str, prefix="EVT_code", dtype=np.float32)
    out = pd.concat([out, code_dummies], axis=1)
    return out


def block_groups(df: pd.DataFrame, block_km: float) -> pd.Series:
    bm = block_km * 1000.0
    return pd.Series(
        [f"{int(np.floor(x / bm))}_{int(np.floor(y / bm))}" for x, y in zip(df["x"], df["y"])],
        index=df.index,
    )


def fit_and_score(df: pd.DataFrame, response: str, predictors: list[str]) -> list[dict]:
    needed = list(dict.fromkeys([response, "x", "y"] + predictors))
    work = (
        df[[c for c in needed if c in df.columns]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    actual_preds = [p for p in predictors if p in work.columns]

    idx = np.arange(len(work))
    rand_tr, rand_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    groups = block_groups(work, BLOCK_KM)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    blk_tr, blk_te = next(gss.split(idx, groups=groups))

    out = []
    for split_name, (tr, te) in [("random", (rand_tr, rand_te)), ("block", (blk_tr, blk_te))]:
        train = work.iloc[tr]
        test = work.iloc[te]
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[actual_preds], train[response])
        pred = model.predict(test[actual_preds])
        y = test[response].to_numpy()
        out.append(
            {
                "split": split_name,
                "rows": int(len(work)),
                "n_predictors": int(len(actual_preds)),
                "r2": float(r2_score(y, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            }
        )
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(INPUT)
    df = prepare(raw)

    group_cols = sorted([c for c in df.columns if c.startswith("EVT_group_")])
    code_cols = sorted([c for c in df.columns if c.startswith("EVT_code_")])

    variants = {
        "M1_baseline_only": BASE_PREDS,
        "M2_evt_group_class": BASE_PREDS + group_cols,
        "M3_evt_raw_code": BASE_PREDS + code_cols,
    }

    all_rows: list[dict] = []
    for response in RESPONSES:
        if response not in df.columns:
            print(f"Skipping {response}: not found", flush=True)
            continue

        print(f"\n=== Response: {response} ===", flush=True)
        for name, preds in variants.items():
            print(f"Running {name} with {len(preds)} candidate predictors...", flush=True)
            metrics = fit_and_score(df, response, preds)
            for row in metrics:
                row["response"] = response
                row["variant"] = name
                all_rows.append(row)
                print(
                    f"  {name} {row['split']}: R2={row['r2']:.4f} RMSE={row['rmse']:.4f} "
                    f"(rows={row['rows']}, p={row['n_predictors']})",
                    flush=True,
                )

    out_df = pd.DataFrame(all_rows)[
        ["response", "variant", "split", "rows", "n_predictors", "r2", "rmse"]
    ].sort_values(["response", "split", "r2"], ascending=[True, True, False])
    out_df.to_csv(OUT_DIR / "foresttype_encoding_compare_all_resilience_metrics.csv", index=False)

    lines = [
        f"# Forest-Type Representation Comparison Across Resilience Dimensions ({TODAY})",
        "",
        "| Response | Variant | Split | Rows | Predictors | R2 | RMSE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in out_df.iterrows():
        lines.append(
            f"| {r['response']} | {r['variant']} | {r['split']} | {int(r['rows'])} | {int(r['n_predictors'])} | {r['r2']:.4f} | {r['rmse']:.4f} |"
        )

    lines.extend(
        [
            "",
            "Variant definitions:",
            f"- M1_baseline_only: {len(BASE_PREDS)} baseline predictors, no forest-type term.",
            f"- M2_evt_group_class: baseline + {len(group_cols)} broad forest-type dummy columns.",
            f"- M3_evt_raw_code: baseline + {len(code_cols)} raw EVT code dummy columns.",
        ]
    )
    (OUT_DIR / "foresttype_encoding_compare_all_resilience_report.md").write_text("\n".join(lines))
    print("\nSaved:")
    print(OUT_DIR / "foresttype_encoding_compare_all_resilience_metrics.csv")
    print(OUT_DIR / "foresttype_encoding_compare_all_resilience_report.md")


if __name__ == "__main__":
    main()
