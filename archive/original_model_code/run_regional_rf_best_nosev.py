#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
TODAY = date.today().strftime("%Y-%m-%d")

RANDOM_STATE = 42
TEST_SIZE = 0.2
DEFAULT_TREES = 200
DEFAULT_BLOCK_KM = 100.0
DEFAULT_RESPONSE = "Resistance"
DEFAULT_VARIANT = "E_EVT_fireregime_postclim"
RECOVERY_RESPONSES = {"T80", "IRI_good_pow2", "STAB_good_pow2"}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict pooled comparison of global/regional RF strategies."
    )
    parser.add_argument("--response", default=DEFAULT_RESPONSE)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--trees", type=int, default=DEFAULT_TREES)
    parser.add_argument("--block-km", type=float, default=DEFAULT_BLOCK_KM)
    return parser.parse_args()


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


def block_groups(df: pd.DataFrame, block_km: float) -> pd.Series:
    block_m = block_km * 1000.0
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    labels = [
        f"{int(np.floor(xi / block_m))}_{int(np.floor(yi / block_m))}"
        for xi, yi in zip(x, y)
    ]
    return pd.Series(labels, index=df.index)


def fit_rf(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def build_variants(response: str, regime_cols: list[str]) -> OrderedDict[str, dict]:
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


def prepare_work(df: pd.DataFrame, response: str, predictors: list[str]) -> pd.DataFrame:
    cols = list(dict.fromkeys([response, "region", "x", "y", *predictors]))
    work = (
        df[cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    return work


def build_shared_splits(
    work: pd.DataFrame, block_km: float
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], pd.Series]:
    idx = np.arange(len(work))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    groups = block_groups(work, block_km=block_km)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    block_train_idx, block_test_idx = next(gss.split(idx, groups=groups))
    return (train_idx, test_idx), (block_train_idx, block_test_idx), groups


def add_region_dummies(work: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dummies = pd.get_dummies(work["region"], prefix="region", drop_first=True).astype(np.float32)
    dummy_cols = list(dummies.columns)
    work2 = pd.concat([work.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return work2, dummy_cols


def predict_global(
    work: pd.DataFrame,
    predictors: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_estimators: int,
) -> tuple[np.ndarray, RandomForestRegressor]:
    X_train = work.iloc[train_idx][predictors]
    y_train = work.iloc[train_idx]["response"]
    X_test = work.iloc[test_idx][predictors]
    model = fit_rf(X_train, y_train, n_estimators)
    pred = model.predict(X_test)
    return pred, model


def predict_regional(
    work: pd.DataFrame,
    predictors: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_estimators: int,
) -> tuple[np.ndarray, dict[str, RandomForestRegressor], list[dict[str, int]]]:
    train = work.iloc[train_idx].reset_index(drop=True)
    test = work.iloc[test_idx].reset_index(drop=True)
    pred = np.full(len(test), np.nan, dtype=float)
    region_models: dict[str, RandomForestRegressor] = {}
    region_counts: list[dict[str, int]] = []

    for region in sorted(work["region"].unique()):
        train_mask = train["region"] == region
        test_mask = test["region"] == region
        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())
        region_counts.append({"region": region, "n_train": n_train, "n_test": n_test})
        if n_train == 0 or n_test == 0:
            continue

        model = fit_rf(
            train.loc[train_mask, predictors],
            train.loc[train_mask, "response"],
            n_estimators,
        )
        pred[test_mask.to_numpy()] = model.predict(test.loc[test_mask, predictors])
        region_models[region] = model

    if np.isnan(pred).any():
        missing_regions = sorted(test.loc[np.isnan(pred), "region"].unique().tolist())
        raise RuntimeError(
            "Regional RF failed to produce predictions for regions: "
            + ", ".join(missing_regions)
        )

    return pred, region_models, region_counts


def collect_region_metrics(
    work: pd.DataFrame,
    test_idx: np.ndarray,
    pred: np.ndarray,
    strategy: str,
    split_name: str,
) -> list[dict]:
    test = work.iloc[test_idx].reset_index(drop=True)
    rows = []
    for region, sub in test.assign(pred=pred).groupby("region", dropna=False):
        rows.append(
            {
                "strategy": strategy,
                "split": split_name,
                "region": region,
                "n_test": int(len(sub)),
                "r2": float(r2_score(sub["response"], sub["pred"])),
                "rmse": float(np.sqrt(mean_squared_error(sub["response"], sub["pred"]))),
            }
        )
    return rows


def top_importance_rows(
    region_models: dict[str, RandomForestRegressor],
    predictors: list[str],
    split_name: str,
    top_n: int = 10,
) -> list[dict]:
    rows = []
    for region, model in region_models.items():
        imp = (
            pd.Series(model.feature_importances_, index=predictors)
            .sort_values(ascending=False)
            .head(top_n)
        )
        for rank, (feature, importance) in enumerate(imp.items(), 1):
            rows.append(
                {
                    "region": region,
                    "split": split_name,
                    "rank": rank,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
    return rows


def evaluate_strategies(
    work: pd.DataFrame,
    predictors: list[str],
    n_estimators: int,
    block_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random_split, block_split, _ = build_shared_splits(work, block_km=block_km)
    work_rdummy, region_dummy_cols = add_region_dummies(work)

    strategy_rows = []
    region_rows = []
    importance_rows = []

    split_map = {"random": random_split, "block": block_split}

    for split_name, (train_idx, test_idx) in split_map.items():
        y_test = work.iloc[test_idx]["response"].to_numpy()

        global_pred, _ = predict_global(
            work, predictors, train_idx, test_idx, n_estimators
        )
        global_metrics = metric_dict(y_test, global_pred)

        rdummy_pred, _ = predict_global(
            work_rdummy,
            predictors + region_dummy_cols,
            train_idx,
            test_idx,
            n_estimators,
        )
        rdummy_metrics = metric_dict(y_test, rdummy_pred)

        regional_pred, region_models, region_counts = predict_regional(
            work, predictors, train_idx, test_idx, n_estimators
        )
        regional_metrics = metric_dict(y_test, regional_pred)

        for strategy, metrics_ in [
            ("global_rf", global_metrics),
            ("global_rf_rdummy", rdummy_metrics),
            ("regional_rf", regional_metrics),
        ]:
            strategy_rows.append(
                {
                    "strategy": strategy,
                    "split": split_name,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "r2": metrics_["r2"],
                    "rmse": metrics_["rmse"],
                }
            )

        region_rows.extend(
            collect_region_metrics(work, test_idx, global_pred, "global_rf", split_name)
        )
        region_rows.extend(
            collect_region_metrics(
                work, test_idx, rdummy_pred, "global_rf_rdummy", split_name
            )
        )
        region_rows.extend(
            collect_region_metrics(work, test_idx, regional_pred, "regional_rf", split_name)
        )

        for rc in region_counts:
            region_rows.append(
                {
                    "strategy": "regional_rf_train_counts",
                    "split": split_name,
                    "region": rc["region"],
                    "n_test": rc["n_test"],
                    "r2": float(rc["n_train"]),
                    "rmse": np.nan,
                }
            )

        if split_name == "block":
            importance_rows.extend(top_importance_rows(region_models, predictors, split_name))

    strategy_df = pd.DataFrame(strategy_rows)
    region_df = pd.DataFrame(region_rows)
    importance_df = pd.DataFrame(importance_rows)

    metrics_df = (
        strategy_df.pivot(index="strategy", columns="split", values=["r2", "rmse", "n_train", "n_test"])
        .sort_index()
    )
    metrics_df.columns = [f"{split}_{metric}" for metric, split in metrics_df.columns]
    metrics_df = metrics_df.reset_index()
    metrics_df["block_delta"] = metrics_df["block_r2"] - metrics_df["random_r2"]
    return metrics_df, region_df, importance_df


def build_report(
    response: str,
    variant: str,
    label: str,
    role: str,
    predictors: list[str],
    rows_used: int,
    metrics_df: pd.DataFrame,
    region_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    block_km: float,
    n_estimators: int,
) -> str:
    lines = [
        f"# Regional RF vs Global RF - {response} ({TODAY})",
        "",
        f"**Input:** `{INPUT.name}`  ",
        f"**Response:** `{response}`  ",
        f"**Variant:** `{variant}`  ",
        f"**Variant label:** {label}  ",
        f"**Variant role:** `{role}`  ",
        f"**Rows used:** {rows_used:,}  ",
        f"**Predictors:** {len(predictors)}  ",
        f"**Block size:** {block_km:g} km  ",
        f"**Trees per RF:** {n_estimators}",
        "",
        "## 1. Strategy-level pooled comparison",
        "",
        "| Strategy | Random R2 | Random RMSE | Block R2 | Block RMSE | Block delta |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in metrics_df.sort_values("strategy").iterrows():
        lines.append(
            f"| {row['strategy']} | {row['random_r2']:.4f} | {row['random_rmse']:.4f} "
            f"| {row['block_r2']:.4f} | {row['block_rmse']:.4f} | {row['block_delta']:+.4f} |"
        )

    lines += [
        "",
        "## 2. Per-region block performance (same pooled split)",
        "",
        "| Strategy | Region | n_test | Block R2 | Block RMSE |",
        "|---|---|---|---|---|",
    ]
    block_rows = region_df[
        (region_df["split"] == "block") & (region_df["strategy"] != "regional_rf_train_counts")
    ]
    for _, row in block_rows.sort_values(["strategy", "region"]).iterrows():
        lines.append(
            f"| {row['strategy']} | {row['region']} | {int(row['n_test']):,} "
            f"| {row['r2']:.4f} | {row['rmse']:.4f} |"
        )

    train_rows = region_df[
        (region_df["split"] == "block") & (region_df["strategy"] == "regional_rf_train_counts")
    ]
    if not train_rows.empty:
        lines += [
            "",
            "## 3. Regional RF block train/test counts",
            "",
            "| Region | n_train | n_test |",
            "|---|---|---|",
        ]
        for _, row in train_rows.sort_values("region").iterrows():
            lines.append(
                f"| {row['region']} | {int(row['r2']):,} | {int(row['n_test']):,} |"
            )

    if not importance_df.empty:
        lines += ["", "## 4. Top-10 regional RF features (block-trained models)", ""]
        for region in sorted(importance_df["region"].unique()):
            sub = importance_df[importance_df["region"] == region].sort_values("rank")
            lines.append(f"**{region}**")
            for _, row in sub.iterrows():
                lines.append(
                    f"{int(row['rank'])}. `{row['feature']}` = {row['importance']:.4f}"
                )
            lines.append("")

    lines += [
        "## 5. Interpretation notes",
        "",
        "- All three strategies use the same pooled random split and the same pooled 100 km block split.",
        "- Per-region block scores are computed from those same held-out predictions, so strategies are directly comparable.",
        "- `regional_rf` means one model per region, trained only on that region's training subset and evaluated only on that region's held-out subset.",
        "- If `global_rf_rdummy` is close to `global_rf`, coarse region identity adds little beyond the existing predictors.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    raw = pd.read_parquet(INPUT)
    if "Forest_at_t0" in raw.columns:
        raw = raw[raw["Forest_at_t0"] == 1].reset_index(drop=True)

    df, regime_cols = ensure_columns(raw)
    variants = build_variants(args.response, regime_cols)
    if args.variant not in variants:
        raise ValueError(f"Unknown variant: {args.variant}")

    spec = variants[args.variant]
    predictors = spec["predictors"]
    work = prepare_work(df, args.response, predictors).rename(columns={args.response: "response"})

    out_dir = ROOT / f"regional_rf_best_nosev_{args.variant}_strict_{TODAY}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running strict regional RF comparison for {args.response} / {args.variant}")
    print(f"Rows used: {len(work):,}")
    print(f"Region counts: {work['region'].value_counts().to_dict()}")

    metrics_df, region_df, importance_df = evaluate_strategies(
        work=work,
        predictors=predictors,
        n_estimators=args.trees,
        block_km=args.block_km,
    )

    metrics_df.to_csv(out_dir / "regional_rf_metrics.csv", index=False)
    region_df.to_csv(out_dir / "regional_prediction_metrics.csv", index=False)
    importance_df.to_csv(out_dir / "regional_importance_summary.csv", index=False)

    report = build_report(
        response=args.response,
        variant=args.variant,
        label=spec["label"],
        role=spec["role"],
        predictors=predictors,
        rows_used=len(work),
        metrics_df=metrics_df,
        region_df=region_df,
        importance_df=importance_df,
        block_km=args.block_km,
        n_estimators=args.trees,
    )
    (out_dir / "regional_rf_report.md").write_text(report, encoding="utf-8")
    (out_dir / "regional_rf_metrics.json").write_text(
        json.dumps(metrics_df.round(6).to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "response": args.response,
                "variant": args.variant,
                "variant_label": spec["label"],
                "variant_role": spec["role"],
                "predictors": predictors,
                "rows_used": len(work),
                "trees": args.trees,
                "block_km": args.block_km,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(report)
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
