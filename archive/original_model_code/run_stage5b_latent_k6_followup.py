#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
TODAY = date.today().strftime("%Y-%m-%d")
OUT_DIR = ROOT / f"stage5b_latent_k6_followup_{TODAY}"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BLOCK_KM = 100.0
RF_TREES = 200
RF_N_JOBS = 1
MORAN_K = 8
LATENT_K = 6
LATENT_N_PCS = 8

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
EVT_PREDS = ["FS_EVT_resistance_proxy_z", "FS_EVT_regeneration_proxy_z"]

LATENT_FEATURE_Z_COLS = [
    "TS_elev_m_z",
    "TS_slope_deg_z",
    "TS_twi_z",
    "TS_SOC_0_30cm_z",
    "FS_TCC_t0_z",
    "FS_CBH_t0agg_z",
    "FS_EVT_resistance_proxy_z",
    "FS_EVT_regeneration_proxy_z",
    "HUM_popdens_win10km_log_z",
    "HUM_viirs_near_t0_log_z",
    "HUM_imperv_near_t0_z",
    "HUM_roaddens_r5km_z",
    "HUM_traildens_r10km_z",
    "CLIM_pr_sum_pre_z",
    "CLIM_tmmn_mean_pre_z",
    "CLIM_hot_days_35C_pre_z",
    "CLIM_tmmx_std_pre_z",
    "CLIM_aridity_pre_z",
    "CLIM_vpd_mean_pre_z",
    "CLIM_vpd_std_pre_z",
]

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
    "FS_EVT_resistance_proxy_z": "FS_EVT_resistance_proxy",
    "FS_EVT_regeneration_proxy_z": "FS_EVT_regeneration_proxy",
}

STAGE5B_INTERPRETATION = [
    {
        "variable": "Intercept",
        "bandwidth": 43,
        "scale_class": "very_local",
        "coefficient_pattern": "Strong southwest-to-northeast baseline contrast; highly patchy local structure.",
        "ecological_interpretation": "Background resistance context varies sharply at very local scale beyond observed predictors.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "FS_CBH_t0agg_z",
        "bandwidth": 122,
        "scale_class": "very_local",
        "coefficient_pattern": "Localized positive hotspots in inland clusters.",
        "ecological_interpretation": "Canopy base height acts as a local buffering structural driver.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "TS_elev_m_z",
        "bandwidth": 652,
        "scale_class": "local",
        "coefficient_pattern": "Stronger positive effects in southwestern mountains; weaker or near-zero in lowlands.",
        "ecological_interpretation": "Topographic resistance effects are spatially contingent and strongest in mountain systems.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "TS_SOC_0_30cm_clean_z",
        "bandwidth": 513,
        "scale_class": "local",
        "coefficient_pattern": "Patchy local variation with sign changes across the domain.",
        "ecological_interpretation": "Soil-carbon effects are local and heterogeneous rather than domain-wide.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "HUM_roaddens_r5km_z",
        "bandwidth": 252,
        "scale_class": "local",
        "coefficient_pattern": "Local positive and negative patches; localized human-access signal.",
        "ecological_interpretation": "Road-density effects are local and context-dependent.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "CLIM_pr_sum_pre_z",
        "bandwidth": 503,
        "scale_class": "local",
        "coefficient_pattern": "Stronger positive association in dry southwestern areas and weaker in wetter northern areas.",
        "ecological_interpretation": "Moisture limitation is spatially contingent; precipitation matters most in dry systems.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "TS_slope_deg_z",
        "bandwidth": 1856,
        "scale_class": "regional",
        "coefficient_pattern": "Broad regional negative pattern with moderate spatial smoothness.",
        "ecological_interpretation": "Slope behaves as a larger-scale terrain constraint rather than a local driver.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "FS_TCC_t0_z",
        "bandwidth": 1018,
        "scale_class": "regional",
        "coefficient_pattern": "Generally positive with localized coastal negatives.",
        "ecological_interpretation": "Tree cover is broadly beneficial, but regional ecological context modifies the sign near some coastal systems.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "HUM_traildens_r10km_z",
        "bandwidth": 4490,
        "scale_class": "regional",
        "coefficient_pattern": "Broad regional differences with smoother transitions.",
        "ecological_interpretation": "Trail-density effects operate at subregional to regional scales.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "CLIM_tmmn_mean_pre_z",
        "bandwidth": 3372,
        "scale_class": "regional",
        "coefficient_pattern": "Broad continuous negative climatic gradient.",
        "ecological_interpretation": "Minimum temperature acts as a regional climatic background driver rather than a local signal.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "HUM_viirs_near_t0_log_z",
        "bandwidth": 11588,
        "scale_class": "quasi_global",
        "coefficient_pattern": "Nearly uniform negative effect across space.",
        "ecological_interpretation": "Nighttime-light intensity behaves like a near-global human-disturbance penalty.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
    {
        "variable": "HUM_imperv_near_t0_z",
        "bandwidth": 11995,
        "scale_class": "quasi_global",
        "coefficient_pattern": "Nearly uniform positive-to-negative domain-wide pattern with minimal local variation.",
        "ecological_interpretation": "Imperviousness acts like a near-global broad-scale human footprint term.",
        "source": "qualitative read from stage5b_coef_maps.png",
    },
]


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
    if "HUM_popdens_win10km_log_z" not in out.columns and "HUM_popdens_win10km" in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns and "HUM_viirs_near_t0" in out.columns:
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
    out = pd.concat([out, dummies], axis=1)
    return out, list(dummies.columns)


def block_groups(df: pd.DataFrame, block_km: float) -> pd.Series:
    block_m = block_km * 1000.0
    labels = [
        f"{int(np.floor(x / block_m))}_{int(np.floor(y / block_m))}"
        for x, y in zip(df["x"], df["y"])
    ]
    return pd.Series(labels, index=df.index)


def fit_rf(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=RF_TREES,
        random_state=RANDOM_STATE,
        n_jobs=RF_N_JOBS,
    )
    model.fit(X, y)
    return model


def compute_moran(coords_df: pd.DataFrame, residuals: np.ndarray) -> float:
    k = min(MORAN_K, len(coords_df) - 1)
    if k < 1:
        return float("nan")
    weights = KNN.from_array(coords_df[["x", "y"]].to_numpy(), k=k)
    weights.transform = "R"
    return float(Moran(residuals.astype(float), weights, permutations=0).I)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, coords_df: pd.DataFrame) -> dict[str, float]:
    residuals = y_true - y_pred
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "moran_i": compute_moran(coords_df, residuals),
    }


def eta_squared_from_groups(values: pd.Series, groups: pd.Series) -> float:
    work = pd.DataFrame({"value": values.astype(float), "group": groups}).dropna()
    grand_mean = work["value"].mean()
    ss_total = ((work["value"] - grand_mean) ** 2).sum()
    if ss_total <= 0:
        return float("nan")
    group_stats = work.groupby("group")["value"].agg(["mean", "size"])
    ss_between = ((group_stats["mean"] - grand_mean) ** 2 * group_stats["size"]).sum()
    return float(ss_between / ss_total)


def save_stage5b_interpretation() -> tuple[Path, Path]:
    df = pd.DataFrame(STAGE5B_INTERPRETATION)
    csv_path = OUT_DIR / "stage5b_formal_interpretation_table.csv"
    md_path = OUT_DIR / "stage5b_formal_interpretation_table.md"
    df.to_csv(csv_path, index=False)

    lines = [
        "# Stage5b Formal Interpretation Table",
        "",
        "| Variable | Bandwidth | Scale class | Coefficient pattern | Ecological interpretation |",
        "|---|---:|---|---|---|",
    ]
    for row in STAGE5B_INTERPRETATION:
        lines.append(
            f"| {row['variable']} | {row['bandwidth']} | {row['scale_class']} | "
            f"{row['coefficient_pattern']} | {row['ecological_interpretation']} |"
        )
    lines.append("")
    lines.append("> Note: coefficient-pattern descriptions are qualitative readings from `stage5b_coef_maps.png` and should be upgraded to quantitative surface summaries if the raw stage5b coefficient table is synced locally.")
    md_path.write_text("\n".join(lines))
    return csv_path, md_path


def run_latent_k6(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keep_cols = ["pixel_id", "row", "col", "x", "y", "lon_wgs84", "lat_wgs84", "region", "Resistance"] + LATENT_FEATURE_Z_COLS
    keep_cols = [c for c in keep_cols if c in df.columns]
    work = (
        df[keep_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=LATENT_FEATURE_Z_COLS)
        .reset_index(drop=True)
    )
    X = work[LATENT_FEATURE_Z_COLS].to_numpy(dtype=float)
    pca = PCA(n_components=LATENT_N_PCS, random_state=RANDOM_STATE)
    scores = pca.fit_transform(X)
    labels = KMeans(n_clusters=LATENT_K, random_state=RANDOM_STATE, n_init=20).fit_predict(scores)

    assign = work[["pixel_id", "row", "col", "x", "y", "lon_wgs84", "lat_wgs84", "region", "Resistance"]].copy()
    assign["latent_k6"] = labels.astype(int)
    for i in range(scores.shape[1]):
        assign[f"PC{i + 1}"] = scores[:, i]

    assign.to_csv(OUT_DIR / "latent_k6_assignments.csv", index=False)
    pd.DataFrame(
        pca.components_.T,
        index=LATENT_FEATURE_Z_COLS,
        columns=[f"PC{i+1}" for i in range(scores.shape[1])],
    ).to_csv(OUT_DIR / "latent_k6_pca_loadings.csv")
    pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(scores.shape[1])],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    ).to_csv(OUT_DIR / "latent_k6_pca_explained_variance.csv", index=False)

    summary = (
        assign.groupby("latent_k6")
        .agg(
            n=("Resistance", "size"),
            resistance_mean=("Resistance", "mean"),
            resistance_median=("Resistance", "median"),
            resistance_std=("Resistance", "std"),
            x_mean=("x", "mean"),
            y_mean=("y", "mean"),
        )
        .reset_index()
    )
    env_means = work.groupby(labels)[LATENT_FEATURE_Z_COLS].mean().reset_index().rename(columns={"index": "latent_k6"})
    env_means.columns = ["latent_k6"] + LATENT_FEATURE_Z_COLS
    summary = summary.merge(env_means, on="latent_k6", how="left")
    summary.to_csv(OUT_DIR / "latent_k6_summary.csv", index=False)

    resistance_stats = (
        assign.groupby("latent_k6")["Resistance"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    resistance_stats.to_csv(OUT_DIR / "latent_k6_resistance_distribution.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    sc = axes[0].scatter(assign["x"], assign["y"], c=assign["latent_k6"], s=3, cmap="tab10", alpha=0.85)
    axes[0].set_title("Latent Regime Map (k=6)")
    axes[0].set_xlabel("x (EPSG:5070)")
    axes[0].set_ylabel("y (EPSG:5070)")
    cbar = fig.colorbar(sc, ax=axes[0], ticks=range(LATENT_K))
    cbar.set_label("latent_k6")

    ordered = [assign.loc[assign["latent_k6"] == k, "Resistance"].to_numpy() for k in range(LATENT_K)]
    axes[1].boxplot(ordered, labels=[f"R{k}" for k in range(LATENT_K)], patch_artist=True)
    axes[1].set_title("Resistance distribution by latent regime (k=6)")
    axes[1].set_xlabel("Regime")
    axes[1].set_ylabel("Resistance")
    fig.savefig(OUT_DIR / "latent_k6_map_and_resistance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "rows_used": int(len(assign)),
        "features": LATENT_FEATURE_Z_COLS,
        "n_pcs": LATENT_N_PCS,
        "k": LATENT_K,
        "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "resistance_eta2_by_latent_k6": eta_squared_from_groups(assign["Resistance"], assign["latent_k6"]),
        "resistance_eta2_by_region": eta_squared_from_groups(assign["Resistance"], assign["region"]),
    }
    (OUT_DIR / "latent_k6_metadata.json").write_text(json.dumps(meta, indent=2))
    return assign, meta


def compare_rf_models(df: pd.DataFrame, evt_cols: list[str], latent_assign: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictors_e = [c for c in BASE_PREDS + EVT_PREDS + evt_cols if c in df.columns]
    latent_join = latent_assign[["pixel_id", "latent_k6"]].copy()

    work = (
        df[list(dict.fromkeys(["pixel_id", "Resistance", "x", "y", "region"] + predictors_e))]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .merge(latent_join, on="pixel_id", how="inner")
        .reset_index(drop=True)
    )

    region_dummies = pd.get_dummies(work["region"], prefix="region", drop_first=True).astype(np.float32)
    latent_dummies = pd.get_dummies(work["latent_k6"], prefix="latent_k6", drop_first=True).astype(np.float32)
    work = pd.concat([work, region_dummies, latent_dummies], axis=1)

    model_specs = OrderedDict(
        [
            ("E_global", predictors_e),
            ("E_plus_region", predictors_e + list(region_dummies.columns)),
            ("E_plus_latent_k6", predictors_e + list(latent_dummies.columns)),
            ("E_plus_region_and_latent_k6", predictors_e + list(region_dummies.columns) + list(latent_dummies.columns)),
        ]
    )

    idx = np.arange(len(work))
    rand_tr, rand_te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    groups = block_groups(work, BLOCK_KM)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    blk_tr, blk_te = next(gss.split(idx, groups=groups))

    metrics_rows = []
    residual_rows = []
    pred_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    for model_name, predictors in model_specs.items():
        for split_name, (tr, te) in [("random", (rand_tr, rand_te)), ("block", (blk_tr, blk_te))]:
            train = work.iloc[tr]
            test = work.iloc[te]
            model = fit_rf(train[predictors], train["Resistance"])
            pred = model.predict(test[predictors])
            y = test["Resistance"].to_numpy(dtype=float)
            metrics = metric_dict(y, pred, test[["x", "y"]])
            metrics_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "rows": int(len(test)),
                    "predictors": int(len(predictors)),
                    **metrics,
                }
            )
            pred_cache[(model_name, split_name)] = (pred, te)

    # Residual-by-regime for block split only
    for model_name in model_specs:
        pred, te = pred_cache[(model_name, "block")]
        test = work.iloc[te].copy()
        test["pred"] = pred
        test["residual"] = test["Resistance"] - test["pred"]
        test["abs_residual"] = np.abs(test["residual"])

        by_latent = (
            test.groupby("latent_k6")
            .agg(
                n=("Resistance", "size"),
                resistance_mean=("Resistance", "mean"),
                pred_mean=("pred", "mean"),
                residual_mean=("residual", "mean"),
                abs_residual_mean=("abs_residual", "mean"),
            )
            .reset_index()
        )
        by_latent["model"] = model_name
        by_latent["group_type"] = "latent_k6"
        by_latent = by_latent.rename(columns={"latent_k6": "group"})
        residual_rows.append(by_latent)

        by_region = (
            test.groupby("region")
            .agg(
                n=("Resistance", "size"),
                resistance_mean=("Resistance", "mean"),
                pred_mean=("pred", "mean"),
                residual_mean=("residual", "mean"),
                abs_residual_mean=("abs_residual", "mean"),
            )
            .reset_index()
        )
        by_region["model"] = model_name
        by_region["group_type"] = "region"
        by_region = by_region.rename(columns={"region": "group"})
        residual_rows.append(by_region)

    metrics_df = pd.DataFrame(metrics_rows)
    residual_df = pd.concat(residual_rows, ignore_index=True)
    metrics_df.to_csv(OUT_DIR / "latent_vs_region_rf_metrics.csv", index=False)
    residual_df.to_csv(OUT_DIR / "latent_vs_region_block_residual_summary.csv", index=False)

    # Model-level effect-size comparison on block residuals
    eta_rows = []
    for model_name in model_specs:
        pred, te = pred_cache[(model_name, "block")]
        test = work.iloc[te].copy()
        test["pred"] = pred
        test["residual"] = test["Resistance"] - test["pred"]
        test["abs_residual"] = np.abs(test["residual"])
        eta_rows.append(
            {
                "model": model_name,
                "split": "block",
                "eta2_resistance_by_latent_k6": eta_squared_from_groups(test["Resistance"], test["latent_k6"]),
                "eta2_resistance_by_region": eta_squared_from_groups(test["Resistance"], test["region"]),
                "eta2_abs_residual_by_latent_k6": eta_squared_from_groups(test["abs_residual"], test["latent_k6"]),
                "eta2_abs_residual_by_region": eta_squared_from_groups(test["abs_residual"], test["region"]),
            }
        )
    eta_df = pd.DataFrame(eta_rows)
    eta_df.to_csv(OUT_DIR / "latent_vs_region_effect_sizes.csv", index=False)
    return metrics_df, eta_df


def write_report(latent_meta: dict, metrics_df: pd.DataFrame, eta_df: pd.DataFrame) -> None:
    block_metrics = metrics_df.query("split == 'block'").copy()
    block_metrics = block_metrics.sort_values("r2", ascending=False)
    best_block = block_metrics.iloc[0]
    lines = [
        f"# Stage5b + latent k=6 follow-up ({TODAY})",
        "",
        "## 1. Stage5b formal interpretation",
        "",
        f"- Formal interpretation table written to `stage5b_formal_interpretation_table.csv` and `stage5b_formal_interpretation_table.md`.",
        "- `stage5b` is treated as the current best stable climate-extended MGWR endpoint.",
        "",
        "## 2. Latent k=6 summary",
        "",
        f"- Rows used for PCA latent regimes: `{latent_meta['rows_used']:,}`",
        f"- PCA PCs retained: `{LATENT_N_PCS}`",
        f"- Cumulative explained variance (8 PCs): `{latent_meta['explained_variance_sum']:.4f}`",
        f"- Eta-squared of Resistance by latent_k6: `{latent_meta['resistance_eta2_by_latent_k6']:.4f}`",
        f"- Eta-squared of Resistance by coarse region: `{latent_meta['resistance_eta2_by_region']:.4f}`",
        "",
        "## 3. RF comparison: E vs E+region vs E+latent_k6",
        "",
        "| Model | Split | Rows | Predictors | R2 | RMSE | Moran's I |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in metrics_df.sort_values(["split", "r2"], ascending=[True, False]).iterrows():
        lines.append(
            f"| {row['model']} | {row['split']} | {int(row['rows'])} | {int(row['predictors'])} | "
            f"{row['r2']:.4f} | {row['rmse']:.4f} | {row['moran_i']:.4f} |"
        )

    lines += [
        "",
        "## 4. Residual-group effect sizes (block split)",
        "",
        "| Model | eta2(abs residual by latent_k6) | eta2(abs residual by region) |",
        "|---|---:|---:|",
    ]
    for _, row in eta_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['eta2_abs_residual_by_latent_k6']:.4f} | {row['eta2_abs_residual_by_region']:.4f} |"
        )

    lines += [
        "",
        "## 5. Summary",
        "",
        f"- Best block model in this follow-up: `{best_block['model']}` with `R2={best_block['r2']:.4f}` and `Moran's I={best_block['moran_i']:.4f}`.",
        "- If `E_plus_latent_k6` beats `E_plus_region` on block R2 and/or Moran's I, that supports latent regimes as a more useful grouping layer than the current coarse regions.",
        "- Quantitative regime-by-coefficient analysis still needs the raw stage5b coefficient surface synced locally; current coefficient interpretation remains image-based.",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_parquet(INPUT)
    df, evt_cols = ensure_columns(df_raw)

    save_stage5b_interpretation()
    latent_assign, latent_meta = run_latent_k6(df)
    metrics_df, eta_df = compare_rf_models(df, evt_cols, latent_assign)
    write_report(latent_meta, metrics_df, eta_df)

    print(
        json.dumps(
            {
                "out_dir": str(OUT_DIR),
                "latent_rows": latent_meta["rows_used"],
                "best_block_model": metrics_df.query("split == 'block'").sort_values("r2", ascending=False).iloc[0]["model"],
            }
        )
    )


if __name__ == "__main__":
    main()
