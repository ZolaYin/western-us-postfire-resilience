#!/usr/bin/env python3
"""Build regional residual-bias diagnostics for retained resilience metrics."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


ROOT = Path(
    "/path/to/google-drive"
    "/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km"
)
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
OUT_DIR = ROOT / "retained_resilience_regional_residual_bias_2026-06-03"
OUT_PNG = OUT_DIR / "retained_resilience_regional_residual_bias_fourpanel.png"
OUT_SUMMARY = OUT_DIR / "retained_resilience_m2_block100km_regional_residual_summary.csv"
OUT_PRED = OUT_DIR / "retained_resilience_m2_block100km_predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
BLOCK_KM = 100.0
N_ESTIMATORS = 300

RESPONSES = [
    ("Resistance", "Resistance"),
    ("IRI_good_pow2", "IRI"),
    ("STAB_good_pow2", "STAB"),
]

REGION_ORDER = ["PNW", "CA_med", "S_Rockies", "N_Rockies", "SW_dry"]
REGION_LABELS = {
    "PNW": "Pacific Northwest",
    "CA_med": "California\nMediterranean",
    "S_Rockies": "Southern\nRockies",
    "N_Rockies": "Northern\nRockies",
    "SW_dry": "Southwest\ndry forests",
}
REGION_COLORS = {
    "PNW": "#2b83ba",
    "CA_med": "#d7191c",
    "S_Rockies": "#fdae61",
    "N_Rockies": "#1a9641",
    "SW_dry": "#7b3294",
}

BASE_PREDS = [
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
    "CLIM_vpd_mean_pre_z",
    "CLIM_vpd_std_pre_z",
    "x",
    "y",
    "x_sq_z",
    "y_sq_z",
    "xy_z",
]

RAW_TO_Z = {
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


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    for z_col, raw_col in RAW_TO_Z.items():
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

    evt_group = out["FS_EVT_group_class"].astype("string").fillna("unknown").astype(str)
    group_dummies = pd.get_dummies(evt_group, prefix="EVT_group", dtype=np.float32)
    out = pd.concat([out, group_dummies], axis=1)
    return out, list(group_dummies.columns)


def block_groups(df: pd.DataFrame) -> pd.Series:
    block_m = BLOCK_KM * 1000.0
    labels = (
        np.floor(df["x"].to_numpy(dtype=float) / block_m).astype(int).astype(str)
        + "_"
        + np.floor(df["y"].to_numpy(dtype=float) / block_m).astype(int).astype(str)
    )
    return pd.Series(labels, index=df.index)


def fit_predict_one(df: pd.DataFrame, response: str, predictors: list[str]) -> pd.DataFrame:
    keep_cols = list(dict.fromkeys([response, "region", "x", "y", *predictors]))
    work = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    groups = block_groups(work)
    idx = np.arange(len(work))
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(idx, groups=groups))

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(work.loc[train_idx, predictors], work.loc[train_idx, response])

    test = work.loc[test_idx, [response, "region", "x", "y"]].copy()
    test["predicted"] = model.predict(work.loc[test_idx, predictors])
    test = test.rename(columns={response: "observed"})
    test["residual"] = test["observed"] - test["predicted"]
    test["abs_residual"] = test["residual"].abs()
    test["response"] = response
    test["n_train"] = len(train_idx)
    test["n_test"] = len(test_idx)
    test["n_groups"] = groups.nunique()
    test["n_test_groups"] = groups.iloc[test_idx].nunique()
    return test


def summarize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for response, sub_response in pred.groupby("response", sort=False):
        total = len(sub_response)
        for region in REGION_ORDER:
            sub = sub_response[sub_response["region"] == region]
            if sub.empty:
                continue
            resid = sub["residual"].to_numpy(dtype=float)
            rows.append(
                {
                    "response": response,
                    "region": region,
                    "n": int(len(sub)),
                    "pct": float(100 * len(sub) / total),
                    "observed_mean": float(sub["observed"].mean()),
                    "predicted_mean": float(sub["predicted"].mean()),
                    "residual_mean": float(sub["residual"].mean()),
                    "abs_residual_mean": float(sub["abs_residual"].mean()),
                    "residual_sd": float(sub["residual"].std(ddof=1)),
                    "residual_p10": float(np.percentile(resid, 10)),
                    "residual_p90": float(np.percentile(resid, 90)),
                    "rmse": float(np.sqrt(np.mean(resid**2))),
                    "bias_interpretation": "overprediction"
                    if float(sub["residual"].mean()) < 0
                    else "underprediction",
                }
            )
    return pd.DataFrame(rows)


def plot_region_map(ax: plt.Axes, df: pd.DataFrame) -> None:
    for region in REGION_ORDER:
        sub = df[df["region"] == region]
        if sub.empty:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            s=0.8,
            marker="s",
            linewidths=0,
            alpha=0.45,
            color=REGION_COLORS[region],
            rasterized=True,
            label=REGION_LABELS[region].replace("\n", " "),
        )
        cx = sub["x"].median()
        cy = sub["y"].median()
        ax.text(
            cx,
            cy,
            REGION_LABELS[region],
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=2.0),
        )
    ax.set_title("(a) Regional groups", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.axis("off")


def plot_bias_panel(ax: plt.Axes, summary: pd.DataFrame, response: str, title: str) -> None:
    sub = summary[summary["response"] == response].set_index("region").reindex(REGION_ORDER)
    y = np.arange(len(sub))
    colors = np.where(sub["residual_mean"].to_numpy() < 0, "#e8751a", "#36aa7b")

    ax.barh(y, sub["residual_mean"], color=colors, height=0.58, edgecolor="none")
    ax.scatter(sub["abs_residual_mean"], y, s=28, color="#1f1f1f", zorder=3)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.grid(axis="x", color="#e8e8e8", lw=0.7)
    ax.set_axisbelow(True)

    xmax = max(float(sub["abs_residual_mean"].max()) * 1.55, 0.02)
    xmin = min(float(sub["residual_mean"].min()) * 1.45, -0.015)
    ax.set_xlim(xmin, xmax)

    ax.set_yticks(y)
    ax.set_yticklabels([REGION_LABELS[r].replace("\n", " ") for r in REGION_ORDER], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(f"Residual in {title.split()[-1]} units", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)

    for yi, (_, row) in enumerate(sub.iterrows()):
        resid = row["residual_mean"]
        ha = "right" if resid < 0 else "left"
        dx = -0.003 if resid < 0 else 0.003
        ax.text(
            resid + dx,
            yi,
            f"{resid:+.3f}",
            va="center",
            ha=ha,
            fontsize=7,
            color="#222222",
        )
        ax.text(
            row["abs_residual_mean"] + xmax * 0.025,
            yi,
            f"|e|={row['abs_residual_mean']:.3f}",
            va="center",
            ha="left",
            fontsize=7,
            color="#222222",
        )
        ax.text(
            xmax * 0.985,
            yi,
            f"n={int(row['n']):,}",
            va="center",
            ha="right",
            fontsize=6.7,
            color="#555555",
        )

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#dddddd")


def plot_fourpanel(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.4), constrained_layout=False)
    plot_region_map(axes[0, 0], df.dropna(subset=["region", "x", "y"]))
    plot_bias_panel(axes[0, 1], summary, "Resistance", "(b) Resistance")
    plot_bias_panel(axes[1, 0], summary, "IRI_good_pow2", "(c) IRI")
    plot_bias_panel(axes[1, 1], summary, "STAB_good_pow2", "(d) STAB")

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.07, top=0.86, wspace=0.26, hspace=0.36)
    handles = [
        mpatches.Patch(color="#e8751a", label="overprediction"),
        mpatches.Patch(color="#36aa7b", label="underprediction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f1f1f", markersize=5, label="mean absolute residual"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Regional residual bias under 100-km spatial block validation",
        fontsize=13,
        fontweight="bold",
        y=0.975,
    )
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(INPUT)
    df, group_cols = prepare(df)
    predictors = BASE_PREDS + group_cols

    predictions = []
    for response, label in RESPONSES:
        print(f"Fitting M2 block RF for {label}...")
        predictions.append(fit_predict_one(df, response, predictors))
    pred = pd.concat(predictions, ignore_index=True)
    summary = summarize_predictions(pred)

    pred.to_csv(OUT_PRED, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    plot_fourpanel(df, summary)

    print(f"Wrote: {OUT_PNG}")
    print(f"Wrote: {OUT_SUMMARY}")
    print(f"Wrote: {OUT_PRED}")


if __name__ == "__main__":
    main()
