from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/westernus_postfire_mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "westernus_current_candidate_table_plus_regions.parquet"
OUT_DIR = ROOT / "rf_variable_ablation_current11state_2026-07-08"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 200
RESPONSE = "Resistance"
NEUTRAL_HIDE_FRACTION = 0.10

PREDICTORS = [
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

TARGET_VARIABLES = [
    ("TS_elev_m_z", "Elevation"),
    ("TS_slope_deg_z", "Slope"),
    ("TS_SOC_0_30cm_z", "SOC"),
    ("FS_TCC_t0_z", "TCC"),
    ("FS_CBH_t0agg_z", "CBH"),
    ("HUM_roaddens_r5km_z", "Road density"),
    ("HUM_traildens_r10km_z", "Trail density"),
    ("HUM_viirs_near_t0_log_z", "VIIRS light"),
    ("HUM_imperv_near_t0_z", "Imperviousness"),
    ("CLIM_pr_sum_pre_z", "Precipitation"),
    ("CLIM_tmmn_mean_pre_z", "Minimum temperature"),
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
    for z_col, base_col in BASE_TO_Z.items():
        if z_col not in out.columns and base_col in out.columns:
            out[z_col] = zscore(out[base_col])
    if "HUM_popdens_win10km_log_z" not in out.columns:
        out["HUM_popdens_win10km_log_z"] = log1p_z(out["HUM_popdens_win10km"])
    if "HUM_viirs_near_t0_log_z" not in out.columns:
        out["HUM_viirs_near_t0_log_z"] = log1p_z(out["HUM_viirs_near_t0"])
    return out


def robust_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    vmax = float(np.nanpercentile(np.abs(finite), 98))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)


def main() -> None:
    raw = pd.read_parquet(INPUT)
    df = ensure_columns(raw)

    needed = [RESPONSE, "pixel_id", "lon_wgs84", "lat_wgs84"] + PREDICTORS
    work = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = work[PREDICTORS].astype(np.float32)
    y = work[RESPONSE].astype(np.float32)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_features="sqrt",
    )
    model.fit(X, y)
    base_pred = model.predict(X)

    effects = {}
    summary = []
    medians = X.median(numeric_only=True)
    for var, label in TARGET_VARIABLES:
        X_alt = X.copy()
        X_alt[var] = medians[var]
        effect = base_pred - model.predict(X_alt)
        effects[var] = effect.astype(np.float32)
        finite = effect[np.isfinite(effect)]
        summary.append(
            {
                "variable": var,
                "label": label,
                "n": int(finite.size),
                "mean": float(np.mean(finite)),
                "mean_abs": float(np.mean(np.abs(finite))),
                "p02": float(np.quantile(finite, 0.02)),
                "p50": float(np.quantile(finite, 0.50)),
                "p98": float(np.quantile(finite, 0.98)),
            }
        )

    out = work[["pixel_id", "lon_wgs84", "lat_wgs84"]].copy()
    for var, effect in effects.items():
        out[f"rf_effect__{var}"] = effect
    out.to_parquet(OUT_DIR / "rf_variable_ablation_current11state_resistance.parquet", index=False)
    pd.DataFrame(summary).to_csv(OUT_DIR / "rf_variable_ablation_current11state_resistance_summary.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig, axes = plt.subplots(3, 4, figsize=(12.2, 8.0), constrained_layout=False)
    axes_flat = axes.ravel()
    lon = work["lon_wgs84"].to_numpy(dtype=float)
    lat = work["lat_wgs84"].to_numpy(dtype=float)
    for ax, (var, label) in zip(axes_flat, TARGET_VARIABLES):
        vals = effects[var]
        norm = robust_norm(vals)
        neutral_threshold = max(abs(norm.vmax) * NEUTRAL_HIDE_FRACTION, np.finfo(float).eps)
        keep = np.isfinite(vals) & (np.abs(vals) >= neutral_threshold)
        if not np.any(keep):
            keep = np.isfinite(vals)
        plot_order = np.argsort(np.abs(vals[keep]))
        sc = ax.scatter(
            lon[keep][plot_order],
            lat[keep][plot_order],
            c=vals[keep][plot_order],
            s=1.0,
            cmap="RdBu_r",
            norm=norm,
            linewidths=0,
            rasterized=True,
        )
        mean_abs = np.nanmean(np.abs(vals))
        ax.set_title(f"{label}\nRF ablation |effect|={mean_abs:.3f}", fontsize=8, weight="bold")
        ax.set_xlim(-125.2, -102.5)
        ax.set_ylim(31.0, 49.6)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.012)
        cb.ax.tick_params(labelsize=6, length=2)

    for ax in axes_flat[len(TARGET_VARIABLES) :]:
        ax.axis("off")

    fig.suptitle(
        "Current 11-state RF variable-ablation effect maps: Resistance",
        fontsize=13,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.018,
        "RF variable-ablation effect = prediction(full predictors) - prediction(variable set to median). "
        "Near-zero neutral points are hidden; darker colors indicate larger absolute local effect.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0.015, 0.04, 0.985, 0.965])
    fig.savefig(OUT_DIR / "fig_rf_variable_ablation_current11state_resistance.png", dpi=300)
    fig.savefig(OUT_DIR / "fig_rf_variable_ablation_current11state_resistance.pdf")
    plt.close(fig)

    print("rows_used", len(work))
    print(OUT_DIR / "fig_rf_variable_ablation_current11state_resistance.png")
    print(OUT_DIR / "rf_variable_ablation_current11state_resistance_summary.csv")


if __name__ == "__main__":
    main()
