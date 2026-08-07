#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import libpysal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from esda.moran import Moran
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:  # pragma: no cover
    lowess = None


ROOT = Path(__file__).resolve().parent
M2_BASELINE_RESID = ROOT / "residual_plot_m2_baseline_2026-05-01" / "m2_baseline_block_100km_predictions.csv"
RF_RESID = ROOT / "residual_plot_best_rf_2026-05-01" / "m2_resid_localz5_noelev_cons_block_100km_predictions.csv"
MGWR_RESID = ROOT / "mgwr_outputs_stage5b_s12k" / "mgwr_residuals.parquet"
OUT_DIR = ROOT / "rf_mgwr_fourpanel_residual_diagnostics_2026-06-01"
OUT_DIR.mkdir(exist_ok=True)


def read_residuals(path: Path, model_label: str) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)
    needed = ["x", "y", "observed", "predicted", "residual"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    out["model"] = model_label
    return out


def make_knn_moran(df: pd.DataFrame, k: int = 8) -> tuple[float, float]:
    coords = df[["x", "y"]].to_numpy(dtype=float)
    weights = libpysal.weights.KNN.from_array(coords, k=k)
    weights.transform = "r"
    moran = Moran(df["residual"].to_numpy(dtype=float), weights, permutations=999)
    return float(moran.I), float(moran.p_sim)


def point_density(x: np.ndarray, y: np.ndarray, bins: int = 160) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    counts, xedges, yedges = np.histogram2d(x[finite], y[finite], bins=bins)
    xi = np.searchsorted(xedges, x, side="right") - 1
    yi = np.searchsorted(yedges, y, side="right") - 1
    xi = np.clip(xi, 0, counts.shape[0] - 1)
    yi = np.clip(yi, 0, counts.shape[1] - 1)
    dens = counts[xi, yi].astype(float)
    max_dens = np.nanmax(dens) if len(dens) else 1.0
    return dens / max_dens if max_dens > 0 else dens


def fitted_lowess(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    if lowess is not None:
        fit = lowess(y_sorted, x_sorted, frac=0.18, it=0, return_sorted=True)
        return fit[:, 0], fit[:, 1]

    # Fallback: binned median trend if statsmodels is unavailable.
    qs = np.linspace(0, 1, 70)
    edges = np.unique(np.quantile(x_sorted, qs))
    mids, vals = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        sel = (x_sorted >= left) & (x_sorted <= right)
        if sel.sum() < 15:
            continue
        mids.append(float(np.median(x_sorted[sel])))
        vals.append(float(np.median(y_sorted[sel])))
    return np.asarray(mids), np.asarray(vals)


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, color="#d8d8d8", linewidth=0.45, alpha=0.65)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def p_label(p: float) -> str:
    return "< 0.001" if p <= 0.001 else f"= {p:.3f}"


def fit_label(slope: float, intercept: float) -> str:
    sign = "+" if intercept >= 0 else "-"
    return f"Fit: y = {slope:.2f}x {sign} {abs(intercept):.2f}"


def draw_fourpanel(
    df: pd.DataFrame,
    model_key: str,
    model_label: str,
    residual_vmax: float,
    obs_pred_xlim: tuple[float, float],
    resid_ylim: tuple[float, float],
    moran_i: float,
    moran_p: float,
) -> dict[str, float]:
    observed = df["observed"].to_numpy(dtype=float)
    predicted = df["predicted"].to_numpy(dtype=float)
    residual = df["residual"].to_numpy(dtype=float)

    r2 = float(r2_score(observed, predicted))
    rmse = float(mean_squared_error(observed, predicted) ** 0.5)
    bias = float(np.mean(residual))
    resid_sd = float(np.std(residual, ddof=1))
    resid_mean = float(np.mean(residual))

    lm = LinearRegression().fit(predicted.reshape(-1, 1), observed)
    slope = float(lm.coef_[0])
    intercept = float(lm.intercept_)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.7), dpi=300)
    ax_map, ax_obs, ax_hist, ax_rvp = axes.ravel()

    cmap_resid = "RdBu"
    residual_norm = TwoSlopeNorm(vmin=-residual_vmax, vcenter=0.0, vmax=residual_vmax)

    # (A) spatial residual distribution.
    sc = ax_map.scatter(
        df["x"],
        df["y"],
        c=residual,
        cmap=cmap_resid,
        norm=residual_norm,
        s=3.0,
        alpha=0.78,
        linewidths=0,
        rasterized=True,
    )
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title(
        "(A) Spatial distribution of residuals\n"
        "(blue = under-prediction; red = over-prediction)",
        fontsize=8.8,
        fontweight="bold",
        pad=5,
    )
    ax_map.text(
        0.03,
        0.96,
        f"Moran's I = {moran_i:.3f}  (p {p_label(moran_p)})\n"
        f"RMSE = {rmse:.3f}   Bias = {bias:+.3f}",
        transform=ax_map.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        bbox=dict(facecolor="white", edgecolor="#666666", linewidth=0.4, alpha=0.92, boxstyle="round,pad=0.25"),
    )
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.032, pad=0.012)
    cbar.set_label("Residual (observed - predicted)", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.6)

    # (B) observed vs predicted.
    density = point_density(predicted, observed)
    order = np.argsort(density)
    sc2 = ax_obs.scatter(
        predicted[order],
        observed[order],
        c=density[order],
        cmap="Blues",
        norm=Normalize(0, 1),
        s=2.6,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    x_min, x_max = obs_pred_xlim
    xx = np.linspace(x_min, x_max, 100)
    ax_obs.plot(xx, xx, color="black", linestyle="--", linewidth=0.85, label="1:1 line")
    ax_obs.plot(xx, slope * xx + intercept, color="#ef3b2c", linewidth=1.0, label=fit_label(slope, intercept))
    ax_obs.set_xlim(obs_pred_xlim)
    ax_obs.set_ylim(obs_pred_xlim)
    ax_obs.set_xlabel(f"{model_label} predicted Resistance", fontsize=7.5)
    ax_obs.set_ylabel("Observed Resistance", fontsize=7.5)
    ax_obs.set_title(f"(B) Observed vs. predicted Resistance\n$R^2$ = {r2:.3f},  RMSE = {rmse:.3f}", fontsize=8.8, fontweight="bold")
    add_grid(ax_obs)
    ax_obs.legend(loc="upper left", fontsize=6.6, frameon=True)
    cbar2 = fig.colorbar(sc2, ax=ax_obs, fraction=0.032, pad=0.012)
    cbar2.set_label("Point density (relative)", fontsize=7.0)
    cbar2.ax.tick_params(labelsize=6.6)

    # (C) residual distribution.
    bins = np.linspace(resid_ylim[0], resid_ylim[1], 70)
    ax_hist.hist(residual, bins=bins, density=True, color="#74add1", edgecolor="white", linewidth=0.25, alpha=0.92, label="Observed residuals")
    xs = np.linspace(resid_ylim[0], resid_ylim[1], 400)
    ax_hist.plot(xs, norm.pdf(xs, resid_mean, resid_sd), color="#ef3b2c", linewidth=1.1, label=f"Normal  N({resid_mean:+.3f}, {resid_sd:.3f})")
    ax_hist.axvline(0, color="black", linestyle="--", linewidth=0.8, label="Zero")
    ax_hist.axvline(resid_mean, color="#3b3b3b", linestyle=":", linewidth=0.9, label=f"Mean = {resid_mean:+.3f}")
    ax_hist.set_xlim(resid_ylim)
    ax_hist.set_xlabel("Residual (observed - predicted)", fontsize=7.5)
    ax_hist.set_ylabel("Density", fontsize=7.5)
    ax_hist.set_title(f"(C) Residual distribution\nMean = {resid_mean:+.3f},  SD = {resid_sd:.3f}", fontsize=8.8, fontweight="bold")
    add_grid(ax_hist)
    ax_hist.legend(loc="upper right", fontsize=6.2, frameon=True)

    # (D) residuals vs fitted.
    density_rvp = point_density(predicted, residual)
    order = np.argsort(density_rvp)
    ax_rvp.scatter(
        predicted[order],
        residual[order],
        c="#6baed6",
        s=2.1,
        alpha=0.42,
        linewidths=0,
        rasterized=True,
    )
    lx, ly = fitted_lowess(predicted, residual)
    ax_rvp.plot(lx, ly, color="#ef3b2c", linewidth=1.2, label="LOWESS trend")
    ax_rvp.axhline(0, color="black", linestyle="--", linewidth=0.8, label="Residual = 0")
    ax_rvp.axhline(resid_sd, color="#8f8f8f", linestyle=":", linewidth=0.8, label=f"+/-1 SD = {resid_sd:.3f}")
    ax_rvp.axhline(-resid_sd, color="#8f8f8f", linestyle=":", linewidth=0.8)
    ax_rvp.set_xlim(obs_pred_xlim)
    ax_rvp.set_ylim(resid_ylim)
    ax_rvp.set_xlabel("Predicted Resistance", fontsize=7.5)
    ax_rvp.set_ylabel("Residual (observed - predicted)", fontsize=7.5)
    ax_rvp.set_title("(D) Residuals vs. predicted Resistance\n(check for heteroscedasticity or systematic bias)", fontsize=8.8, fontweight="bold")
    add_grid(ax_rvp)
    ax_rvp.legend(loc="upper right", fontsize=6.4, frameon=True)

    for ax in [ax_obs, ax_hist, ax_rvp]:
        ax.tick_params(labelsize=6.8)

    fig.subplots_adjust(left=0.055, right=0.965, top=0.94, bottom=0.07, wspace=0.22, hspace=0.32)
    fig.savefig(OUT_DIR / f"{model_key}_fourpanel_residual_diagnostics_resistance.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{model_key}_fourpanel_residual_diagnostics_resistance.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "model": model_key,
        "n": float(len(df)),
        "r2": r2,
        "rmse": rmse,
        "bias_observed_minus_predicted": bias,
        "residual_mean": resid_mean,
        "residual_sd": resid_sd,
        "moran_i": moran_i,
        "moran_p": moran_p,
        "fit_slope": slope,
        "fit_intercept": intercept,
    }


def draw_residual_map_comparison(
    panels: list[tuple[pd.DataFrame, str, dict[str, float]]],
    residual_vmax: float,
    output_stem: str,
    figure_title: str = "Spatial distribution of residuals",
) -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 4.6), dpi=300)
    if len(panels) == 1:
        axes = [axes]

    cmap_resid = "RdBu"
    residual_norm = TwoSlopeNorm(vmin=-residual_vmax, vcenter=0.0, vmax=residual_vmax)
    last_sc = None

    for ax, (df, title, stats) in zip(axes, panels):
        last_sc = ax.scatter(
            df["x"],
            df["y"],
            c=df["residual"],
            cmap=cmap_resid,
            norm=residual_norm,
            s=2.8,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=8.8, fontweight="bold", pad=5)
        ax.text(
            0.03,
            0.96,
            f"$R^2$ = {stats['r2']:.3f}\n"
            f"RMSE = {stats['rmse']:.3f}\n"
            f"Bias = {stats['bias_observed_minus_predicted']:+.3f}\n"
            f"Moran's I = {stats['moran_i']:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox=dict(facecolor="white", edgecolor="#666666", linewidth=0.4, alpha=0.92, boxstyle="round,pad=0.25"),
        )

    assert last_sc is not None
    cbar = fig.colorbar(last_sc, ax=axes, fraction=0.028, pad=0.015)
    cbar.set_label("Residual (observed - predicted)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.8)

    fig.suptitle(figure_title, fontsize=9.2, fontweight="bold", y=0.97)
    fig.savefig(OUT_DIR / f"{output_stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.framealpha": 0.92,
        }
    )

    m2 = read_residuals(M2_BASELINE_RESID, "M2 baseline")
    rf = read_residuals(RF_RESID, "RF")
    mgwr = read_residuals(MGWR_RESID, "MGWR")
    all_resid = np.concatenate([m2["residual"].to_numpy(), rf["residual"].to_numpy(), mgwr["residual"].to_numpy()])
    residual_vmax = float(min(0.45, np.nanpercentile(np.abs(all_resid), 99)))
    resid_ylim = (-residual_vmax * 1.35, residual_vmax * 1.35)

    all_values = np.concatenate(
        [
            rf["observed"].to_numpy(),
            rf["predicted"].to_numpy(),
            m2["observed"].to_numpy(),
            m2["predicted"].to_numpy(),
            mgwr["observed"].to_numpy(),
            mgwr["predicted"].to_numpy(),
        ]
    )
    lower = float(max(0.0, np.nanpercentile(all_values, 0.2) - 0.02))
    upper = float(min(1.12, np.nanpercentile(all_values, 99.8) + 0.02))
    obs_pred_xlim = (lower, upper)

    m2_moran_i, m2_moran_p = make_knn_moran(m2, k=8)
    rf_moran_i, rf_moran_p = make_knn_moran(rf, k=8)
    mgwr_moran_i, mgwr_moran_p = make_knn_moran(mgwr, k=8)

    rows = [
        draw_fourpanel(m2, "m2_baseline", "M2 baseline RF", residual_vmax, obs_pred_xlim, resid_ylim, m2_moran_i, m2_moran_p),
        draw_fourpanel(rf, "best_rf", "RF", residual_vmax, obs_pred_xlim, resid_ylim, rf_moran_i, rf_moran_p),
        draw_fourpanel(mgwr, "mgwr", "MGWR", residual_vmax, obs_pred_xlim, resid_ylim, mgwr_moran_i, mgwr_moran_p),
    ]
    summary = pd.DataFrame(rows)
    summary["residual_vmax_for_map"] = residual_vmax
    summary["residual_ymin"] = resid_ylim[0]
    summary["residual_ymax"] = resid_ylim[1]
    summary["obs_pred_xmin"] = obs_pred_xlim[0]
    summary["obs_pred_xmax"] = obs_pred_xlim[1]
    summary.to_csv(OUT_DIR / "fourpanel_residual_diagnostics_summary.csv", index=False)

    stats_by_model = {row["model"]: row for row in rows}
    draw_residual_map_comparison(
        [
            (m2, "(A) M2 baseline RF", stats_by_model["m2_baseline"]),
            (rf, "(B) M2-based residual-corrected RF", stats_by_model["best_rf"]),
        ],
        residual_vmax,
        "m2_baseline_vs_best_rf_residual_maps_resistance",
        "Resistance residuals under 100-km spatial block validation",
    )
    draw_residual_map_comparison(
        [
            (rf, "(A) M2-based residual-corrected RF", stats_by_model["best_rf"]),
            (mgwr, "(B) MGWR fitted residuals", stats_by_model["mgwr"]),
        ],
        residual_vmax,
        "best_rf_vs_mgwr_residual_maps_resistance",
        "Resistance residuals from RF block prediction and MGWR fitting",
    )
    print(f"Saved outputs in: {OUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
