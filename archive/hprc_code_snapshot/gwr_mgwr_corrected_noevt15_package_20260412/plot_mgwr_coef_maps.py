#!/usr/bin/env python3
"""
Plot MGWR coefficient spatial maps.

Reads mgwr_coefficients.parquet and mgwr_bandwidths.csv produced by
run_mgwr_corrected_noevt15.py and saves a clean grid of coefficient maps.

Usage:
  python plot_mgwr_coef_maps.py \
    --coef-file  <dir>/mgwr_coefficients.parquet \
    --bw-file    <dir>/mgwr_bandwidths.csv \
    --output     stage5b_coef_maps.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coef-file", required=True, help="mgwr_coefficients.parquet")
    p.add_argument("--bw-file",   required=True, help="mgwr_bandwidths.csv")
    p.add_argument("--output",    required=True, help="output PNG path")
    p.add_argument("--ncols",     type=int, default=3)
    p.add_argument("--dpi",       type=int, default=200)
    p.add_argument("--s",         type=float, default=2.0, help="scatter point size")
    return p.parse_args()


def main():
    args = parse_args()

    coef = pd.read_parquet(args.coef_file)
    bw   = pd.read_csv(args.bw_file).set_index("term")["bandwidth"].to_dict()

    x = coef["x"].to_numpy()
    y = coef["y"].to_numpy()

    # all coefficient columns = everything except x, y
    terms = [c for c in coef.columns if c not in ("x", "y")]

    ncols = args.ncols
    nrows = int(np.ceil(len(terms) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5, nrows * 4.5),
                             constrained_layout=True)
    axes = np.array(axes).ravel()

    for i, term in enumerate(terms):
        ax = axes[i]
        vals = coef[term].to_numpy()

        # clip to 2nd–98th percentile for color range
        lo, hi = np.nanpercentile(vals, [2, 98])
        vmax = max(abs(lo), abs(hi))
        vmin = -vmax

        sc = ax.scatter(x, y, c=vals, s=args.s,
                        cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)

        bw_val = bw.get(term, "?")
        ax.set_title(f"{term}\n(bw={bw_val})", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")

    # hide unused axes
    for j in range(len(terms), len(axes)):
        axes[j].set_visible(False)

    # NO suptitle — title info is already in each subplot

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
