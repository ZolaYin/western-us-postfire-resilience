import os
import numpy as np
import pandas as pd

from libpysal.weights import KNN
from esda.moran import Moran

# ----------------------------
# User settings
# ----------------------------
PARQUET_PATH = "outputs_resistance/MGWR_Resistance_with_residual_bwmin20.parquet"

# k for KNN graph (try 8, 12, 20, 40)
K = 8

# how many largest components to compute Moran's I for
TOP_N = 15

# permutations for Moran p-value (999 is standard; increase if you want)
PERMUTATIONS = 999

# output
OUT_DIR = "diag_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = os.path.join(OUT_DIR, f"moran_by_component_k{K}_top{TOP_N}.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, f"moran_by_component_k{K}_summary.txt")


def connected_components_from_knn(neighbors_dict: dict, n: int):
    """
    Compute connected component labels from a neighbor dictionary.
    neighbors_dict: {i: [neighbors...]} for i in [0..n-1]
    Return: labels (n,), comp_sizes dict
    """
    visited = np.zeros(n, dtype=bool)
    labels = -np.ones(n, dtype=int)
    comp_id = 0
    comp_sizes = {}

    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS
        stack = [i]
        visited[i] = True
        labels[i] = comp_id
        size = 0

        while stack:
            u = stack.pop()
            size += 1
            for v in neighbors_dict.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    labels[v] = comp_id
                    stack.append(v)

        comp_sizes[comp_id] = size
        comp_id += 1

    return labels, comp_sizes


def main():
    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded: {PARQUET_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # ---- column checks ----
    # ---- detect coordinate columns ----
    candidates = [("x", "y"), ("x_X", "y_X"), ("x_beta", "y_beta")]
    xy = None
    for cx, cy in candidates:
        if (cx in df.columns) and (cy in df.columns):
            xy = (cx, cy)
            break

    if xy is None:
        raise ValueError(f"Cannot find coordinate columns. Tried: {candidates}. "
                         f"Available cols: {list(df.columns)[:30]} ...")

    XCOL, YCOL = xy
    print(f"Using coord columns: {XCOL}, {YCOL}")

    # residual column detection
    if "residual" in df.columns:
        res_col = "residual"
    else:
        # fall back: try common alternatives
        candidates = [c for c in df.columns if "resid" in c.lower()]
        if len(candidates) == 1:
            res_col = candidates[0]
        else:
            raise ValueError(
                f"Cannot find residual column. Found candidates={candidates}. "
                f"Please rename residual column to 'residual' or edit script."
            )

    # drop NA / inf
    df = df[["x", "y", res_col]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.reset_index(drop=True)
    n = len(df)
    print(f"After clean: n={n}")

    coords = df[["x", "y"]].to_numpy()
    y = df[res_col].to_numpy().astype(float)

    # ---- build KNN weights ----
    w = KNN.from_array(coords, k=K)
    # Symmetrize to ensure undirected connectivity
    w = w.symmetrize()

    # ---- compute connected components ----
    # neighbors dict from weights
    neighbors = w.neighbors  # dict: i -> list
    labels, comp_sizes = connected_components_from_knn(neighbors, n)

    n_components = len(comp_sizes)
    sizes_sorted = sorted(comp_sizes.items(), key=lambda x: x[1], reverse=True)
    top_sizes = [s for _, s in sizes_sorted[:10]]
    largest = sizes_sorted[0][1]
    largest_frac = largest / n

    print(f"\n== Connectivity (KNN k={K}, symmetrized) ==")
    print(f"n_components = {n_components}")
    print(f"largest_component_size = {largest} ({largest_frac:.3f} of all points)")
    print(f"top10_component_sizes = {top_sizes}")

    # ---- Moran's I per component ----
    # Prepare results rows
    rows = []
    # take top N component ids
    top_comp_ids = [cid for cid, _ in sizes_sorted[:TOP_N]]

    for rank, cid in enumerate(top_comp_ids, start=1):
        idx = np.where(labels == cid)[0]
        m = len(idx)
        if m < (K + 2):
            # too small to do meaningful KNN within component
            rows.append({
                "rank": rank,
                "component_id": cid,
                "n_points": m,
                "moran_I": np.nan,
                "p_sim": np.nan,
                "z_sim": np.nan,
                "note": f"too small (n<{K+2})"
            })
            continue

        # subset
        coords_c = coords[idx]
        y_c = y[idx]

        # IMPORTANT: build weights within component (avoid cross-component edges)
        w_c = KNN.from_array(coords_c, k=min(K, m-1))
        w_c = w_c.symmetrize()

        mor = Moran(y_c, w_c, permutations=PERMUTATIONS)
        rows.append({
            "rank": rank,
            "component_id": cid,
            "n_points": m,
            "moran_I": float(mor.I),
            "p_sim": float(mor.p_sim),
            "z_sim": float(mor.z_sim),
            "note": ""
        })

    res = pd.DataFrame(rows)

    # ---- weighted summary ----
    # weighted mean I across computed components (exclude NaN)
    valid = res.dropna(subset=["moran_I"]).copy()
    if len(valid) > 0:
        w_mean_I = np.average(valid["moran_I"], weights=valid["n_points"])
    else:
        w_mean_I = np.nan

    # share covered by TOP_N components
    covered_n = int(sum([s for _, s in sizes_sorted[:TOP_N]]))
    covered_frac = covered_n / n

    # save
    res.to_csv(OUT_CSV, index=False)
    with open(OUT_SUMMARY, "w") as f:
        f.write(f"PARQUET_PATH: {PARQUET_PATH}\n")
        f.write(f"K (KNN): {K}\n")
        f.write(f"TOP_N: {TOP_N}\n")
        f.write(f"PERMUTATIONS: {PERMUTATIONS}\n\n")
        f.write("Connectivity:\n")
        f.write(f"  n_points: {n}\n")
        f.write(f"  n_components: {n_components}\n")
        f.write(f"  largest_component_size: {largest}\n")
        f.write(f"  largest_component_frac: {largest_frac:.6f}\n")
        f.write(f"  top10_component_sizes: {top_sizes}\n\n")
        f.write("Moran by component (top N):\n")
        f.write(res.to_string(index=False))
        f.write("\n\n")
        f.write("Weighted summary:\n")
        f.write(f"  weighted_mean_I (over valid comps in top N): {w_mean_I}\n")
        f.write(f"  topN_covered_points: {covered_n}\n")
        f.write(f"  topN_covered_frac: {covered_frac:.6f}\n")

    print(f"\nSaved CSV: {OUT_CSV}")
    print(f"Saved summary: {OUT_SUMMARY}")
    print("\nTop N Moran results:")
    print(res)


if __name__ == "__main__":
    main()
