import os
import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

# -----------------------
# Config
# -----------------------
INFILE = "df_mgwr_resistance_z_with_logy.csv"
OUT_DIR = "outputs_resistance"
os.makedirs(OUT_DIR, exist_ok=True)

Y_name = "Resistance_log"

X_model1 = [
    "CLIM_pr_gs_sum_mm_pre",
    "CLIM_vpd_gs_mean_kpa_pre",
    "FS_TCC_t0",
    "FS_AIproxy_t0",
    "TS_elev_m",
    "TS_slope_deg",
    "TS_SOC_0_30cm"
]
Xz = [c + "_z" for c in X_model1]

# More realistic fallback settings
KERNELS = ["bisquare"]
BW_MIN_CANDIDATES = [20, 40, 60, 80]
FIXED = False   # adaptive bandwidth

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(INFILE)

need = ["pixel_id", "row", "col", "t0_year", "sev", "x", "y", Y_name] + Xz
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")

# Keep only needed columns first, then reset index
df = df[need].copy().reset_index(drop=True)

coords = df[["x", "y"]].to_numpy(np.float64)
y_resp = df[[Y_name]].to_numpy(np.float64)
X = df[Xz].to_numpy(np.float64)

# Remove rows with nan/inf
bad = (
    np.isnan(coords).any(axis=1) |
    np.isnan(y_resp).any(axis=1) |
    np.isnan(X).any(axis=1) |
    np.isinf(coords).any(axis=1) |
    np.isinf(y_resp).any(axis=1) |
    np.isinf(X).any(axis=1)
)

if bad.any():
    print(f"Removing {bad.sum()} rows with NaN/Inf")
    df = df.loc[~bad].copy().reset_index(drop=True)
    coords = coords[~bad]
    y_resp = y_resp[~bad]
    X = X[~bad]

print(f"n = {coords.shape[0]} | k = {X.shape[1]}")
print(f"{Y_name} summary:")
print(df[Y_name].describe())

# Optional: quick check for duplicated coordinates
dup_coords = pd.DataFrame(coords, columns=["x", "y"]).duplicated().sum()
print(f"Duplicated coordinates: {dup_coords}")

# -----------------------
# Fit MGWR with fallback loop
# -----------------------
mgwr_res = None
chosen = None
bw = None
last_error = None

for ker in KERNELS:
    for bw_min in BW_MIN_CANDIDATES:
        try:
            print(f"\n--- Trying kernel={ker}, multi_bw_min={bw_min} ---")

            selector = Sel_BW(
                coords,
                y_resp,
                X,
                multi=True,
                fixed=FIXED,
                kernel=ker,
                constant=True
            )

            bw = selector.search(multi_bw_min=[bw_min])
            print("Selected bandwidths:", bw)
            print("Minimum selected bandwidth:", np.min(np.array(bw, dtype=float)))

            mgwr = MGWR(
                coords,
                y_resp,
                X,
                selector=selector,
                fixed=FIXED,
                kernel=ker,
                constant=True
            )

            mgwr_res = mgwr.fit()
            chosen = {
                "kernel": ker,
                "multi_bw_min": bw_min,
                "bandwidths": np.array(bw, dtype=float).tolist()
            }
            break

        except Exception as e:
            last_error = e
            print("Failed:", repr(e))
            mgwr_res = None
            chosen = None
            continue

    if mgwr_res is not None:
        break

if mgwr_res is None:
    raise RuntimeError(
        f"MGWR failed for all tried settings. Last error: {repr(last_error)}"
    )

print("\n✅ MGWR fitted.")
print("Chosen:", chosen)
print("AICc:", float(mgwr_res.aicc))
print("R2:", float(mgwr_res.R2))
print("Adj_R2:", float(mgwr_res.adj_R2))

# -----------------------
# Export results
# -----------------------
param_cols = ["Intercept"] + Xz

params = pd.DataFrame(mgwr_res.params, columns=param_cols)
tvals = pd.DataFrame(mgwr_res.tvalues, columns=param_cols)

out = df[["pixel_id", "x", "y", "row", "col", "t0_year", "sev", Y_name]].copy()
out = pd.concat(
    [
        out.reset_index(drop=True),
        params.add_prefix("b_").reset_index(drop=True),
        tvals.add_prefix("t_").reset_index(drop=True)
    ],
    axis=1
)

tag = f"{chosen['kernel']}_bwmin{chosen['multi_bw_min']}"
out_path = os.path.join(OUT_DIR, f"MGWR_Resistance_results_{tag}_logy.parquet")
txt_path = os.path.join(OUT_DIR, f"MGWR_Resistance_summary_{tag}_logy.txt")

out.to_parquet(out_path, index=False)

with open(txt_path, "w") as f:
    f.write("Chosen setting:\n")
    f.write(str(chosen) + "\n\n")
    f.write(f"n = {coords.shape[0]}, k = {X.shape[1]}\n\n")
    f.write(f"Y variable: {Y_name}\n\n")
    f.write("MGWR summary:\n")
    f.write(str(mgwr_res.summary()))

print("✅ Saved:", out_path)
print("✅ Saved:", txt_path)
