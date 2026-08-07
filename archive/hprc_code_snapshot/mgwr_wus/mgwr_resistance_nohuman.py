import os
import numpy as np
import pandas as pd

from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

# -----------------------
# Config
# -----------------------
IN_PARQUET = "df_mgwr_resistance_z.parquet"
OUT_DIR = "outputs_resistance_nohuman"
os.makedirs(OUT_DIR, exist_ok=True)

Y_name = "Resistance"
X_model1 = [
    "CLIM_pr_gs_sum_mm_pre",
    "CLIM_vpd_gs_mean_kpa_pre",
    "FS_TCC_t0",
    "FS_AIproxy_t0",
    "TS_elev_m",
    "TS_slope_deg",
    "TS_SOC_0_30cm",
]
Xz = [c + "_z" for c in X_model1]

# Try settings (more stable first)
KERNELS = ["gaussian", "bisquare"]
BW_MIN_CANDIDATES = [200, 300, 400, 600]  # adaptive neighbor minimums
FIXED = False  # adaptive bandwidth

# -----------------------
# Load data
# -----------------------
df = pd.read_parquet(IN_PARQUET)

need = ["pixel_id", "row", "col", "t0_year", "sev", "x", "y", Y_name] + Xz
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns: {missing}")

coords = df[["x", "y"]].to_numpy(np.float64)
y = df[[Y_name]].to_numpy(np.float64)
X = df[Xz].to_numpy(np.float64)

bad = (
    np.isnan(coords).any(axis=1) | np.isnan(y).any(axis=1) | np.isnan(X).any(axis=1) |
    np.isinf(coords).any(axis=1) | np.isinf(y).any(axis=1) | np.isinf(X).any(axis=1)
)
if bad.any():
    df = df.loc[~bad].copy()
    coords = coords[~bad]
    y = y[~bad]
    X = X[~bad]

print(f"n={coords.shape[0]} | k={X.shape[1]}")

# -----------------------
# Fit MGWR with fallback loop
# -----------------------
mgwr_res = None
chosen = None
bw = None

for ker in KERNELS:
    for bw_min in BW_MIN_CANDIDATES:
        try:
            print(f"\n--- Trying kernel={ker}, multi_bw_min={bw_min} ---")
            selector = Sel_BW(
                coords, y, X,
                multi=True,
                fixed=FIXED,
                kernel=ker,
                constant=True
            )
            bw = selector.search(multi_bw_min=[bw_min])
            print("bandwidths:", bw)

            mgwr = MGWR(
                coords, y, X,
                selector=selector,
                fixed=FIXED,
                kernel=ker,
                constant=True
            )
            mgwr_res = mgwr.fit()
            chosen = {"kernel": ker, "multi_bw_min": bw_min, "bandwidths": bw}
            break
        except Exception as e:
            print("Failed:", repr(e))
            mgwr_res = None
            chosen = None
            continue
    if mgwr_res is not None:
        break

if mgwr_res is None:
    raise RuntimeError("MGWR failed for all tried settings. Consider increasing bw_min or reducing X.")

print("\n✅ MGWR fitted.")
print("Chosen:", chosen)
print("AICc:", float(mgwr_res.aicc))
print("R2:", float(mgwr_res.R2))
print("Adj_R2:", float(mgwr_res.adj_R2))

# -----------------------
# Export results
# -----------------------
param_cols = ["Intercept"] + Xz
params = pd.DataFrame(mgwr_res.params, columns=param_cols, index=df.index)
tvals  = pd.DataFrame(mgwr_res.tvalues, columns=param_cols, index=df.index)
localR2 = pd.Series(mgwr_res.localR2.flatten(), index=df.index, name="localR2")

out = df[["pixel_id", "x", "y", "row", "col", "t0_year", "sev", Y_name]].copy()
out = pd.concat([out, params.add_prefix("b_"), tvals.add_prefix("t_"), localR2], axis=1)

out_path = os.path.join(OUT_DIR, "MGWR_Resistance_results.parquet")
out.to_parquet(out_path, index=False)

txt_path = os.path.join(OUT_DIR, "MGWR_Resistance_summary.txt")
with open(txt_path, "w") as f:
    f.write("Chosen setting:\n")
    f.write(str(chosen) + "\n\n")
    f.write(str(mgwr_res.summary()))

print("✅ Saved:", out_path)
print("✅ Saved:", txt_path)

