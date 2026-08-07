import os
import numpy as np
import pandas as pd

from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

IN_PARQUET = "df_mgwr_resistance_z.parquet"
OUT_DIR = "outputs_gwr_fast"
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
    "HUM_imperv_near_t0",
    "HUM_popdens_near_t0",
    "HUM_viirs_near_t0",
]
Xz = [c + "_z" for c in X_model1]
NEED = ["x", "y", Y_name] + Xz + ["pixel_id","row","col","t0_year","sev"]

n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ["MKL_NUM_THREADS"] = str(n_jobs)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_jobs)

print(f"Using n_jobs={n_jobs}")

df = pd.read_parquet(IN_PARQUET)
miss = [c for c in NEED if c not in df.columns]
if miss:
    raise KeyError(f"Missing columns: {miss}")

coords = df[["x","y"]].to_numpy(np.float64)
y = df[[Y_name]].to_numpy(np.float64)
X = df[Xz].to_numpy(np.float64)

bad = (
    np.isnan(coords).any(axis=1) | np.isnan(y).any(axis=1) | np.isnan(X).any(axis=1) |
    np.isinf(coords).any(axis=1) | np.isinf(y).any(axis=1) | np.isinf(X).any(axis=1)
)
if bad.any():
    print("Dropping bad rows:", int(bad.sum()))
    df = df.loc[~bad].copy()
    coords = coords[~bad]
    y = y[~bad]
    X = X[~bad]

print(f"n={coords.shape[0]} | k={X.shape[1]}")

# GWR: 一个带宽
kernel = "gaussian"  # 通常更稳
fixed = False        # adaptive neighbors

try:
    selector = Sel_BW(coords, y, X, multi=False, fixed=fixed, kernel=kernel, constant=True, n_jobs=n_jobs)
except TypeError:
    selector = Sel_BW(coords, y, X, multi=False, fixed=fixed, kernel=kernel, constant=True)

bw = selector.search()
print("Chosen GWR bw:", bw)

try:
    gwr = GWR(coords, y, X, bw=bw, fixed=fixed, kernel=kernel, constant=True, n_jobs=n_jobs)
except TypeError:
    gwr = GWR(coords, y, X, bw=bw, fixed=fixed, kernel=kernel, constant=True)

res = gwr.fit()
print("GWR fitted.")
print("AICc:", float(res.aicc))
print("R2:", float(res.R2))
print("Adj_R2:", float(res.adj_R2))

param_cols = ["Intercept"] + Xz
params = pd.DataFrame(res.params, columns=param_cols, index=df.index)
tvals  = pd.DataFrame(res.tvalues, columns=param_cols, index=df.index)
localR2 = pd.Series(res.localR2.flatten(), index=df.index, name="localR2")

out = df[["pixel_id","x","y","row","col","t0_year","sev",Y_name]].copy()
out = pd.concat([out, params.add_prefix("b_"), tvals.add_prefix("t_"), localR2], axis=1)

out_path = os.path.join(OUT_DIR, "GWR_Resistance_results.parquet")
out.to_parquet(out_path, index=False)

txt_path = os.path.join(OUT_DIR, "GWR_Resistance_summary.txt")
with open(txt_path, "w") as f:
    f.write(f"kernel={kernel}, fixed={fixed}, bw={bw}\n\n")
    f.write(str(res.summary()))

print("Saved:", out_path)
print("Saved:", txt_path)
