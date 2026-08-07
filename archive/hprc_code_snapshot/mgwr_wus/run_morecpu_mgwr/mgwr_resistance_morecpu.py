import os
import numpy as np
import pandas as pd

from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

IN_PARQUET = "df_mgwr_resistance_z.parquet"
OUT_DIR = "outputs_resistance"
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

# ----- threads / jobs -----
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
# 这些环境变量让 numpy/scipy/mkl 不要“每进程再开线程”，避免乱套
os.environ["OMP_NUM_THREADS"] = str(n_jobs)
os.environ["MKL_NUM_THREADS"] = str(n_jobs)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_jobs)
os.environ["NUMEXPR_NUM_THREADS"] = str(n_jobs)

print(f"Using n_jobs={n_jobs}")

# ----- load -----
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

# 先从更稳定的 gaussian 开始
KERNELS = ["gaussian", "bisquare"]
BW_MIN_CANDIDATES = [200, 300, 400, 600]

mgwr_res = None
chosen = None

for ker in KERNELS:
    for bw_min in BW_MIN_CANDIDATES:
        try:
            print(f"\n--- Trying kernel={ker}, multi_bw_min={bw_min} ---")

            # 【未经验证】不同 mgwr 版本 Sel_BW/MGWR 是否支持 n_jobs 参数可能不同；
            # 这里用“尽量传入”的写法，不支持就会抛 TypeError，我们会捕获并退回不传 n_jobs。
            try:
                selector = Sel_BW(coords, y, X, multi=True, fixed=False, kernel=ker, constant=True, n_jobs=n_jobs)
            except TypeError:
                selector = Sel_BW(coords, y, X, multi=True, fixed=False, kernel=ker, constant=True)

            bw = selector.search(multi_bw_min=[bw_min])
            print("bandwidths:", bw)

            try:
                mgwr = MGWR(coords, y, X, selector=selector, fixed=False, kernel=ker, constant=True, n_jobs=n_jobs)
            except TypeError:
                mgwr = MGWR(coords, y, X, selector=selector, fixed=False, kernel=ker, constant=True)

            mgwr_res = mgwr.fit()
            chosen = {"kernel": ker, "multi_bw_min": bw_min, "bandwidths": bw}
            break

        except Exception as e:
            print("Failed:", repr(e))
            mgwr_res = None
            chosen = None

    if mgwr_res is not None:
        break

if mgwr_res is None:
    raise RuntimeError("MGWR failed for all settings.")

print("\nMGWR fitted.")
print("Chosen:", chosen)
print("AICc:", float(mgwr_res.aicc))
print("R2:", float(mgwr_res.R2))
print("Adj_R2:", float(mgwr_res.adj_R2))

param_cols = ["Intercept"] + Xz
params = pd.DataFrame(mgwr_res.params, columns=param_cols, index=df.index)
tvals  = pd.DataFrame(mgwr_res.tvalues, columns=param_cols, index=df.index)
localR2 = pd.Series(mgwr_res.localR2.flatten(), index=df.index, name="localR2")

out = df[["pixel_id","x","y","row","col","t0_year","sev",Y_name]].copy()
out = pd.concat([out, params.add_prefix("b_"), tvals.add_prefix("t_"), localR2], axis=1)

out_path = os.path.join(OUT_DIR, "MGWR_Resistance_results.parquet")
out.to_parquet(out_path, index=False)

txt_path = os.path.join(OUT_DIR, "MGWR_Resistance_summary.txt")
with open(txt_path, "w") as f:
    f.write("Chosen:\n")
    f.write(str(chosen) + "\n\n")
    f.write(str(mgwr_res.summary()))

print("Saved:", out_path)
print("Saved:", txt_path)
