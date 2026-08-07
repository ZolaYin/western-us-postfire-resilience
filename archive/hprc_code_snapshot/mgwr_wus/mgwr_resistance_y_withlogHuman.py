import os
import numpy as np
import pandas as pd
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

# -----------------------
# Config
# -----------------------
INFILE = "df_mgwr_resistance_z.csv"   # 保持原始 mgwr 数据
OUT_DIR = "outputs_resistance_human_inside"
os.makedirs(OUT_DIR, exist_ok=True)

Y_name = "Resistance"

# -----------------------
# Base predictors
# -----------------------
BASE_X = [
    "CLIM_pr_gs_sum_mm_pre",
    "CLIM_vpd_gs_mean_kpa_pre",
    "FS_TCC_t0",
    "FS_AIproxy_t0",
    "TS_elev_m",
    "TS_slope_deg",
    "TS_SOC_0_30cm"
]

# -----------------------
# Human variables to process inside script：
#   "HUM_imperv_near_t0_log1p_z"
#   "HUM_popdens_near_t0_log1p_z"
#   "HUM_viirs_near_t0_clip99_z"
# -----------------------
HUMAN_X = [
    "HUM_popdens_near_t0_log1p_z",
    "HUM_imperv_near_t0_log1p_z",
    "HUM_viirs_near_t0_clip99_z"
]

# -----------------------
# Fallback settings
# -----------------------
KERNELS = ["bisquare", "gaussian"]
BW_MIN_CANDIDATES = [100, 150]
FIXED = False  # adaptive bandwidth

# =======================
# Helper functions
# =======================
def safe_zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sd = series.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sd

def clip_upper(series: pd.Series, q: float = 0.99) -> pd.Series:
    upper = series.quantile(q)
    return series.clip(upper=upper)

# -----------------------
# Load data
# -----------------------
if not os.path.exists(INFILE):
    raise FileNotFoundError(f"Input file not found: {INFILE}")

if INFILE.endswith(".csv"):
    df = pd.read_csv(INFILE)
elif INFILE.endswith(".parquet"):
    df = pd.read_parquet(INFILE)
else:
    raise ValueError("INFILE must be .csv or .parquet")

print("Loaded:", INFILE)
print("Shape:", df.shape)

# -----------------------
# Process human variables INSIDE script
# 保留原始数据列，同时新增处理后的列
# -----------------------
raw_human_needed = [
    "HUM_imperv_near_t0",
    "HUM_popdens_near_t0",
    "HUM_viirs_near_t0"
]
missing_human = [c for c in raw_human_needed if c not in df.columns]
if missing_human:
    raise KeyError(f"Missing human raw columns: {missing_human}")

for c in raw_human_needed:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[np.isinf(df[c]), c] = np.nan

# 1) Impervious: log1p + z
df["HUM_imperv_near_t0_log1p"] = np.log1p(df["HUM_imperv_near_t0"].clip(lower=0))
df["HUM_imperv_near_t0_log1p_z"] = safe_zscore(df["HUM_imperv_near_t0_log1p"])

# 2) Population density: log1p + z
df["HUM_popdens_near_t0_log1p"] = np.log1p(df["HUM_popdens_near_t0"].clip(lower=0))
df["HUM_popdens_near_t0_log1p_z"] = safe_zscore(df["HUM_popdens_near_t0_log1p"])

# 3) VIIRS: clip p99 + z
df["HUM_viirs_near_t0_clip99"] = clip_upper(df["HUM_viirs_near_t0"].clip(lower=0), q=0.99)
df["HUM_viirs_near_t0_clip99_z"] = safe_zscore(df["HUM_viirs_near_t0_clip99"])

print("\nProcessed human variables created inside script:")
print([
    "HUM_imperv_near_t0_log1p_z",
    "HUM_popdens_near_t0_log1p_z",
    "HUM_viirs_near_t0_clip99_z"
])

# -----------------------
# Build final predictor list automatically
# -----------------------
X_model1 = BASE_X + HUMAN_X
print("\nFinal X_model1:")
print(X_model1)

# -----------------------
# Required columns
# -----------------------
need = ["pixel_id", "row", "col", "t0_year", "sev", "x", "y", Y_name] + X_model1
missing = [c for c in need if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns for MGWR: {missing}")

df = df[need].copy().reset_index(drop=True)

coords = df[["x", "y"]].to_numpy(np.float64)
y_resp = df[[Y_name]].to_numpy(np.float64)
X = df[X_model1].to_numpy(np.float64)

# -----------------------
# Remove bad rows
# -----------------------
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

print(f"\nn = {coords.shape[0]} | k = {X.shape[1]}")
print(f"{Y_name} summary:")
print(df[Y_name].describe())

dup_coords = pd.DataFrame(coords, columns=["x", "y"]).duplicated().sum()
print(f"Duplicated coordinates: {dup_coords}")

# -----------------------
# Correlation matrix of final X
# -----------------------
print("\n=== Correlation matrix of final X ===")
corr_df = df[X_model1].corr()
print(corr_df.round(3))
corr_path = os.path.join(OUT_DIR, "X_correlation_matrix_withHuman.csv")
corr_df.to_csv(corr_path)

# -----------------------
# Fit MGWR with fallback loop
# -----------------------
mgwr_res = None
chosen = None
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
            bw_array = np.array(bw, dtype=float)

            print("Selected bandwidths:", bw_array)
            print("Minimum selected bandwidth:", float(np.min(bw_array)))
            print("Maximum selected bandwidth:", float(np.max(bw_array)))

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
                "bandwidths": bw_array.tolist(),
                "human_x": HUMAN_X
            }

            print("\n✅ MGWR fitted successfully.")
            print("Chosen:", chosen)
            print("AICc:", float(mgwr_res.aicc))
            print("R2:", float(mgwr_res.R2))
            print("Adj_R2:", float(mgwr_res.adj_R2))
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

# -----------------------
# Export results
# -----------------------
param_cols = ["Intercept"] + X_model1
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

human_tag = "noHuman" if len(HUMAN_X) == 0 else "_".join(HUMAN_X)
tag = f"{chosen['kernel']}_bwmin{chosen['multi_bw_min']}"

out_path = os.path.join(OUT_DIR, f"MGWR_Resistance_results_{tag}_{human_tag}.parquet")
csv_path = os.path.join(OUT_DIR, f"MGWR_Resistance_results_{tag}_{human_tag}.csv")
txt_path = os.path.join(OUT_DIR, f"MGWR_Resistance_summary_{tag}_{human_tag}.txt")

out.to_parquet(out_path, index=False)
out.to_csv(csv_path, index=False)

with open(txt_path, "w") as f:
    f.write("Chosen setting:\n")
    f.write(str(chosen) + "\n\n")
    f.write(f"Y variable: {Y_name}\n")
    f.write(f"Predictors: {X_model1}\n\n")
    f.write(f"n = {coords.shape[0]}, k = {X.shape[1]}\n\n")
    f.write("MGWR summary:\n")
    f.write(str(mgwr_res.summary()))

print("\n✅ Saved correlation matrix:", corr_path)
print("✅ Saved parquet:", out_path)
print("✅ Saved csv:", csv_path)
print("✅ Saved summary:", txt_path)
