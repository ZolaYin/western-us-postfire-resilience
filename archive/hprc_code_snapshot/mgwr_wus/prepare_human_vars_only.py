import os
import numpy as np
import pandas as pd

# =========================
# Config
# =========================
INFILE = "df_mgwr_resistance_z.csv"   # 原始 y 版本数据
OUT_CSV = "df_mgwr_resistance_z_with_human_processed.csv"
OUT_PARQUET = "df_mgwr_resistance_z_with_human_processed.parquet"

# 只处理 human 变量，不自动加入 MGWR X 列表
HUMAN_RAW = [
    "HUM_imperv_near_t0",
    "HUM_popdens_near_t0",
    "HUM_viirs_near_t0"
]

# =========================
# Helper functions
# =========================
def safe_zscore(series: pd.Series) -> pd.Series:
    """z-score；如果标准差为 0，则返回全 0"""
    mu = series.mean()
    sd = series.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sd

def clip_upper(series: pd.Series, q: float = 0.99) -> pd.Series:
    """按上分位数截断"""
    upper = series.quantile(q)
    return series.clip(upper=upper)

def summarize_var(df: pd.DataFrame, col: str):
    s = df[col]
    print(f"\n--- {col} ---")
    print(s.describe())
    print(f"n_nan = {s.isna().sum()}")

# =========================
# Load data
# =========================
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

# =========================
# Check required columns
# =========================
missing = [c for c in HUMAN_RAW if c not in df.columns]
if missing:
    raise KeyError(f"Missing columns in input file: {missing}")

# 转 numeric
for c in HUMAN_RAW:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    df.loc[np.isinf(df[c]), c] = np.nan

print("\nRaw human variable summaries:")
for c in HUMAN_RAW:
    summarize_var(df, c)

# =========================
# Process human variables
# =========================

# 1) Impervious surface: log1p + z-score
df["HUM_imperv_near_t0_log1p"] = np.log1p(df["HUM_imperv_near_t0"].clip(lower=0))
df["HUM_imperv_near_t0_log1p_z"] = safe_zscore(df["HUM_imperv_near_t0_log1p"])

# 2) Population density: log1p + z-score
df["HUM_popdens_near_t0_log1p"] = np.log1p(df["HUM_popdens_near_t0"].clip(lower=0))
df["HUM_popdens_near_t0_log1p_z"] = safe_zscore(df["HUM_popdens_near_t0_log1p"])

# 3) VIIRS: clip p99 + z-score
df["HUM_viirs_near_t0_clip99"] = clip_upper(df["HUM_viirs_near_t0"].clip(lower=0), q=0.99)
df["HUM_viirs_near_t0_clip99_z"] = safe_zscore(df["HUM_viirs_near_t0_clip99"])

PROCESSED_COLS = [
    "HUM_imperv_near_t0_log1p_z",
    "HUM_popdens_near_t0_log1p_z",
    "HUM_viirs_near_t0_clip99_z"
]

print("\nProcessed human variable summaries:")
for c in PROCESSED_COLS:
    summarize_var(df, c)

# =========================
# Save
# =========================
df.to_csv(OUT_CSV, index=False)
df.to_parquet(OUT_PARQUET, index=False)

print("\nSaved:")
print("  CSV     :", OUT_CSV)
print("  Parquet :", OUT_PARQUET)

# =========================
# Print selectable variable names only
# =========================
print("\nYou can manually choose from these processed human variables later:")
for c in PROCESSED_COLS:
    print(c)

print("\nDone.")
