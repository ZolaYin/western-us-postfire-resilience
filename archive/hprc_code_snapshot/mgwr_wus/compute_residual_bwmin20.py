import pandas as pd
import numpy as np

X_FILE = "df_mgwr_resistance_z.parquet"
B_FILE = "outputs_resistance/MGWR_Resistance_results_bwmin20.parquet"
OUT_FILE = "outputs_resistance/MGWR_Resistance_with_residual_bwmin20.parquet"

df_X = pd.read_parquet(X_FILE)
df_b = pd.read_parquet(B_FILE)

print("X shape:", df_X.shape)
print("Beta shape:", df_b.shape)

# merge with explicit suffixes (so names are predictable)
df = df_b.merge(df_X, on="pixel_id", how="left", suffixes=("_beta", "_X"))
print("Merged shape:", df.shape)

# ---- auto pick Y column ----
res_cols = [c for c in df.columns if c.lower().startswith("resistance")]
print("Resistance-like cols:", res_cols)

# Prefer the observed Y from X table
if "Resistance_X" in df.columns:
    Ycol = "Resistance_X"
elif "Resistance" in df_X.columns and "Resistance" in df.columns:
    # if no suffix happened for some reason
    Ycol = "Resistance"
elif "Resistance_beta" in df.columns:
    # fallback: use beta copy (should match observed, but not preferred)
    Ycol = "Resistance_beta"
else:
    raise KeyError(f"Cannot find Resistance column after merge. Found: {res_cols}")

print("Using Y column:", Ycol)

# ---- compute fitted & residual ----
bcols = [c for c in df.columns if c.startswith("b_")]
if "b_Intercept" not in bcols:
    raise KeyError("Missing b_Intercept in MGWR results parquet")

xcols = [c.replace("b_", "") for c in bcols if c != "b_Intercept"]

missing_X = [c for c in xcols if c not in df.columns]
if missing_X:
    raise KeyError(f"Missing X columns after merge: {missing_X[:10]} ... total {len(missing_X)}")

y_hat = df["b_Intercept"].to_numpy(dtype=float).copy()
for c in xcols:
    y_hat += df[f"b_{c}"].to_numpy(dtype=float) * df[c].to_numpy(dtype=float)

df["fitted"] = y_hat
df["residual"] = df[Ycol].to_numpy(dtype=float) - y_hat

print("Residual mean:", float(np.mean(df["residual"])))
print("Residual std :", float(np.std(df["residual"])))

df.to_parquet(OUT_FILE, index=False)
print("Saved:", OUT_FILE)
