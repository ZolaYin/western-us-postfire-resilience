# ============================================================
# MGWR Residual Diagnostics (Spatial + Temporal)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from libpysal.weights import KNN
from esda.moran import Moran
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# ------------------------------------------------------------
# 1. Load MGWR result
# ------------------------------------------------------------

FILE = "outputs_resistance/MGWR_Resistance_results_bwmin20.parquet"
df = pd.read_parquet(FILE)

print("Loaded:", FILE)
print("Shape:", df.shape)
print("Columns:", df.columns[:20])

# ------------------------------------------------------------
# 2. Compute fitted + residual
# ------------------------------------------------------------

Y = "Resistance"
assert Y in df.columns, "Missing Resistance column"
assert "t0_year" in df.columns, "Missing t0_year"

bcols = [c for c in df.columns if c.startswith("b_")]
assert "b_Intercept" in bcols, "Missing intercept"

xcols = [c.replace("b_", "") for c in bcols if c != "b_Intercept"]

missing = [c for c in xcols if c not in df.columns]
if missing:
    raise ValueError(f"Missing X columns in parquet: {missing}")

y = df[Y].to_numpy()
y_hat = df["b_Intercept"].to_numpy().copy()

for c in xcols:
    y_hat += df[f"b_{c}"].to_numpy() * df[c].to_numpy()

df["residual"] = y - y_hat

print("\nResidual mean:", np.mean(df["residual"]))
print("Residual std :", np.std(df["residual"]))

# ------------------------------------------------------------
# 3. Spatial autocorrelation (Moran's I)
# ------------------------------------------------------------

coords = list(zip(df["x"], df["y"]))
w = KNN.from_array(coords, k=8)
w.transform = "r"

mi = Moran(df["residual"].values, w)

print("\n--- Spatial Moran's I ---")
print("I:", mi.I)
print("p-value:", mi.p_norm)

# ------------------------------------------------------------
# 4. Temporal autocorrelation (yearly mean residual)
# ------------------------------------------------------------

g = df.groupby("t0_year")["residual"].mean().reset_index()
g = g.sort_values("t0_year")

series = g["residual"].to_numpy()

print("\nYear range:", g["t0_year"].min(), "-", g["t0_year"].max())
print("Number of years:", len(series))

if len(series) > 2:
    r1 = np.corrcoef(series[1:], series[:-1])[0,1]
    print("\n--- Temporal Lag-1 ---")
    print("Lag-1 correlation:", r1)

    lb = acorr_ljungbox(series, lags=[1,2,3], return_df=True)
    print("\nLjung-Box test:")
    print(lb)
else:
    print("Not enough years for lag analysis")

# ------------------------------------------------------------
# 5. Save summary
# ------------------------------------------------------------

g.to_csv("outputs_resistance/temporal_mean_residual_bwmin20.csv", index=False)
print("\nSaved yearly residual summary.")
