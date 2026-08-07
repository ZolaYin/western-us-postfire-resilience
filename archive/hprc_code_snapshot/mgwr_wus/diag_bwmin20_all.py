import pandas as pd
import numpy as np

from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm

from libpysal.weights import KNN
from esda.moran import Moran

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
FILE = "outputs_resistance/MGWR_Resistance_with_residual_bwmin20.parquet"
df = pd.read_parquet(FILE)

print("Loaded:", FILE)
print("Shape:", df.shape)

# Basic check
assert "residual" in df.columns
assert "fitted" in df.columns
assert "t0_year_beta" in df.columns or "t0_year" in df.columns

# unify year column name
year_col = "t0_year_beta" if "t0_year_beta" in df.columns else "t0_year"

# ------------------------------------------------------------
# 1) Temporal autocorrelation on yearly mean residual
# ------------------------------------------------------------
g = df.groupby(year_col)["residual"].mean().reset_index().sort_values(year_col)
s = g["residual"].to_numpy()

print("\n=== Temporal residual autocorrelation ===")
print("Year range:", int(g[year_col].min()), "-", int(g[year_col].max()))
print("n years:", len(s))

if len(s) >= 3:
    r1 = float(np.corrcoef(s[1:], s[:-1])[0,1])
    print("Lag-1 corr (annual mean residual):", r1)

    nl = min(10, len(s)-1)
    print("ACF (0..{}):".format(nl), [float(x) for x in acf(s, nlags=nl, fft=False)])

    lb = acorr_ljungbox(s, lags=[1,2,3], return_df=True)
    print("\nLjung-Box (annual mean residual):")
    print(lb)
else:
    print("Not enough years for temporal test.")

# ------------------------------------------------------------
# 2) Spatial autocorrelation on residual (Moran's I)
# ------------------------------------------------------------
print("\n=== Spatial autocorrelation (Moran's I) on residual ===")
coords = np.column_stack([df["x_beta"] if "x_beta" in df.columns else df["x"],
                          df["y_beta"] if "y_beta" in df.columns else df["y"]])

w = KNN.from_array(coords, k=8)
w.transform = "r"

mi = Moran(df["residual"].to_numpy(dtype=float), w)
print("Moran's I:", float(mi.I))
print("p-value (normal approx):", float(mi.p_norm))

# ------------------------------------------------------------
# 3) Homoscedasticity check (Breusch–Pagan) using fitted values
#    (simple + robust; BP is sensitive with big n, but useful)
# ------------------------------------------------------------
print("\n=== Heteroskedasticity (Breusch–Pagan) ===")
X_bp = sm.add_constant(df["fitted"].to_numpy(dtype=float))
bp = het_breuschpagan(df["residual"].to_numpy(dtype=float), X_bp)

print("LM stat   :", float(bp[0]))
print("LM p-value:", float(bp[1]))
print("F stat    :", float(bp[2]))
print("F p-value :", float(bp[3]))
