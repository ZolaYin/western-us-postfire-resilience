import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILE = "outputs_resistance/MGWR_Resistance_with_residual_bwmin20.parquet"
df = pd.read_parquet(FILE)

print("Loaded:", FILE)
print("Shape:", df.shape)

# -------------------------------
# 基本列名
# -------------------------------
res = df["residual"].to_numpy(dtype=float)
fit = df["fitted"].to_numpy(dtype=float)

year_col = "t0_year_beta" if "t0_year_beta" in df.columns else "t0_year"
sev_col  = "sev_beta" if "sev_beta" in df.columns else "sev"

# ============================================================
# 1️⃣ Residual vs Fitted + 分箱方差
# ============================================================

plt.figure(figsize=(8,6))
plt.scatter(fit, res, s=5, alpha=0.3)
plt.axhline(0, linestyle="--")
plt.xlabel("Fitted")
plt.ylabel("Residual")
plt.title("Residual vs Fitted")
plt.tight_layout()
plt.savefig("residual_vs_fitted_bwmin20.png", dpi=300)
plt.close()

# 分 10 桶
bins = pd.qcut(fit, 10, duplicates="drop")
var_by_bin = df.groupby(bins)["residual"].var().reset_index()

plt.figure(figsize=(8,6))
plt.plot(range(len(var_by_bin)), var_by_bin["residual"])
plt.xlabel("Fitted bins (low → high)")
plt.ylabel("Residual variance")
plt.title("Residual variance by fitted decile")
plt.tight_layout()
plt.savefig("residual_variance_bins_bwmin20.png", dpi=300)
plt.close()

print("Saved: residual_vs_fitted_bwmin20.png")
print("Saved: residual_variance_bins_bwmin20.png")

# ============================================================
# 2️⃣ 按年份 / severity 的 residual 方差
# ============================================================

# 年份
var_year = df.groupby(year_col)["residual"].var().reset_index()

plt.figure(figsize=(8,6))
plt.plot(var_year[year_col], var_year["residual"], marker="o")
plt.xlabel("Year")
plt.ylabel("Residual variance")
plt.title("Residual variance by year")
plt.tight_layout()
plt.savefig("residual_variance_by_year_bwmin20.png", dpi=300)
plt.close()

# severity
var_sev = df.groupby(sev_col)["residual"].var().reset_index()

plt.figure(figsize=(8,6))
plt.plot(var_sev[sev_col], var_sev["residual"], marker="o")
plt.xlabel("Severity")
plt.ylabel("Residual variance")
plt.title("Residual variance by severity")
plt.tight_layout()
plt.savefig("residual_variance_by_severity_bwmin20.png", dpi=300)
plt.close()

print("Saved: residual_variance_by_year_bwmin20.png")
print("Saved: residual_variance_by_severity_bwmin20.png")
