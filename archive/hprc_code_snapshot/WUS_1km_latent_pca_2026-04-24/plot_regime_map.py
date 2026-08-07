import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

df = pd.read_csv("/scratch/user/YOUR_NETID/WUS_1km_latent_pca_2026-04-24/latent_regime_pca_fixed_8pcs_2026-04-24/kmeans_k6_assignments.csv")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

colors = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]
cmap = mcolors.ListedColormap(colors)

# full map
sc = axes[0].scatter(df["x"], df["y"], c=df["kmeans_k6"], cmap=cmap,
                     s=0.3, alpha=0.6, vmin=0, vmax=5)
axes[0].set_title("Latent Regime Map (k=6, full dataset)", fontsize=13)
axes[0].set_xlabel("x (EPSG:5070)"); axes[0].set_ylabel("y (EPSG:5070)")
axes[0].set_aspect("equal")
plt.colorbar(sc, ax=axes[0], label="Regime", ticks=range(6))

# resistance by regime (boxplot)
regime_groups = [df[df["kmeans_k6"]==k]["Resistance"].values for k in range(6)]
bp = axes[1].boxplot(regime_groups, patch_artist=True, medianprops=dict(color="black",linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1].set_xlabel("Regime"); axes[1].set_ylabel("Resistance")
axes[1].set_title("Resistance distribution by latent regime (k=6)", fontsize=13)
axes[1].set_xticklabels([f"R{k}" for k in range(6)])

plt.tight_layout()
plt.savefig("/scratch/user/YOUR_NETID/WUS_1km_latent_pca_2026-04-24/regime_k6_map_resistance.png", dpi=150, bbox_inches="tight")
print("saved")
