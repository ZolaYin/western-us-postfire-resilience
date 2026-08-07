import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTDIR = "/scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_package_20260412/mgwr_outputs_stage5b_reduced_pr_tmmn_11_s12k"

coef = pd.read_parquet(f"{OUTDIR}/mgwr_coefficients.parquet")
bw = pd.read_csv(f"{OUTDIR}/mgwr_bandwidths.csv")

VARS = ['Intercept','TS_elev_m_z','TS_slope_deg_z','TS_SOC_0_30cm_clean_z',
        'FS_TCC_t0_z','FS_CBH_t0agg_z','HUM_viirs_near_t0_log_z',
        'HUM_imperv_near_t0_z','HUM_roaddens_r5km_z','HUM_traildens_r10km_z',
        'CLIM_pr_sum_pre_z','CLIM_tmmn_mean_pre_z']

SHORT = {'Intercept':'Intercept','TS_elev_m_z':'Elevation',
         'TS_slope_deg_z':'Slope','TS_SOC_0_30cm_clean_z':'SOC',
         'FS_TCC_t0_z':'TCC','FS_CBH_t0agg_z':'CBH',
         'HUM_viirs_near_t0_log_z':'VIIRS','HUM_imperv_near_t0_z':'Imperv',
         'HUM_roaddens_r5km_z':'Road dens','HUM_traildens_r10km_z':'Trail dens',
         'CLIM_pr_sum_pre_z':'Precip','CLIM_tmmn_mean_pre_z':'Min temp'}

bw_dict = dict(zip(bw['term'], bw['bandwidth']))

# ── Figure 1: bandwidth bar chart ──────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(10, 5))
names = [SHORT.get(v, v) for v in VARS]
bws   = [bw_dict.get(v, 0) for v in VARS]
colors = ['#d62728' if b <= 200 else '#ff7f0e' if b <= 1000
          else '#2ca02c' if b <= 5000 else '#1f77b4' for b in bws]
bars = ax.barh(names[::-1], bws[::-1], color=colors[::-1])
ax.axvline(12000, color='gray', ls='--', lw=1, label='n=12000 (global)')
ax.axvline(1200,  color='orange', ls=':', lw=1, label='10% of n')
ax.set_xlabel("Bandwidth (# neighbors)")
ax.set_title("MGWR stage5b — bandwidth per variable\n(red=very local <200, orange=local <1000, green=regional <5000, blue=quasi-global)")
ax.legend(fontsize=8)
plt.tight_layout()
fig1.savefig(f"{OUTDIR}/stage5b_bandwidths.png", dpi=150, bbox_inches='tight')
plt.close()
print("bandwidth chart saved")

# ── Figure 2: spatial coefficient maps ────────────────────────────────────
fig2, axes = plt.subplots(4, 3, figsize=(18, 22))
axes = axes.flatten()
x, y = coef['x'].values, coef['y'].values

for i, var in enumerate(VARS):
    ax = axes[i]
    vals = coef[var].values
    # clip to 2nd-98th percentile for color scale
    lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
    absmax = max(abs(lo), abs(hi))
    sc = ax.scatter(x, y, c=vals, cmap='RdBu_r',
                    vmin=-absmax, vmax=absmax, s=0.8, alpha=0.7)
    bw_val = bw_dict.get(var, '?')
    ax.set_title(f"{SHORT.get(var, var)}\n(bw={int(bw_val)})", fontsize=10)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)

fig2.suptitle("MGWR stage5b — spatial coefficient distributions\n(red=positive, blue=negative; clipped to 2nd–98th pct)", fontsize=13)
plt.tight_layout()
fig2.savefig(f"{OUTDIR}/stage5b_coef_maps.png", dpi=150, bbox_inches='tight')
plt.close()
print("coefficient maps saved")
