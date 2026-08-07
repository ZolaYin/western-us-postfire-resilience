# GW-RF-Related Pilot - Resistance - E_EVT_fireregime_postclim (2026-04-24)

**Input:** `westernus_current_candidate_table_plus_regions.parquet`  
**Response:** `Resistance`  
**Variant:** `E_EVT_fireregime_postclim`  
**Variant label:** E: baseline + EVT proxy + fire-regime dummies  
**Rows used:** 121,739  
**Base predictors:** 34  
**Neighbor k:** 16  
**Block size:** 100 km  
**Trees per RF:** 200

This pilot uses a spatial-RF-style neighborhood summary design as a GW-RF-related first pass.

## 1. Global RF vs Spatial RF summary

| Split | Model | R2 | RMSE | Residual Moran's I |
|---|---|---|---|---|
| block | global_rf | 0.3519 | 0.1640 | 0.5296 |
| block | spatial_rf_knn16 | 0.2787 | 0.1730 | 0.5670 |
| random | global_rf | 0.6634 | 0.1171 | 0.0847 |
| random | spatial_rf_knn16 | 0.6757 | 0.1149 | 0.0742 |

## 2. Top importances - global RF (block split)

| Rank | Feature | Importance |
|---|---|---|
| 1 | HUM_viirs_near_t0_log_z | 0.137816 |
| 2 | xy_z | 0.061787 |
| 3 | FS_CBH_t0agg_z | 0.051752 |
| 4 | CLIM_tmmx_std_pre_z | 0.050849 |
| 5 | FS_EVT_resistance_proxy_z | 0.044858 |
| 6 | HUM_traildens_r10km_z | 0.043393 |
| 7 | TS_elev_m_z | 0.039182 |
| 8 | x_sq_z | 0.038590 |
| 9 | TS_roughness_z | 0.038385 |
| 10 | x | 0.037796 |
| 11 | HUM_popdens_win10km_log_z | 0.036671 |
| 12 | TS_SOC_0_30cm_z | 0.035880 |
| 13 | y_sq_z | 0.033850 |
| 14 | FS_TCC_t0_z | 0.033788 |
| 15 | CLIM_pr_sum_pre_z | 0.033644 |

## 3. Top importances - spatial RF (block split)

| Rank | Feature | Importance |
|---|---|---|
| 1 | HUM_viirs_near_t0_log_z | 0.117423 |
| 2 | FS_EVT_resistance_proxy_z | 0.044022 |
| 3 | xy_z | 0.036856 |
| 4 | FS_CBH_t0agg_z | 0.035522 |
| 5 | nn16_mean_HUM_viirs_near_t0_log_z | 0.035149 |
| 6 | nn16_mean_FS_TCC_t0_z | 0.033531 |
| 7 | x_sq_z | 0.026501 |
| 8 | x | 0.025562 |
| 9 | CLIM_tmmx_std_pre_z | 0.024248 |
| 10 | y | 0.022780 |
| 11 | TS_SOC_0_30cm_z | 0.022197 |
| 12 | y_sq_z | 0.021810 |
| 13 | TS_roughness_z | 0.021606 |
| 14 | nn16_mean_CLIM_tmmx_std_pre_z | 0.021419 |
| 15 | nn16_mean_FS_CBH_t0agg_z | 0.020035 |
