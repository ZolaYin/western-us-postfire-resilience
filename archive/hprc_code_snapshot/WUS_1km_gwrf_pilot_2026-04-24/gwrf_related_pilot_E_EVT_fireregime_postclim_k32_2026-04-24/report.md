# GW-RF-Related Pilot - Resistance - E_EVT_fireregime_postclim (2026-04-24)

**Input:** `westernus_current_candidate_table_plus_regions.parquet`  
**Response:** `Resistance`  
**Variant:** `E_EVT_fireregime_postclim`  
**Variant label:** E: baseline + EVT proxy + fire-regime dummies  
**Rows used:** 121,739  
**Base predictors:** 34  
**Neighbor k:** 32  
**Block size:** 100 km  
**Trees per RF:** 200

This pilot uses a spatial-RF-style neighborhood summary design as a GW-RF-related first pass.

## 1. Global RF vs Spatial RF summary

| Split | Model | R2 | RMSE | Residual Moran's I |
|---|---|---|---|---|
| block | global_rf | 0.3519 | 0.1640 | 0.5296 |
| block | spatial_rf_knn32 | 0.2959 | 0.1709 | 0.5640 |
| random | global_rf | 0.6634 | 0.1171 | 0.0847 |
| random | spatial_rf_knn32 | 0.6833 | 0.1135 | 0.0675 |

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
| 1 | HUM_viirs_near_t0_log_z | 0.118622 |
| 2 | FS_EVT_resistance_proxy_z | 0.044020 |
| 3 | xy_z | 0.036844 |
| 4 | FS_CBH_t0agg_z | 0.036758 |
| 5 | nn32_mean_HUM_viirs_near_t0_log_z | 0.035932 |
| 6 | nn32_mean_FS_TCC_t0_z | 0.029570 |
| 7 | x | 0.025404 |
| 8 | CLIM_tmmx_std_pre_z | 0.025182 |
| 9 | x_sq_z | 0.024947 |
| 10 | TS_roughness_z | 0.023484 |
| 11 | TS_SOC_0_30cm_z | 0.022641 |
| 12 | y | 0.021148 |
| 13 | nn32_mean_CLIM_tmmx_std_pre_z | 0.020399 |
| 14 | TS_elev_m_z | 0.020390 |
| 15 | nn32_mean_FS_CBH_t0agg_z | 0.019778 |
