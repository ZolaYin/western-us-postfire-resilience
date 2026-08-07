# GW-RF-Related Pilot - Resistance - C_EVTproxy_postclim (2026-04-24)

**Input:** `westernus_current_candidate_table_plus_regions.parquet`  
**Response:** `Resistance`  
**Variant:** `C_EVTproxy_postclim`  
**Variant label:** C: baseline + EVT proxy  
**Rows used:** 121,739  
**Base predictors:** 29  
**Neighbor k:** 16  
**Block size:** 100 km  
**Trees per RF:** 200

This pilot uses a spatial-RF-style neighborhood summary design as a GW-RF-related first pass.

## 1. Global RF vs Spatial RF summary

| Split | Model | R2 | RMSE | Residual Moran's I |
|---|---|---|---|---|
| block | global_rf | 0.3490 | 0.1643 | 0.5305 |
| block | spatial_rf_knn16 | 0.2803 | 0.1728 | 0.5686 |
| random | global_rf | 0.6589 | 0.1179 | 0.0846 |
| random | spatial_rf_knn16 | 0.6725 | 0.1155 | 0.0744 |

## 2. Top importances - global RF (block split)

| Rank | Feature | Importance |
|---|---|---|
| 1 | HUM_viirs_near_t0_log_z | 0.137841 |
| 2 | xy_z | 0.061788 |
| 3 | CLIM_tmmx_std_pre_z | 0.053685 |
| 4 | FS_CBH_t0agg_z | 0.051718 |
| 5 | FS_EVT_resistance_proxy_z | 0.045030 |
| 6 | HUM_traildens_r10km_z | 0.042901 |
| 7 | TS_elev_m_z | 0.040221 |
| 8 | x | 0.039671 |
| 9 | TS_roughness_z | 0.038819 |
| 10 | x_sq_z | 0.037869 |
| 11 | HUM_popdens_win10km_log_z | 0.037095 |
| 12 | TS_SOC_0_30cm_z | 0.036708 |
| 13 | CLIM_pr_sum_pre_z | 0.034422 |
| 14 | FS_TCC_t0_z | 0.034002 |
| 15 | y_sq_z | 0.033938 |

## 3. Top importances - spatial RF (block split)

| Rank | Feature | Importance |
|---|---|---|
| 1 | HUM_viirs_near_t0_log_z | 0.117627 |
| 2 | FS_EVT_resistance_proxy_z | 0.044034 |
| 3 | xy_z | 0.040192 |
| 4 | FS_CBH_t0agg_z | 0.036430 |
| 5 | nn16_mean_HUM_viirs_near_t0_log_z | 0.035669 |
| 6 | nn16_mean_FS_TCC_t0_z | 0.033138 |
| 7 | CLIM_tmmx_std_pre_z | 0.029482 |
| 8 | x | 0.027788 |
| 9 | x_sq_z | 0.026168 |
| 10 | TS_SOC_0_30cm_z | 0.023233 |
| 11 | y | 0.022849 |
| 12 | nn16_mean_CLIM_tmmx_std_pre_z | 0.022373 |
| 13 | y_sq_z | 0.021922 |
| 14 | TS_roughness_z | 0.021677 |
| 15 | nn16_mean_FS_CBH_t0agg_z | 0.020816 |
