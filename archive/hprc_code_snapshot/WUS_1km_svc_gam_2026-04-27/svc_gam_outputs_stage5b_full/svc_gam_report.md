# SVC-GAM Report (2026-04-27)

## Method
Spatially varying coefficient GAM (mgcv `bam()`) with:
- `s(x, y, k=40)`: intercept surface = MGWR β₀(i) analog
- `s(x_j, k=5)`: global non-linear effect of each predictor
- `ti(x, y, by=x_j, k=c(8,8))`: β_j(i) surface analog; coefficient maps are based on the `ti()` term only

## CV Metrics
| Split | R² | RMSE |
|---|---|---|
| Block (test)  | -1.3579 | 0.3113 |
| Block (train) | 0.2664 | 0.1721 |
| Random (test) | -0.2073 | 0.2216 |
| Random (train)| 0.2563 | 0.1736 |

## EDF comparison (MGWR bandwidth vs GAM spatial EDF)
Higher EDF in ti() term = more spatial variation in that coefficient.

| Predictor | MGWR bw | GAM EDF (ti) | p-value |
|---|---|---|---|
| FS_CBH_t0agg_z | 122 | 45.4 | 0 |
| HUM_roaddens_r5km_z | 252 | 44.1 | 0 |
| CLIM_pr_sum_pre_z | 503 | 47.6 | 0 |
| TS_SOC_0_30cm_clean_z | 513 | 45.2 | 0 |
| TS_elev_m_z | 652 | 47.4 | 0 |
| FS_TCC_t0_z | 1018 | 41.9 | 0 |

## Key outputs
- `svc_gam_coef_maps.png`: spatial coefficient maps (β_j(i)) for each predictor
- `svc_gam_edf_table.csv`: effective degrees of freedom per smooth term
- `svc_gam_vs_mgwr.csv`: comparison table
- `svc_gam_summary.txt`: full model summary
