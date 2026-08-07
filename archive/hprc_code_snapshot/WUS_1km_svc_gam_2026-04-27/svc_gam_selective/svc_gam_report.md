# SVC-GAM Report (2026-04-27)

## Method
Spatially varying coefficient GAM (mgcv `bam()`) with:
- `s(x, y, k=40)`: intercept surface = MGWR β₀(i) analog
- `s(x_j, k=5)`: global non-linear effect of each predictor
- `ti(x, y, by=x_j, k=c(8,8))`: β_j(i) surface analog; coefficient maps are based on the `ti()` term only

## CV Metrics
| Split | R² | RMSE |
|---|---|---|
| Block (test)  | 0.0470 | 0.1979 |
| Block (train) | 0.2154 | 0.1780 |
| Random (test) | 0.2051 | 0.1798 |
| Random (train)| 0.2097 | 0.1789 |

## EDF comparison (MGWR bandwidth vs GAM spatial EDF)
Higher EDF in ti() term = more spatial variation in that coefficient.

| Predictor | MGWR bw | GAM EDF (ti) | p-value |
|---|---|---|---|
| FS_CBH_t0agg_z | 122 | 45.9 | 0 |
| HUM_roaddens_r5km_z | 252 | 46.5 | 0 |

## Key outputs
- `svc_gam_coef_maps.png`: spatial coefficient maps (β_j(i)) for each predictor
- `svc_gam_edf_table.csv`: effective degrees of freedom per smooth term
- `svc_gam_vs_mgwr.csv`: comparison table
- `svc_gam_summary.txt`: full model summary
