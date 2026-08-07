# Corrected noEVT GWR/MGWR Package

- Source table: `/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/westernus_roadtrail_append_legacy_nlcd_20260411/westernus_current_candidate_table_plus_cbh_tcc_roadtrail.parquet`
- Output table: `/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/gwr_mgwr_corrected_noevt15_package_20260412/GWR_MGWR_ready_table_corrected_noevt15.parquet`
- Rows kept: `133357`
- Predictor count: `15`
- TS_SOC invalid `-9999` handled as missing: `15`
- Exact historical full 15-variable noEVT list: `uncertain`

Predictors:
- `TS_elev_m_z`
- `TS_slope_deg_z`
- `TS_twi_z`
- `TS_SOC_0_30cm_clean_z`
- `FS_TCC_t0_z`
- `FS_CBH_t0agg_z`
- `HUM_popdens_win10km_log_z`
- `HUM_roaddens_r5km_z`
- `HUM_traildens_r10km_z`
- `HUM_imperv_near_t0_z`
- `HUM_viirs_near_t0_log_z`
- `CLIM_pr_sum_pre_z`
- `CLIM_tmmn_mean_pre_z`
- `CLIM_hot_days_35C_pre_z`
- `CLIM_tmmx_std_pre_z`

Evidence basis:
- Verified grouped GWR package uses A_topo_soil = elev+slope+twi+SOC.
- Verified formal report states a noEVT stable extension exists and all-in noEVT reaches R2=0.7023.
- Verified intermediate success row mentions adding TCC, CBH, popdens, tmmx_std, tmmn, and viirs.
- This package completes the noEVT launch set with road, trail, impervious, and pre-fire pr_sum + hot_days.

Local run entrypoints in this package:
- `run_gwr_corrected_noevt15.py`
- `run_mgwr_corrected_noevt15.py`
- `submit_mgwr_corrected_noevt15.sbatch`
