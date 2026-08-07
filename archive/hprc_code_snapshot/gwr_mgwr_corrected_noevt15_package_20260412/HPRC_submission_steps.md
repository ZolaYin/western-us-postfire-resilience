# HPRC MGWR Submission Steps

This package is designed for staged MGWR runs on corrected Western US Resistance data.

Important caution:
- Do **not** start from the 15-variable stage.
- Use the stages in order.
- Stop and inspect outputs after each stage.
- The exact historical full 15-variable noEVT HPRC list is `uncertain`, so this is a corrected staged launch plan, not a claim of exact historical replication.

## Stage order

1. `stage1_topo_soil_4`
2. `stage2_topo_soil_forest_6`
3. `stage3_plus_human_core_9`
4. `stage4_plus_access_11`
5. `stage5_plus_climate_15`

## Local package path

`/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/gwr_mgwr_corrected_noevt15_package_20260412`

## Suggested HPRC workdir

`/scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_20260412`

## Step 1: copy the package to HPRC

```bash
scp -r "/path/to/google-drive/我的云端硬盘/US_Fire_and_Ecology_Data/WUS_1km/gwr_mgwr_corrected_noevt15_package_20260412" Faster:/scratch/user/YOUR_NETID/
```

## Step 2: log in and enter the workdir

```bash
ssh Faster
cd /scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_package_20260412
```

## Step 3: submit stage 1

```bash
sbatch --export=ALL,STAGE_NAME=stage1_topo_soil_4,PREDICTOR_FILE=predictors_stage1_topo_soil_4.txt submit_mgwr_stage.sbatch
```

## Step 4: inspect stage 1 outputs before moving on

Expected output directory:

`mgwr_outputs_stage1_topo_soil_4`

Key files:
- `mgwr_run_metadata.json`
- `mgwr_bandwidths.csv`
- `mgwr_coefficients.parquet`
- `mgwr_residuals.parquet`

## Step 5: submit later stages one by one

Stage 2:

```bash
sbatch --export=ALL,STAGE_NAME=stage2_topo_soil_forest_6,PREDICTOR_FILE=predictors_stage2_topo_soil_forest_6.txt submit_mgwr_stage.sbatch
```

Stage 3:

```bash
sbatch --export=ALL,STAGE_NAME=stage3_plus_human_core_9,PREDICTOR_FILE=predictors_stage3_plus_human_core_9.txt submit_mgwr_stage.sbatch
```

Stage 4:

```bash
sbatch --export=ALL,STAGE_NAME=stage4_plus_access_11,PREDICTOR_FILE=predictors_stage4_plus_access_11.txt submit_mgwr_stage.sbatch
```

Stage 5:

```bash
sbatch --export=ALL,STAGE_NAME=stage5_plus_climate_15,PREDICTOR_FILE=predictors_stage5_plus_climate_15.txt submit_mgwr_stage.sbatch
```

## Step 6: monitor jobs

```bash
squeue -u $USER
```

Check logs:

```bash
ls -lh mgwr_stage_*.out mgwr_stage_*.err
tail -n 50 mgwr_stage_<JOBID>.out
tail -n 50 mgwr_stage_<JOBID>.err
```

## What counts as success

- The job exits normally.
- `mgwr_outputs_<stage>/mgwr_bandwidths.csv` exists.
- `mgwr_outputs_<stage>/mgwr_coefficients.parquet` exists.
- There is no singular-matrix or bandwidth-search crash in the `.err` file.

## Stop rules

- If a stage crashes, do not submit the next stage yet.
- If stage 3 or later becomes unstable, fall back to the last successful smaller stage.
