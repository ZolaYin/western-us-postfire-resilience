#!/bin/bash
#SBATCH --job-name=multiscale_rf
#SBATCH --output=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27/multiscale_rf_%j.out
#SBATCH --error=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27/multiscale_rf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=long

module purge
module load GCCcore/13.2.0 Python/3.11.5
source /scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_package_20260412/.venv/bin/activate

WORKDIR=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27
INPUT=/scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_package_20260412/GWR_MGWR_ready_table_corrected_noevt15.parquet
FEATURES=${WORKDIR}/full133k_mgwrscale.parquet
OUTDIR=${WORKDIR}/results_full133k

echo "=== Step 1: Build mgwr-scale features (133k) ==="
python3 ${WORKDIR}/build_multiscale_features.py \
  --input     ${INPUT} \
  --output    ${FEATURES} \
  --scales-km 150 300 600 1200 \
  --max-k     5000 \
  --col-map   TS_SOC_0_30cm_clean_z=TS_SOC_0_30cm_z

echo "=== Step 2: Run RF with block CV ==="
python3 ${WORKDIR}/run_rf_multiscale.py \
  --input      ${FEATURES} \
  --output-dir ${OUTDIR} \
  --response   Resistance \
  --n-trees    300 \
  --n-jobs     16 \
  --col-map    TS_SOC_0_30cm_clean_z=TS_SOC_0_30cm_z

echo "Done."
