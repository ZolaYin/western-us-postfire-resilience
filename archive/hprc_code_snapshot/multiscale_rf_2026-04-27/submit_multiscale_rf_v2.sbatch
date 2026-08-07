#!/bin/bash
#SBATCH --job-name=ms_rf_v2
#SBATCH --output=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27/ms_rf_v2_%j.out
#SBATCH --error=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27/ms_rf_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=long

module purge
module load GCCcore/13.2.0 Python/3.11.5
source /scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_package_20260412/.venv/bin/activate

WORKDIR=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27
INPUT=/scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_package_20260412/GWR_MGWR_ready_table_corrected_noevt15.parquet

python3 ${WORKDIR}/run_rf_multiscale_v2.py \
  --input        ${INPUT} \
  --output-dir   ${WORKDIR}/results_v2_full133k \
  --response     Resistance \
  --n-trees      300 \
  --n-jobs       16 \
  --max-k        5000 \
  --col-map      TS_SOC_0_30cm_clean_z=TS_SOC_0_30cm_z

echo "Done."
