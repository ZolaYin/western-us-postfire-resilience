#!/bin/bash
#SBATCH --job-name=model2_mgwr
#SBATCH --output=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27/model2_mgwr_%j.out
#SBATCH --error=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27/model2_mgwr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --partition=long

module purge
module load GCCcore/13.2.0 Python/3.11.5
source /scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_package_20260412/.venv/bin/activate

WORKDIR=/scratch/user/YOUR_NETID/multiscale_rf_2026-04-27

python3 ${WORKDIR}/run_rf_mgwrscale_model2_compare.py \
  --input       ${WORKDIR}/westernus_current_candidate_table_plus_regions.parquet \
  --output-dir  ${WORKDIR}/results_model2_compare \
  --response    Resistance \
  --n-trees     300 \
  --n-jobs      16 \
  --max-k       5000

echo "Done."
