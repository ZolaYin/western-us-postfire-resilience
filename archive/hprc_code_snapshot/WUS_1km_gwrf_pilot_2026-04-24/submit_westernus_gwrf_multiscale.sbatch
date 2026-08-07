#!/bin/bash
#SBATCH --job-name=mgrf_pilot
#SBATCH --output=mgrf_pilot_%j.out
#SBATCH --error=mgrf_pilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --partition=long

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/westernus_current_candidate_table_plus_regions.parquet}
K_SPATIAL=${K_SPATIAL:-2000 5000 10000}
TREES=${TREES:-100}
OUTPUT_DIR="$WORKDIR/mgrf_pilot_$(date +%Y-%m-%d)"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

source /scratch/user/YOUR_NETID/WUS_1km_gwrf_pilot_2026-04-24/.venv-gwrf/bin/activate

python run_westernus_gwrf_multiscale.py \
  --input "$INPUT_TABLE" \
  --k-spatial $K_SPATIAL \
  --trees "$TREES" \
  --n-jobs 16 \
  --output-dir "$OUTPUT_DIR"
