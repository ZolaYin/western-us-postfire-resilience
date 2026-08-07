#!/bin/bash
#SBATCH --job-name=rf_regime_cmp
#SBATCH --output=rf_regime_cmp_%j.out
#SBATCH --error=rf_regime_cmp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=medium

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/westernus_current_candidate_table_plus_regions.parquet}
REGIME_ASSIGNMENTS=${REGIME_ASSIGNMENTS:?REGIME_ASSIGNMENTS not set}
TREES=${TREES:-300}
OUTPUT_DIR="$WORKDIR/rf_regime_comparison_$(date +%Y-%m-%d)"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

source /scratch/user/YOUR_NETID/WUS_1km_gwrf_pilot_2026-04-24/.venv-gwrf/bin/activate

python run_rf_regime_comparison.py \
  --input "$INPUT_TABLE" \
  --regime-assignments "$REGIME_ASSIGNMENTS" \
  --trees "$TREES" \
  --output-dir "$OUTPUT_DIR"
