#!/bin/bash
#SBATCH --job-name=mgwr_my
#SBATCH --output=logs/mgwr_%x_%j.out
#SBATCH --error=logs/mgwr_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --partition=long

# Required env vars:
#   RESPONSE   — e.g. Resistance / STAB_good_pow2 / IRI_good_pow2
#   STAGE      — stage4b or stage5b
#   SEED       — 42, 123, or 456

set -euo pipefail

WORKDIR=/scratch/user/YOUR_NETID/mgwr_multi_y_2026-04-27
VENV=/scratch/user/YOUR_NETID/gwr_mgwr_corrected_noevt15_package_20260412/.venv

RESPONSE=${RESPONSE:?RESPONSE not set}
STAGE=${STAGE:?STAGE not set}
SEED=${SEED:?SEED not set}

INPUT="${WORKDIR}/samples/sample_n12000_seed${SEED}.parquet"
PRED_FILE="${WORKDIR}/predictors_${STAGE}.txt"
OUT_DIR="${WORKDIR}/results/${RESPONSE}_${STAGE}_seed${SEED}"

mkdir -p "${WORKDIR}/logs" "$OUT_DIR"
cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

source "$VENV/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

echo "RESPONSE=$RESPONSE  STAGE=$STAGE  SEED=$SEED"
echo "INPUT=$INPUT"
echo "OUTPUT=$OUT_DIR"

python run_mgwr_multi_y.py \
  --input       "$INPUT" \
  --predictors-file "$PRED_FILE" \
  --response    "$RESPONSE" \
  --random-state "$SEED" \
  --output-dir  "$OUT_DIR"
