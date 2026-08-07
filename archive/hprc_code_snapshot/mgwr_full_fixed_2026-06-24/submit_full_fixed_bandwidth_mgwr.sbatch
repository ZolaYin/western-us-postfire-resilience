#!/bin/bash
#SBATCH --job-name=mgwr_full_fixed
#SBATCH --output=mgwr_full_fixed_%j.out
#SBATCH --error=mgwr_full_fixed_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/intersection_full.parquet}
PREDICTOR_FILE=${PREDICTOR_FILE:-$WORKDIR/predictors_stage5b.txt}
BANDWIDTH_FILE=${BANDWIDTH_FILE:-$WORKDIR/mgwr_bandwidths_12k.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKDIR/full_fixed_mgwr_$(date +%Y%m%d_%H%M%S)}
RESPONSE=${RESPONSE:-Resistance}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-/scratch/user/YOUR_NETID/conda_envs/gwr_env/bin/python}
MAX_ITER=${MAX_ITER:-8}
CHUNK_SIZE=${CHUNK_SIZE:-512}
GLOBAL_THRESHOLD_FRAC=${GLOBAL_THRESHOLD_FRAC:-0.90}

cd "$WORKDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"$BOOTSTRAP_PYTHON" run_full_fixed_bandwidth_mgwr.py \
  --input "$INPUT_TABLE" \
  --response "$RESPONSE" \
  --predictors-file "$PREDICTOR_FILE" \
  --bandwidth-file "$BANDWIDTH_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --scale-bandwidths \
  --reference-n 12000 \
  --max-iter "$MAX_ITER" \
  --chunk-size "$CHUNK_SIZE" \
  --global-threshold-frac "$GLOBAL_THRESHOLD_FRAC" \
  --resume
