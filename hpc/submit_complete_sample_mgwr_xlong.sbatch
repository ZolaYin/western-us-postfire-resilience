#!/bin/bash
#SBATCH --job-name=mgwr_complete
#SBATCH --output=mgwr_complete_%j.out
#SBATCH --error=mgwr_complete_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=21-00:00:00
#SBATCH --partition=xlong

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/intersection_full.parquet}
PREDICTOR_FILE=${PREDICTOR_FILE:-predictors_stage5b.txt}
PILOT_BW_FILE=${PILOT_BW_FILE:-mgwr_bandwidths_12k.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKDIR/complete_sample_mgwr_$(date +%Y%m%d_%H%M%S)}
RESPONSE=${RESPONSE:-Resistance}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-/scratch/user/YOUR_NETID/conda_envs/gwr_env/bin/python}
N_JOBS=${N_JOBS:-12}

cd "$WORKDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"$BOOTSTRAP_PYTHON" run_complete_sample_mgwr_standard.py \
  --input "$INPUT_TABLE" \
  --response "$RESPONSE" \
  --predictors-file "$WORKDIR/$PREDICTOR_FILE" \
  --pilot-bandwidth-file "$WORKDIR/$PILOT_BW_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --reference-n 12000 \
  --init-mode p75 \
  --min-bw-factor 0.25 \
  --max-bw-factor 2.50 \
  --hard-min-bw 80 \
  --hard-max-frac 0.999 \
  --tol 1e-5 \
  --max-iter 80 \
  --tol-multi 5e-5 \
  --max-iter-multi 12 \
  --bws-same-times 3 \
  --n-jobs "$N_JOBS"
