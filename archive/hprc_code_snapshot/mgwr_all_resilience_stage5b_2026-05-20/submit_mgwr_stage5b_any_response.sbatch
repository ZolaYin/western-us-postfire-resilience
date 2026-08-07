#!/bin/bash
#SBATCH --job-name=mgwr_s5b_any
#SBATCH --output=mgwr_s5b_any_%j.out
#SBATCH --error=mgwr_s5b_any_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/sample_n12000_seed42.parquet}
PREDICTOR_FILE=${PREDICTOR_FILE:-predictors_stage5b.txt}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKDIR/output_default}
RESPONSE=${RESPONSE:-Resistance}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-/scratch/user/YOUR_NETID/conda_envs/gwr_env/bin/python}

cd "$WORKDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"$BOOTSTRAP_PYTHON" run_mgwr_stage5b_any_response.py \
  --input "$INPUT_TABLE" \
  --response "$RESPONSE" \
  --predictors-file "$WORKDIR/$PREDICTOR_FILE" \
  --output-dir "$OUTPUT_DIR"
