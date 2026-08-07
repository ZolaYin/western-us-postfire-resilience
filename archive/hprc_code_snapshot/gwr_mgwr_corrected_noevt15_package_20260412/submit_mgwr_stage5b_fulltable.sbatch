#!/bin/bash
#SBATCH --job-name=mgwr_stage5b_full
#SBATCH --output=mgwr_stage5b_full_%j.out
#SBATCH --error=mgwr_stage5b_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --partition=long

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/GWR_MGWR_ready_table_corrected_noevt15.parquet}
PREDICTOR_FILE=${PREDICTOR_FILE:-predictors_stage5b_plus_pr_tmmn_11.txt}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKDIR/mgwr_outputs_stage5b_fulltable}

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

source "$WORKDIR/.venv/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1

python run_mgwr_corrected_noevt15.py \
  --input "$INPUT_TABLE" \
  --predictors-file "$WORKDIR/$PREDICTOR_FILE" \
  --output-dir "$OUTPUT_DIR"
