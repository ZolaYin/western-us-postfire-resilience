#!/bin/bash
#SBATCH --job-name=gwr_combo
#SBATCH --output=gwr_combo_%j.out
#SBATCH --error=gwr_combo_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
COMBO_NAME=${COMBO_NAME:?COMBO_NAME not set}
PREDICTOR_FILE=${PREDICTOR_FILE:?PREDICTOR_FILE not set}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/GWR_MGWR_ready_table_corrected_noevt15_sample12k.parquet}
SAMPLE_N=${SAMPLE_N:-12000}
VENV_DIR="$WORKDIR/.venv"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

source "$VENV_DIR/bin/activate"

python run_gwr_corrected_noevt15.py \
  --input "$INPUT_TABLE" \
  --predictors-file "$WORKDIR/$PREDICTOR_FILE" \
  --output-dir "$WORKDIR/gwr_outputs_${COMBO_NAME}" \
  --sample-n "$SAMPLE_N" \
  --bw-min 100
