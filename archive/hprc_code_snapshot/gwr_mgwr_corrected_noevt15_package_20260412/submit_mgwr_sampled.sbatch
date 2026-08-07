#!/bin/bash
#SBATCH --job-name=mgwr_stage
#SBATCH --output=mgwr_stage_%j.out
#SBATCH --error=mgwr_stage_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
STAGE_NAME=${STAGE_NAME:?STAGE_NAME not set}
PREDICTOR_FILE=${PREDICTOR_FILE:?PREDICTOR_FILE not set}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/GWR_MGWR_ready_table_corrected_noevt15.parquet}
SAMPLE_N=${SAMPLE_N:-}
VENV_DIR="$WORKDIR/.venv"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

source "$VENV_DIR/bin/activate"

CMD=(python run_mgwr_corrected_noevt15.py \
  --input "$INPUT_TABLE" \
  --predictors-file "$WORKDIR/$PREDICTOR_FILE" \
  --output-dir "$WORKDIR/mgwr_outputs_${STAGE_NAME}")

if [ -n "$SAMPLE_N" ]; then
  CMD+=(--sample-n "$SAMPLE_N")
fi

"${CMD[@]}"
