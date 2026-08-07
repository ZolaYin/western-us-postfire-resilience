#!/bin/bash
#SBATCH --job-name=mgwr_stage
#SBATCH --output=mgwr_stage_%j.out
#SBATCH --error=mgwr_stage_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
STAGE_NAME=${STAGE_NAME:?STAGE_NAME not set}
PREDICTOR_FILE=${PREDICTOR_FILE:?PREDICTOR_FILE not set}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/GWR_MGWR_ready_table_corrected_noevt15.parquet}
VENV_DIR="$WORKDIR/.venv"

cd "$WORKDIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0

if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install mgwr pandas pyarrow numpy

python run_mgwr_corrected_noevt15.py \
  --input "$INPUT_TABLE" \
  --predictors-file "$WORKDIR/$PREDICTOR_FILE" \
  --output-dir "$WORKDIR/mgwr_outputs_${STAGE_NAME}"
