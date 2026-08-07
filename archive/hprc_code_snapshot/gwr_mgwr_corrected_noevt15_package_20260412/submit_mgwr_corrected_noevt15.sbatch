#!/bin/bash
#SBATCH --job-name=mgwr_westernus_noevt15
#SBATCH --output=mgwr_noevt15_%j.out
#SBATCH --error=mgwr_noevt15_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
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
  --input "$WORKDIR/GWR_MGWR_ready_table_corrected_noevt15.parquet" \
  --predictors-file "$WORKDIR/predictors_noevt15_inferred_from_reports.txt" \
  --output-dir "$WORKDIR/mgwr_outputs_noevt15"
