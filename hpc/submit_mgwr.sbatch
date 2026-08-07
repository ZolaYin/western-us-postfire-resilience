#!/bin/bash
#SBATCH --job-name=mgwr_wus
#SBATCH --output=mgwr_%j.out
#SBATCH --error=mgwr_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00

set -euo pipefail

WORKDIR=/scratch/user/YOUR_NETID/mgwr_wus
VENV_DIR="$WORKDIR/.venv"

cd "$WORKDIR"

module purge
module load Python/3.11.3-GCCcore-12.3.0

if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install mgwr pandas pyarrow

python run_mgwr.py \
  --input "$WORKDIR/MGWR_model_input.parquet" \
  --output-dir "$WORKDIR/mgwr_outputs"
