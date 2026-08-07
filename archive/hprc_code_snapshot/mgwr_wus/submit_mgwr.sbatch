#!/bin/bash
#SBATCH --job-name=mgwr_wus
#SBATCH --output=mgwr_%j.out
#SBATCH --error=mgwr_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

set -euo pipefail

WORKDIR=/scratch/user/YOUR_NETID/mgwr_wus
VENV_DIR="$WORKDIR/.venv"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install pandas pyarrow numpy scipy libpysal spglm spreg mgwr

python run_mgwr.py \
  --input "$WORKDIR/MGWR_model_input.parquet" \
  --output-dir "$WORKDIR/mgwr_outputs" \
  --response-col Resistance \
  --coord-cols x,y \
  --id-cols pixel_id,row,col,t0_year \
  --bw-min 20 \
  --prefix mgwr
