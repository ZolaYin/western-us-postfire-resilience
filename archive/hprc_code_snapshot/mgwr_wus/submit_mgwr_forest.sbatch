#!/bin/bash
#SBATCH --job-name=mgwr_forest
#SBATCH --output=mgwr_forest_%j.out
#SBATCH --error=mgwr_forest_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

WORKDIR=/scratch/user/YOUR_NETID/mgwr_wus
VENV_DIR="$WORKDIR/.venv"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5
source "$VENV_DIR/bin/activate"

python run_mgwr.py \
  --input "$WORKDIR/MGWR_model_input.parquet" \
  --output-dir "$WORKDIR/mgwr_outputs_forest" \
  --response-col Resistance \
  --coord-cols x,y \
  --id-cols pixel_id,row,col,t0_year \
  --predictors-file "$WORKDIR/forest_predictors.txt" \
  --bw-min 20 \
  --prefix mgwr_forest
