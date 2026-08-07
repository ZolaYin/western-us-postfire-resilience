#!/bin/bash
#SBATCH --job-name=gwrf_pilot
#SBATCH --output=gwrf_pilot_%j.out
#SBATCH --error=gwrf_pilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --partition=long

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
VARIANT=${VARIANT:?VARIANT not set}
NEIGHBOR_K=${NEIGHBOR_K:?NEIGHBOR_K not set}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/westernus_current_candidate_table_plus_regions.parquet}
RESPONSE=${RESPONSE:-Resistance}
TREES=${TREES:-300}
BLOCK_KM=${BLOCK_KM:-100}
SAMPLE_N=${SAMPLE_N:-}
VENV_DIR="$WORKDIR/.venv-gwrf"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

if [ ! -d "$VENV_DIR" ]; then
  python -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install pandas pyarrow numpy scikit-learn esda libpysal
else
  source "$VENV_DIR/bin/activate"
fi

OUTPUT_DIR="$WORKDIR/gwrf_related_pilot_${VARIANT}_k${NEIGHBOR_K}_$(date +%Y-%m-%d)"

CMD=(python run_westernus_gwrf_related_pilot.py \
  --input "$INPUT_TABLE" \
  --response "$RESPONSE" \
  --variant "$VARIANT" \
  --neighbor-k "$NEIGHBOR_K" \
  --trees "$TREES" \
  --block-km "$BLOCK_KM" \
  --output-dir "$OUTPUT_DIR")

if [ -n "$SAMPLE_N" ]; then
  CMD+=(--sample-n "$SAMPLE_N")
fi

"${CMD[@]}"
