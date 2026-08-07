#!/bin/bash
#SBATCH --job-name=latent_pca
#SBATCH --output=latent_pca_%j.out
#SBATCH --error=latent_pca_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=medium

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/westernus_current_candidate_table_plus_regions.parquet}
SAMPLE_N=${SAMPLE_N:-}
PCA_MODE=${PCA_MODE:-fixed}
N_PCS=${N_PCS:-8}
CLUSTERS=${CLUSTERS:-4 5 6 7 8}
VENV_DIR="${VENV_DIR:-/scratch/user/YOUR_NETID/WUS_1km_gwrf_pilot_2026-04-24/.venv-gwrf}"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

source "$VENV_DIR/bin/activate"

OUTPUT_DIR="$WORKDIR/latent_regime_pca_${PCA_MODE}_${N_PCS}pcs_$(date +%Y-%m-%d)"

CMD=(python run_westernus_latent_regime_pca.py \
  --input "$INPUT_TABLE" \
  --pca-mode "$PCA_MODE" \
  --n-pcs "$N_PCS" \
  --clusters $CLUSTERS \
  --output-dir "$OUTPUT_DIR")

if [ -n "$SAMPLE_N" ]; then
  CMD+=(--sample-n "$SAMPLE_N")
fi

"${CMD[@]}"
