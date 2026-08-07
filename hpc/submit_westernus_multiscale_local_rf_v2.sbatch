#!/bin/bash
#SBATCH --job-name=localrf_v2
#SBATCH --output=localrf_v2_%j.out
#SBATCH --error=localrf_v2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --partition=long

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/westernus_current_candidate_table_plus_regions.parquet}
VARIANT=${VARIANT:-E_EVT_fireregime_postclim}
K_SPATIAL=${K_SPATIAL:-2000 5000 10000}
TREES=${TREES:-100}
MIN_SAMPLES_LEAF=${MIN_SAMPLES_LEAF:-5}
MAX_FEATURES=${MAX_FEATURES:-sqrt}
BLOCK_KM=${BLOCK_KM:-100}
SAMPLE_N=${SAMPLE_N:-}
XY_MODE=${XY_MODE:-both}
LOCAL_SPLITS=${LOCAL_SPLITS:-block}
VENV_DIR="$WORKDIR/.venv-localrf-v2"

cd "$WORKDIR"

module purge
module load GCCcore/13.2.0 Python/3.11.5

source /scratch/user/YOUR_NETID/WUS_1km_gwrf_pilot_2026-04-24/.venv-gwrf/bin/activate

OUT_DIR="$WORKDIR/localrf_v2_${VARIANT}_$(date +%F_%H%M%S)"

CMD=(python run_westernus_multiscale_local_rf_v2.py \
  --input "$INPUT_TABLE" \
  --variant "$VARIANT" \
  --trees "$TREES" \
  --min-samples-leaf "$MIN_SAMPLES_LEAF" \
  --max-features "$MAX_FEATURES" \
  --block-km "$BLOCK_KM" \
  --xy-mode "$XY_MODE" \
  --local-splits "$LOCAL_SPLITS" \
  --output-dir "$OUT_DIR" \
  --k-spatial $K_SPATIAL)

if [ -n "$SAMPLE_N" ]; then
  CMD+=(--sample-n "$SAMPLE_N")
fi

"${CMD[@]}"
