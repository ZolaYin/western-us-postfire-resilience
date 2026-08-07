#!/bin/bash
#SBATCH --job-name=rf_resid2
#SBATCH --output=rf_resid2_%j.out
#SBATCH --error=rf_resid2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/westernus_current_candidate_table_plus_regions.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKDIR/residual_multiscale_rf_round2_2026-04-28}
PYTHON_BIN=${PYTHON_BIN:-/scratch/user/YOUR_NETID/conda_envs/gwr_env/bin/python}
BASE_TREES=${BASE_TREES:-300}
RESID_TREES=${RESID_TREES:-200}
RESID_MIN_LEAF=${RESID_MIN_LEAF:-100}
RESID_MAX_DEPTH=${RESID_MAX_DEPTH:-14}
RESID_MAX_FEATURES=${RESID_MAX_FEATURES:-0.5}
MAX_K=${MAX_K:-5000}
INNER_CV_FOLDS=${INNER_CV_FOLDS:-5}
BLOCK_KM_LIST=${BLOCK_KM_LIST:-"100 75"}
SAMPLE_N=${SAMPLE_N:-}
VARIANTS=${VARIANTS:-}
SAVE_PREDICTIONS=${SAVE_PREDICTIONS:-0}

cd "$WORKDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK:-16}

"$PYTHON_BIN" - <<'PY'
mods = ["pandas", "pyarrow", "numpy", "scipy", "sklearn"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit("Missing required modules in PYTHON_BIN: " + ",".join(missing))
PY

CMD=(
  "$PYTHON_BIN" run_westernus_multiscale_rf_residual_round2.py
  --input "$INPUT_TABLE"
  --output-dir "$OUTPUT_DIR"
  --base-trees "$BASE_TREES"
  --resid-trees "$RESID_TREES"
  --resid-min-leaf "$RESID_MIN_LEAF"
  --resid-max-depth "$RESID_MAX_DEPTH"
  --resid-max-features "$RESID_MAX_FEATURES"
  --max-k "$MAX_K"
  --inner-cv-folds "$INNER_CV_FOLDS"
  --block-km-list ${BLOCK_KM_LIST}
)

if [ -n "$VARIANTS" ]; then
  CMD+=(--variants ${VARIANTS})
fi

if [ -n "$SAMPLE_N" ]; then
  CMD+=(--sample-n "$SAMPLE_N")
fi

if [ "$SAVE_PREDICTIONS" = "1" ]; then
  CMD+=(--save-predictions)
fi

"${CMD[@]}"
