#!/bin/bash
#SBATCH --job-name=rf_ms_round2
#SBATCH --output=rf_ms_round2_%j.out
#SBATCH --error=rf_ms_round2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/westernus_current_candidate_table_plus_regions.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKDIR/westernus_multiscale_rf_round2_2026-04-28}
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-/scratch/user/YOUR_NETID/conda_envs/gwr_env/bin/python}
VENV_DIR="$WORKDIR/.venv_multiscale_rf_round2"
N_TREES=${N_TREES:-300}
MAX_K=${MAX_K:-5000}
BLOCK_KM=${BLOCK_KM:-100}
VARIANTS=${VARIANTS:-}
SAMPLE_N=${SAMPLE_N:-}
SAVE_PREDICTIONS=${SAVE_PREDICTIONS:-0}

cd "$WORKDIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

if [ ! -d "$VENV_DIR" ]; then
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if ! python - <<'PY'
mods = ["pandas", "pyarrow", "numpy", "scipy", "sklearn"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    raise SystemExit(",".join(missing))
PY
then
  python -m pip install --upgrade pip
  python -m pip install pandas pyarrow numpy scipy scikit-learn
fi

CMD=(
  python run_westernus_multiscale_rf_round2.py
  --input "$INPUT_TABLE"
  --output-dir "$OUTPUT_DIR"
  --n-trees "$N_TREES"
  --block-km "$BLOCK_KM"
  --max-k "$MAX_K"
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
