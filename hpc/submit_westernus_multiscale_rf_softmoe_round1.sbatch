#!/bin/bash
#SBATCH --job-name=rf_softmoe1
#SBATCH --output=rf_softmoe1_%j.out
#SBATCH --error=rf_softmoe1_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00

set -euo pipefail

WORKDIR=${WORKDIR:-$PWD}
INPUT_TABLE=${INPUT_TABLE:-$WORKDIR/westernus_current_candidate_table_plus_regions.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKDIR/softmoe_rf_round1_2026-04-28}
PYTHON_BIN=${PYTHON_BIN:-/scratch/user/YOUR_NETID/conda_envs/gwr_env/bin/python}
GLOBAL_TREES=${GLOBAL_TREES:-300}
EXPERT_TREES=${EXPERT_TREES:-120}
EXPERT_MIN_LEAF=${EXPERT_MIN_LEAF:-10}
MAX_K=${MAX_K:-5000}
BLOCK_KM=${BLOCK_KM:-100}
SAMPLE_N=${SAMPLE_N:-}
GATE_FIT_SAMPLE_N=${GATE_FIT_SAMPLE_N:-60000}
GATE_SOFTNESS=${GATE_SOFTNESS:-1.0}
MIN_EFFECTIVE_N=${MIN_EFFECTIVE_N:-300}
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
  "$PYTHON_BIN" run_westernus_multiscale_rf_softmoe_round1.py
  --input "$INPUT_TABLE"
  --output-dir "$OUTPUT_DIR"
  --global-trees "$GLOBAL_TREES"
  --expert-trees "$EXPERT_TREES"
  --expert-min-leaf "$EXPERT_MIN_LEAF"
  --block-km "$BLOCK_KM"
  --max-k "$MAX_K"
  --gate-fit-sample-n "$GATE_FIT_SAMPLE_N"
  --gate-softness "$GATE_SOFTNESS"
  --min-effective-n "$MIN_EFFECTIVE_N"
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
