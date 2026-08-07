#!/bin/bash
#SBATCH --job-name=svc_gam_s5b
#SBATCH --output=svc_gam_s5b_%j.out
#SBATCH --error=svc_gam_s5b_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=7-00:00:00

set -euo pipefail

WORKDIR="${WORKDIR:-/scratch/user/YOUR_NETID/WUS_1km_svc_gam_2026-04-27}"
INPUT_CSV="${INPUT_CSV:-$WORKDIR/westernus_stage5b_svc_input.csv}"
SCRIPT_PATH="${SCRIPT_PATH:-$WORKDIR/run_westernus_svc_gam.R}"
OUTDIR="${OUTDIR:-$WORKDIR/svc_gam_outputs_stage5b_full}"
RLIB="${RLIB:-$WORKDIR/r_libs}"

mkdir -p "$WORKDIR" "$OUTDIR" "$RLIB"
cd "$WORKDIR"

module load GCC/13.3.0 R/4.4.2

export R_LIBS_USER="$RLIB"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

Rscript -e 'lib <- Sys.getenv("R_LIBS_USER"); dir.create(lib, recursive=TRUE, showWarnings=FALSE); .libPaths(c(lib, .libPaths())); if (!requireNamespace("mgcv", quietly=TRUE)) stop("mgcv is not installed in R_LIBS_USER: ", lib)'

Rscript "$SCRIPT_PATH" \
  --input "$INPUT_CSV" \
  --output-dir "$OUTDIR" \
  --nthreads "$SLURM_CPUS_PER_TASK"
