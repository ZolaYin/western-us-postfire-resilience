#!/bin/bash
#SBATCH --job-name=softmoe_lisa
#SBATCH --output=softmoe_lisa_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=medium

cd /scratch/user/YOUR_NETID

PYTHON=/scratch/user/YOUR_NETID/conda_envs/gwr_env/bin/python

OUTDIR="softmoe_lisa_round1_$(date +%Y-%m-%d)"
mkdir -p "$OUTDIR"
cp run_westernus_softmoe_lisa_round1.py "$OUTDIR/"

$PYTHON run_westernus_softmoe_lisa_round1.py \
    --input  /scratch/user/YOUR_NETID/westernus_current_candidate_table_plus_regions.parquet \
    --output-dir "$OUTDIR" \
    --n-jobs 32

echo "Done. Results in $OUTDIR"
