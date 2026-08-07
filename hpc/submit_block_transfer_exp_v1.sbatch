#!/bin/bash
#SBATCH --job-name=blk_transfer_v1
#SBATCH --output=blk_transfer_v1_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=cpu

cd /scratch/user/YOUR_NETID

module load Python/3.11.3-GCCcore-12.3.0 || true
source "$HOME/.bashrc" 2>/dev/null || true

OUTDIR="block_transfer_exp_v1_$(date +%Y-%m-%d)"
mkdir -p "$OUTDIR"
cp run_block_transfer_exp_v1.py "$OUTDIR/"

pip install xgboost --quiet 2>/dev/null || true

python run_block_transfer_exp_v1.py \
    --input  /scratch/user/YOUR_NETID/westernus_current_candidate_table_plus_regions.parquet \
    --output-dir "$OUTDIR" \
    --n-jobs 32

echo "Done. Results in $OUTDIR"
