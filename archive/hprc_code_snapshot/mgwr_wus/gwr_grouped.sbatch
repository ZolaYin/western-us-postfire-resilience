#!/bin/bash
#SBATCH --job-name=gwr_grouped
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=gwr_grouped_%j.out
#SBATCH --error=gwr_grouped_%j.err

# 进入工作目录
cd /scratch/user/YOUR_NETID/mgwr_wus

# 激活环境
source mgwr_env/bin/activate

echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "Running grouped GWR..."

python gwr_grouped.py

echo "Job finished at $(date)"
