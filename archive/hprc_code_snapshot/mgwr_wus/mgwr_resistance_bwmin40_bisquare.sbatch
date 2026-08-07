#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

# === Environment ===
module purge

# 激活你的虚拟环境
source /scratch/user/YOUR_NETID/mgwr_wus/mgwr_env/bin/activate

# 进入工作目录
cd /scratch/user/YOUR_NETID/mgwr_wus

echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "Running MGWR..."

# 跑 MGWR
python mgwr_resistance_bwmin40_bisquare.py

echo "Job finished"
echo "Time: $(date)"

