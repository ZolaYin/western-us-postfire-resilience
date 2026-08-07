#!/bin/bash
#SBATCH --job-name=mgwr_nohuman
#SBATCH --partition=faster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=mgwr_nohuman_%j.out
#SBATCH --error=mgwr_nohuman_%j.err

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
python mgwr_resistance_nohuman.py

echo "Job finished"
echo "Time: $(date)"

