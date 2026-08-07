#!/bin/bash
#SBATCH --job-name=mgwr_bwmin1200
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=5-12:00:00
#SBATCH --output=mgwr_bwmin1200_%j.out
#SBATCH --error=mgwr_bwmin1200_%j.err

cd /scratch/user/YOUR_NETID/mgwr_wus
source /scratch/user/YOUR_NETID/mgwr_wus/mgwr_env/bin/activate

python mgwr_resistance_bwmin1200.py
