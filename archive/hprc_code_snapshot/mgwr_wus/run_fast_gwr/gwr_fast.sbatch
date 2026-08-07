#!/bin/bash
#SBATCH --job-name=gwr_res_fast
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=gwr_fast_%j.out
#SBATCH --error=gwr_fast_%j.err

cd /scratch/user/YOUR_NETID/mgwr_wus/run_fast_gwr

source /scratch/user/YOUR_NETID/mgwr_wus/mgwr_env/bin/activate

python gwr_resistance_fast.py
