#!/bin/bash
#SBATCH --job-name=mgwr_res_morecpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=mgwr_morecpu_%j.out
#SBATCH --error=mgwr_morecpu_%j.err

cd /scratch/user/YOUR_NETID/mgwr_wus/run_morecpu_mgwr

source /scratch/user/YOUR_NETID/mgwr_wus/mgwr_env/bin/activate

python mgwr_resistance_morecpu.py
