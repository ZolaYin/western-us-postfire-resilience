#!/bin/bash
#SBATCH --job-name=moran_comp
#SBATCH --partition=faster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=16G

module load intel-python/2022.1.0

cd /scratch/user/YOUR_NETID/mgwr_wus

python check_connectivity_moran_by_component.py
