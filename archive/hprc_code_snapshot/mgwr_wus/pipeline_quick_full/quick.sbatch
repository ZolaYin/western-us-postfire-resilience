#!/bin/bash
#SBATCH --job-name=mgwr_quick
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=quick_%j.out
#SBATCH --error=quick_%j.err

module purge
source ../mgwr_env/bin/activate

python mgwr_quick.py
