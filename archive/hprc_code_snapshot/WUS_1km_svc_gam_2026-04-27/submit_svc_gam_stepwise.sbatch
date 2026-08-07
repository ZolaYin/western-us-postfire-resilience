#!/bin/bash
#SBATCH --job-name=svc_stepwise
#SBATCH --output=/scratch/user/YOUR_NETID/WUS_1km_svc_gam_2026-04-27/svc_stepwise_%j.out
#SBATCH --error=/scratch/user/YOUR_NETID/WUS_1km_svc_gam_2026-04-27/svc_stepwise_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=long

module purge
module load GCC/12.3.0 GCCcore/13.2.0 R/4.3.2

WORKDIR=/scratch/user/YOUR_NETID/WUS_1km_svc_gam_2026-04-27
INPUT=/scratch/user/YOUR_NETID/mgwr_multi_y_2026-04-27/samples/sample_n12000_seed42.parquet
OUTDIR=/svc_stepwise_results

Rscript /run_westernus_svc_gam_stepwise.R   --input    --output-dir    --nthreads 8
