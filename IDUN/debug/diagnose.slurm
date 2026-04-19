#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name="nn_diag"
#SBATCH --output=IDUN/output/debug/nnunet_diagnose_%j.out
#SBATCH --error=IDUN/output/debug/nnunet_diagnose_%j.err

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

python misc/diagnose_nnunet.py \
    --nnunet-base ${CLUSTER_BASE}/MedicalDataSets/nnunet \
    --nnunet-results ${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet
