#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --job-name="nn_diag"
#SBATCH --output=IDUN/output/debug/%j.out
#SBATCH --error=IDUN/output/debug/%j.err

cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

python misc/diagnose_nnunet.py \
    --nnunet-base /cluster/work/modestas/MedicalDataSets/nnunet \
    --nnunet-results /cluster/work/modestas/AIS4900_master/runs/downstream/nnunet
