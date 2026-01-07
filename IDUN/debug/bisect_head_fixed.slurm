#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="bisect_HEAD_fixed"
#SBATCH --output=IDUN/output/debug/bisect_HEAD_fixed_%j.out
#SBATCH --error=IDUN/output/debug/bisect_HEAD_fixed_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CURRENT HEAD - should have .float() restored (fixed)
cd ${SLURM_SUBMIT_DIR}

echo "=== Test: HEAD (should be fixed) ==="
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

# Run training from current HEAD
echo "Starting training at HEAD..."
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model.image_size=64 \
    training.epochs=100 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=bisect_HEAD_fixed_

conda deactivate
