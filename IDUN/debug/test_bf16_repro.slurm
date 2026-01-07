#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="test_bf16_repro"
#SBATCH --output=IDUN/output/debug/test_bf16_repro_%j.out
#SBATCH --error=IDUN/output/debug/test_bf16_repro_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Testing BF16 (legacy) behavior with use_fp32_loss=false ==="
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

# Reinstall package to pick up changes
pip install -e . --quiet

# Run training with BF16 loss computation (should reproduce OLD/orange behavior)
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model.image_size=64 \
    training.epochs=200 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    +training.use_fp32_loss=false \
    training.name=debug_bf16_repro_

conda deactivate
