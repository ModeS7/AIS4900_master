#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp9_7_scoreaug_v2"
#SBATCH --output=IDUN/output/train/diffusion/exp9_7_scoreaug_v2_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp9_7_scoreaug_v2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp9.7 ScoreAug v2: Structured Non-destructive/Destructive ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp9_7: ScoreAug v2 mode with structured augmentation
# - Non-destructive (can stack): rotation, flip, translation (50% each)
# - Destructive (pick one): cutout OR fixed patterns (50/50 split)
# - 16 fixed patterns: checkerboard, grid dropout, coarse dropout, patch dropout
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.score_aug.enabled=true \
    training.score_aug.v2_mode=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.nondestructive_prob=0.5 \
    training.score_aug.destructive_prob=0.5 \
    training.score_aug.cutout_vs_pattern=0.5 \
    training.score_aug.use_omega_conditioning=true \
    training.name=exp9_7_

conda deactivate
