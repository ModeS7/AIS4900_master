#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp9_5_scoreaug_compose"
#SBATCH --output=IDUN/output/train/diffusion/exp9_5_scoreaug_compose_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp9_5_scoreaug_compose_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp9_5: ScoreAug Compose Mode - all transforms applied independently
# Reference: https://arxiv.org/abs/2508.07926
#
# Compose mode: each transform has 50% independent probability
# Can get: none, one, multiple, or all transforms per batch
#
# Transforms (each 50% chance):
#   - spatial: 7 D4 symmetries (rotation + flip combos)
#   - translation: +-40% X, +-20% Y
#   - cutout: random rectangle 10-30%
#   - brightness: disabled (didn't perform well)
#
# Possible combinations: 2^3 = 8 (spatial, translate, cutout)
# No traditional augmentation - ScoreAug only
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
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.cutout=true \
    training.score_aug.brightness=false \
    training.score_aug.compose=true \
    training.score_aug.compose_prob=0.5 \
    training.score_aug.use_omega_conditioning=true \
    training.name=exp9_5_

conda deactivate
