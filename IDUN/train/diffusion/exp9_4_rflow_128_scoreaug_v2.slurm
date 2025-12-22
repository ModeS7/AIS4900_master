#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp9_4_scoreaug_v2"
#SBATCH --output=IDUN/output/train/diffusion/exp9_4_scoreaug_v2_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp9_4_scoreaug_v2_%j.err
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

# exp9_4: ScoreAug v2 - improved transforms (based on exp9_1)
# Reference: https://arxiv.org/abs/2508.07926
#
# Changes from exp9_1:
#   - flip: Added hflip + vflip + rotation combos (7 D4 symmetries total)
#   - translation: Increased to +-40% X, +-20% Y (asymmetric for brain shape)
#   - cutout: Now random rectangle (10-30% each dimension independently)
#   - brightness: Still disabled (didn't perform well in exp9_2)
#
# Single-transform mode: picks ONE from [identity, spatial, translate, cutout]
# Each category: 25% probability
# Spatial: uniform over 7 D4 options (~3.6% each)
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
    training.score_aug.use_omega_conditioning=true \
    training.name=exp9_4_

conda deactivate
