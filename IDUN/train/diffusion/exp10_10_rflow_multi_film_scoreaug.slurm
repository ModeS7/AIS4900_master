#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp10_10_multi_film_scoreaug"
#SBATCH --output=IDUN/output/train/diffusion/exp10_10_multi_film_scoreaug_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp10_10_multi_film_scoreaug_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp10.10: Multi-modality Diffusion + FiLM + ScoreAug Compose ==="
echo "FiLM mode embedding + ScoreAug compose mode"
echo "FiLM: learns scale (gamma) and shift (beta) per modality"
echo "4x training data: bravo, flair, t1_pre, t1_gd (all conditioned on seg)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp10_10: Multi-modality diffusion + FiLM + ScoreAug compose
# - mode=multi uses mode embedding (model learns modality from mode_id)
# - mode_embedding_strategy=film: FiLM conditioning instead of additive
# - FiLM learns gamma (scale) and beta (shift) per modality
# - 4x training data from all modalities
# - ScoreAug compose mode: each transform has 50% independent probability
#
# Transforms (each 50% chance):
#   - spatial: 7 D4 symmetries (rotation + flip combos)
#   - translation: +-40% X, +-20% Y
#   - cutout: random rectangle 10-30%
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=multi \
    mode.mode_embedding_strategy=film \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.epochs=125 \
    training.score_aug.enabled=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.cutout=true \
    training.score_aug.brightness=false \
    training.score_aug.compose=true \
    training.score_aug.compose_prob=0.5 \
    training.score_aug.use_omega_conditioning=true \
    training.name=exp10_10_

conda deactivate
