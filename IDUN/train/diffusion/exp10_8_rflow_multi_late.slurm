#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp10_8_multi_late"
#SBATCH --output=IDUN/output/train/diffusion/exp10_8_multi_late_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp10_8_multi_late_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp10.8: Multi-modality Diffusion with Late Mode Conditioning ==="
echo "Mode embedding injected only at UNet level 2+ (64x64 and 32x32 resolutions)"
echo "Early layers (256x256, 128x128) forced to learn shared representations"
echo "Late layers can specialize for modality-specific features"
echo "4x training data: bravo, flair, t1_pre, t1_gd (all conditioned on seg)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp10_8: Multi-modality diffusion with LATE mode conditioning
# - mode_embedding_strategy=late injects mode only at later UNet levels
# - late_mode_start_level=2 means:
#   - Level 0 (256px): no mode info
#   - Level 1 (128px): no mode info
#   - Level 2 (64px): mode info injected
#   - Level 3 (32px): mode info injected
#   - Middle block: mode info injected
# - Forces early layers to learn shared features
# - Late layers can specialize for modality-specific details
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=multi \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.epochs=125 \
    mode.mode_embedding_strategy=late \
    mode.late_mode_start_level=2 \
    training.name=exp10_8_

conda deactivate
