#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp8_10_ddpm"
#SBATCH --output=IDUN/output/train/diffusion/exp8_10_ddpm_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp8_10_ddpm_%j.err
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

# exp8_10: Original DDPM architecture (Ho et al. 2020)
# "Denoising Diffusion Probabilistic Models"
# Reference: https://arxiv.org/abs/2006.11239
#
# Adapted from 256x256 config to 128x128:
# - Original: channel_mult = [1, 1, 2, 2, 4, 4] (6 levels for 256)
# - Adapted: channel_mult = [1, 1, 2, 2, 4] (5 levels for 128)
# - channels = [128, 128, 256, 256, 512]
# - attention at 16x16 (level 3)
# - num_res_blocks = 2
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    model.channels=[128,128,256,256,512] \
    model.attention_levels=[false,false,false,true,true] \
    model.num_res_blocks=2 \
    model.num_head_channels=64 \
    training.augment=true \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=exp8_10_

conda deactivate
