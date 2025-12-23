#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp8_8_adm"
#SBATCH --output=IDUN/output/train/diffusion/exp8_8_adm_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp8_8_adm_%j.err
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

# exp8_8: ADM-inspired architecture (Dhariwal & Nichol 2021)
# "Diffusion Models Beat GANs on Image Synthesis"
# Reference: https://arxiv.org/abs/2105.05233
#
# Adapted from ImageNet 128x128 config:
# - channel_mult = [1, 1, 2, 3, 4] with base 256 → scaled to base 128
# - channels = [128, 128, 256, 384, 512] (5 levels)
# - attention at 32, 16, 8 resolutions (levels 2, 3, 4)
# - num_res_blocks = 2 (ADM standard)
# - num_head_channels = 64
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    model.channels=[128,128,256,384,512] \
    model.attention_levels=[false,false,true,true,true] \
    model.num_res_blocks=2 \
    model.num_head_channels=64 \
    training.augment=true \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=exp8_8_

conda deactivate
