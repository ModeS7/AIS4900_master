#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp8_8b_adm"
#SBATCH --output=IDUN/output/train/diffusion/exp8_8b_adm_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp8_8b_adm_%j.err
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

# exp8_8b: ADM-inspired architecture - NO AUGMENTATION (like exp2)
# Re-run of exp8_8 to see true architecture potential
# "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol 2021)
# Reference: https://arxiv.org/abs/2105.05233
#
# Adapted from ImageNet 128x128 config:
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
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=exp8_8b_

conda deactivate
