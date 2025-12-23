#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp8_9_edm"
#SBATCH --output=IDUN/output/train/diffusion/exp8_9_edm_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp8_9_edm_%j.err
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

# exp8_9: EDM-inspired architecture (Karras et al. 2022)
# "Elucidating the Design Space of Diffusion-Based Generative Models"
# Reference: https://arxiv.org/abs/2206.00364
#
# Key EDM insights:
# - Uniform channels in later levels (plateau, not pyramid)
# - More res_blocks (4 instead of 2)
# - channel_mult = [1, 2, 2, 2] with base 128
# - channels = [128, 256, 256, 256] (4 levels, uniform tail)
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    model.channels=[128,256,256,256] \
    model.attention_levels=[false,false,true,true] \
    model.num_res_blocks=4 \
    model.num_head_channels=64 \
    training.augment=true \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=exp8_9_

conda deactivate
