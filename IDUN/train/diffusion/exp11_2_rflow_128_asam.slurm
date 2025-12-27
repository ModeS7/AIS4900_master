#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp11_2_asam"
#SBATCH --output=IDUN/output/train/diffusion/exp11_2_asam_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp11_2_asam_%j.err
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

# exp11_2: ASAM baseline (based on exp2)
# ASAM (Adaptive SAM) normalizes perturbation by weight magnitude
# Better for models with varying weight scales
# Reference: https://arxiv.org/abs/2102.11600
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.sam.enabled=true \
    training.sam.rho=0.05 \
    training.sam.adaptive=true \
    training.name=exp11_2_

conda deactivate
