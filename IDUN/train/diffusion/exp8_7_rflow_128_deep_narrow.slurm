#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp8_7_deep_narrow"
#SBATCH --output=IDUN/output/train/diffusion/exp8_7_deep_narrow_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp8_7_deep_narrow_%j.err
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

# exp8_7: Deep narrow 4-level [48,96,192,256]
# Hypothesis: Even narrower than exp8_4, gradual growth
# Tests minimum viable width with 4 levels
# Starting at 48 (very narrow) to maximize regularization effect
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    model.channels=[48,96,192,256] \
    model.attention_levels=[false,false,true,true] \
    model.num_head_channels=32 \
    training.augment=true \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=exp8_7_

conda deactivate
