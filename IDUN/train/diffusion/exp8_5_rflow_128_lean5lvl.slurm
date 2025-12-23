#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp8_5_lean5lvl"
#SBATCH --output=IDUN/output/train/diffusion/exp8_5_lean5lvl_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp8_5_lean5lvl_%j.err
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

# exp8_5: Very deep narrow 5-level network [64,64,128,128,256]
# Hypothesis: Maximum depth with minimum channels
# Constant channels at early levels (Lean UNet finding)
# Tests if depth >> width helps small medical datasets
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    model.channels=[64,64,128,128,256] \
    model.attention_levels=[false,false,false,true,true] \
    model.num_head_channels=32 \
    training.augment=true \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=exp8_5_

conda deactivate
