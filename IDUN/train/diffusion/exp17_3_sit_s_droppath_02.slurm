#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp17_3_droppath"
#SBATCH --output=IDUN/output/train/diffusion/exp17_3_droppath_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp17_3_droppath_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp17_3: SiT-S with DropPath (Stochastic Depth) = 0.2
# =============================================================================
# DropPath randomly drops transformer blocks during training.
# Rate increases linearly from 0 (first block) to 0.2 (last block).
# This prevents overfitting by forcing the network to not rely on any single path.
#
# More aggressive regularization - recommended for very small datasets.
#
# Reference: https://arxiv.org/abs/1603.09382 (Deep Networks with Stochastic Depth)
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: SiT-S with DropPath rate = 0.2"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=sit \
    model.variant=S \
    model.patch_size=2 \
    model.drop_path_rate=0.2 \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.name=exp17_3_sit_s_droppath02_

conda deactivate
