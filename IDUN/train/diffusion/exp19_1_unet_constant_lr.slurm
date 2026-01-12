#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp19_1_constant_lr"
#SBATCH --output=IDUN/output/train/diffusion/exp19_1_constant_lr_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp19_1_constant_lr_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp19_1: UNet with Constant Learning Rate
# =============================================================================
# Constant LR maintains constant learning rate after warmup instead of
# cosine decay. This may help with overfitting on small datasets by:
# - Maintaining optimization pressure throughout training
# - Preventing premature convergence to local minima
#
# Standard cosine decay: LR -> eta_min (near 0) by end of training
# Constant LR: LR stays at base_lr after warmup
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: UNet with Constant LR (warmup 5 epochs, then constant)"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Constant LR configuration:
# - scheduler=constant: use constant LR after warmup
# - warmup_epochs=5: standard 5 epoch warmup
# - learning_rate=1e-4: standard base learning rate
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.scheduler=constant \
    training.name=exp19_1_unet_constant_lr_

conda deactivate
