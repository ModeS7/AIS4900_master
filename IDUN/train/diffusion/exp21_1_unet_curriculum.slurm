#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp21_1_curriculum"
#SBATCH --output=IDUN/output/train/diffusion/exp21_1_curriculum_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp21_1_curriculum_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp21_1: UNet with Curriculum Timestep Scheduling
# =============================================================================
# Curriculum learning progressively shifts from easy (low noise) to hard
# (high noise) timesteps. Benefits:
# - Model learns coarse structure first, then fine details
# - Gradual introduction of harder examples
# - May help with stability and convergence
#
# Configuration:
# - warmup_epochs=50: reach full range by epoch 50
# - Initial range: [0.0, 0.3] (only low-noise timesteps)
# - Final range: [0.0, 1.0] (full timestep range)
#
# Linear interpolation between ranges over warmup period.
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "Experiment: UNet with Curriculum Timesteps (warmup=50, [0-0.3] -> [0-1.0])"
echo "============================================================"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Curriculum timestep configuration:
# - enabled=true: activate curriculum learning
# - warmup_epochs=50: epochs to reach full timestep range
# - min_t_start/end=0.0: minimum timestep stays at 0
# - max_t_start=0.3: start with only low-noise timesteps (0-30%)
# - max_t_end=1.0: end with full range (0-100%)
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.curriculum.enabled=true \
    training.curriculum.warmup_epochs=50 \
    training.curriculum.min_t_start=0.0 \
    training.curriculum.max_t_start=0.3 \
    training.curriculum.min_t_end=0.0 \
    training.curriculum.max_t_end=1.0 \
    training.name=exp21_1_unet_curriculum_

conda deactivate
