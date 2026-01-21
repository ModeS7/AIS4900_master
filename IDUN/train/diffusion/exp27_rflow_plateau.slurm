#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp27_plateau"
#SBATCH --output=IDUN/output/train/diffusion/exp27_plateau_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp27_plateau_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp27: RFlow with ReduceLROnPlateau Scheduler ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# ============================================================================
# Based on: exp1_3_continuous_timesteps (baseline)
# Changed: scheduler from cosine to plateau (ReduceLROnPlateau)
#
# ReduceLROnPlateau:
#   - Reduces LR when validation loss stops improving
#   - factor: 0.5 (halve LR on plateau)
#   - patience: 10 (wait 10 epochs before reducing)
#   - min_lr: 1e-6
#
# This is useful for adaptive training where we want to let the model
# train longer at high LR if it's still improving, and reduce LR only
# when it plateaus.
# ============================================================================

time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model.image_size=128 \
    training.scheduler=plateau \
    training.plateau.factor=0.5 \
    training.plateau.patience=10 \
    training.plateau.min_lr=1e-6 \
    training.name=exp27_rflow_plateau_

conda deactivate
