#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1e_1_snr_gamma"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp1e_1_pixel_bravo_snr_gamma_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp1e_1_pixel_bravo_snr_gamma_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp1e 3D Pixel Diffusion: Bravo + Min-SNR-γ Loss Weighting ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ============================================================================
# 3D Pixel-Space Diffusion: Bravo + Min-SNR-γ (γ=5.0) Loss Weighting
# ============================================================================
# Same as exp1_1 but with RFlow-specific Min-SNR-γ loss weighting.
# Downweights high-SNR timesteps (near clean) where the model wastes
# capacity on trivially easy predictions.
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    strategy.snr_gamma=5.0 \
    mode=bravo \
    model=default_3d \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.name=exp1e_1_pixel_bravo_snr_gamma_

conda deactivate
