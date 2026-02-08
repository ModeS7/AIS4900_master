#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp12_1_wavelet_bravo"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp12_1_wavelet_pixel_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp12_1_wavelet_pixel_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp12_1 Wavelet 3D Pixel Diffusion: Bravo 256x256 (Haar 2x2x2) ==="
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
# 3D Pixel-Space Diffusion with Haar Wavelet: Bravo 256x256x160
# ============================================================================
# Haar wavelet 2x2x2:
#   - Input:  [B, 1, 160, 256, 256] (full resolution)
#   - Haar:   [B, 8, 80, 128, 128]  (frequency subbands: LLL..HHH)
#   - UNet 5-level: in=16, out=8, L0=32ch@128x128x80 ... L4=512ch@8x8x5
#
# Same shape as exp11_1 (s2d) but with frequency separation instead of
# raw pixel rearrangement. Haar produces LLL=local average + 7 detail bands.
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d_5lvl \
    wavelet.enabled=true \
    volume.height=256 \
    volume.width=256 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.name=exp12_1_wavelet_bravo_

conda deactivate
