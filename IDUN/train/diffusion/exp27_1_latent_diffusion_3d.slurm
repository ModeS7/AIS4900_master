#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp27_latent_diff_3d"
#SBATCH --output=IDUN/output/train/diffusion/exp27_latent_diff_3d_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp27_latent_diff_3d_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp27.1 3D Latent Diffusion Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# 3D Latent Diffusion with exp8_1 VQ-VAE 3D + ControlNet
# ============================================================================
# - Trains RFlow diffusion in compressed latent space
# - Uses 3D VQ-VAE from exp8_1 (4 latent channels, 4x spatial compression)
# - Latent shape: [B, 4, 40, 64, 64] from [B, 1, 160, 256, 256]
# - ControlNet for pixel-resolution conditioning (preserves small tumors)
# - Joint training: UNet + ControlNet trained together (freeze_unet=false)
# - Auto-encodes dataset if cache missing (~100 patients × 4 modalities = ~400 volumes)
# - batch_size=1 (3D volumes)
#
# Key metrics (all logged to TensorBoard):
# - Training: Loss/MSE_train (latent space)
# - Validation: PSNR, MS-SSIM (pixel space, after decoding)
# - Validation: LPIPS, KID, CMMD (2.5D slice-wise)
# - Visualization: Generated_3D_CenterSlice
# ============================================================================

COMPRESSION_CHECKPOINT="/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_best.pt"

# Check if checkpoint exists
if [ ! -f "$COMPRESSION_CHECKPOINT" ]; then
    echo "ERROR: Compression checkpoint not found at:"
    echo "  $COMPRESSION_CHECKPOINT"
    echo ""
    echo "Please update COMPRESSION_CHECKPOINT to point to your exp8_1 checkpoint."
    echo "Example: find /cluster/work/modestas/AIS4900_master/runs/compression_3d -name 'checkpoint_best.pt' | grep exp8_1"
    exit 1
fi

echo "Using compression checkpoint: $COMPRESSION_CHECKPOINT"

time python -m medgen.scripts.train_diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    latent=default \
    controlnet=default \
    training.batch_size=1 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.gradient_checkpointing=true \
    latent.enabled=true \
    latent.compression_checkpoint=${COMPRESSION_CHECKPOINT} \
    latent.compression_type=auto \
    latent.auto_encode=true \
    controlnet.enabled=true \
    controlnet.freeze_unet=false \
    controlnet.gradient_checkpointing=true \
    training.name=exp27_1_latent_controlnet_

conda deactivate
