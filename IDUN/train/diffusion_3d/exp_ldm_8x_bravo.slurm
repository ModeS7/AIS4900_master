#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="ldm_8x_bravo"
#SBATCH --output=IDUN/output/train/diffusion_3d/ldm_8x_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/ldm_8x_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== LDM_8x: 3D Latent Diffusion with VQ-VAE 8x Compression ==="
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
# LDM_8x: Latent Diffusion with VQ-VAE 8x Compression
# ============================================================================
# Compression: VQ-VAE 8x spatial (exp8_14)
#   - Input:  [B, 1, 160, 256, 256] (pixel space)
#   - Latent: [B, 4, 20, 32, 32]    (compressed, 8x smaller than 4x)
#   - Decoder VRAM: ~10 GB
#
# UNet: LDM_8x architecture (~14B params, ~60GB training target)
#   - channels: [512, 1024, 2048, 4096]
#   - attention_levels: [False, False, True, True]
#   - num_res_blocks: [3, 3, 3, 3]
#   - num_head_channels: 64
#
# Key advantage: 8x smaller latent = 8x less activation memory
#   -> Can fit MUCH larger UNet than LDM_4x
#
# Workflow:
#   1. Auto-encode dataset to latents (cached with hash validation)
#   2. Train diffusion on latents (latent_mse for train)
#   3. Decode to pixel space for validation metrics (PSNR, MS-SSIM, LPIPS)
#   4. Generation metrics in pixel space (KID, CMMD)
#
# Metrics logged:
#   Latent space (no decoding):
#   - Train: Loss/latent_mse_train
#   - Val:   Loss/latent_mse_val
#
#   Pixel space (decoded for val/gen only):
#   - Val:   PSNR, MS-SSIM, LPIPS
#   - Val:   regional/tumor_loss, regional/background_loss, regional/tumor_bg_ratio
#   - Val:   regional/{tiny,small,medium,large} (per tumor size)
#   - Gen:   KID, CMMD, lpips_diversity, msssim_diversity
#
#   Visualizations:
#   - Generated_3D_CenterSlice (every figure_interval)
#   - Denoising_Trajectory_3D (every figure_interval)
# ============================================================================

# VQ-VAE 8x compression checkpoint (exp8_14)
# Find the checkpoint dynamically
COMPRESSION_CHECKPOINT=$(find /cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality -path "*exp8_14*" -name 'checkpoint_best.pt' 2>/dev/null | head -1)

# Check if checkpoint exists
if [ -z "$COMPRESSION_CHECKPOINT" ] || [ ! -f "$COMPRESSION_CHECKPOINT" ]; then
    echo "ERROR: VQ-VAE 8x (exp8_14) checkpoint not found."
    echo ""
    echo "Available compression checkpoints:"
    find /cluster/work/modestas/AIS4900_master/runs/compression_3d -name 'checkpoint_best.pt' 2>/dev/null | head -10
    echo ""
    echo "To train VQ-VAE 8x first, run:"
    echo "  sbatch IDUN/train/compression/exp8_14_vqvae3d_combined.slurm"
    exit 1
fi

echo "Using compression checkpoint: $COMPRESSION_CHECKPOINT"
echo ""
echo "UNet Architecture: LDM_8x"
echo "  channels:         [512, 1024, 2048, 4096]"
echo "  attention_levels: [False, False, True, True]"
echo "  num_res_blocks:   [3, 3, 3, 3]"
echo "  num_head_channels: 64"
echo ""

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    training.epochs=1000 \
    training.learning_rate=1e-4 \
    training.batch_size=1 \
    training.gradient_checkpointing=true \
    latent.enabled=true \
    latent.compression_checkpoint=${COMPRESSION_CHECKPOINT} \
    latent.compression_type=vqvae \
    latent.auto_encode=true \
    latent.validate_cache=true \
    'model.channels=[512, 1024, 2048, 4096]' \
    'model.attention_levels=[false, false, true, true]' \
    'model.num_res_blocks=[3, 3, 3, 3]' \
    model.num_head_channels=64 \
    model.norm_num_groups=32 \
    training.logging.regional_losses=true \
    training.generation_metrics.enabled=true \
    training.generation_metrics.samples_per_epoch=1 \
    training.generation_metrics.samples_extended=4 \
    training.generation_metrics.samples_test=10 \
    training.name=ldm_8x_bravo_

conda deactivate
