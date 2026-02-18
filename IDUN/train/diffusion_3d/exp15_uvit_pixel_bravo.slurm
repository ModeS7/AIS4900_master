#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp15_uvit3d"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp15_uvit_pixel_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp15_uvit_pixel_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp15: U-ViT-S @ 160x128x128, patch=8 (5,120 tokens) ==="
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
# U-ViT-S: Bravo mode @ 160x128x128, patch_size=8
# ============================================================================
# Architecture: U-ViT-S (from paper)
#   - hidden_size: 512, depth: 13 (6 enc + 1 mid + 6 dec), heads: 8
#   - Token conditioning (timestep as token, no adaLN)
#   - Skip connections between encoder/decoder
#   - Same compute as DiT at patch=8, tests skip connection benefit
#
# Mode: bravo (in_channels=2, out_channels=1)
# Resolution: 128x128x160 (pixel space)
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=uvit_3d \
    model.variant=S \
    model.patch_size=8 \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.gradient_checkpointing=true \
    training.batch_size=1 \
    training.name=exp15_uvit_pixel_bravo_

conda deactivate
