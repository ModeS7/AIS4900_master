#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp16_hdit3d"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp16_hdit_pixel_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp16_hdit_pixel_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp16: HDiT-S @ 160x128x128, patch=4, levels=[2,4,6,4,2] ==="
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
# HDiT-S: Bravo mode @ 160x128x128, patch_size=4
# ============================================================================
# Architecture: HDiT-S (DiT-S hidden=384, heads=6)
#   - level_depths: [2, 4, 6, 4, 2]
#   - Level 0: 40,960 tokens (2 enc + 2 dec blocks)
#   - Level 1:  5,120 tokens (4 enc + 4 dec blocks)
#   - Bottleneck: 640 tokens (6 blocks)
#   - adaLN-Zero conditioning (same as DiT)
#   - Token merge/split for multi-resolution processing
#   - Enables patch=4 (finer spatial resolution than patch=8)
#
# Mode: bravo (in_channels=2, out_channels=1)
# Resolution: 128x128x160 (pixel space)
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=hdit_3d \
    model.variant=S \
    model.patch_size=4 \
    "model.level_depths=[2,4,6,4,2]" \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.gradient_checkpointing=true \
    training.batch_size=1 \
    training.name=exp16_hdit_pixel_bravo_

conda deactivate
