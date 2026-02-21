#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp1b_pixel_bravo"
#SBATCH --output=IDUN/output/train/diffusion_3d/exp1b_pixel_bravo_%j.out
#SBATCH --error=IDUN/output/train/diffusion_3d/exp1b_pixel_bravo_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp1b 3D Pixel Diffusion [-1,1]: Bravo 128x128x160 ==="
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
# exp1b: Same as exp1 but with [-1,1] rescaling
# 3D Pixel-Space Diffusion: Bravo conditioned on Seg Mask
# ============================================================================
# Architecture: 270M UNet (default_3d)
# Mode: bravo (in_channels=2, out_channels=1)
# Resolution: 128x128x160
# Change: training.rescale_data=true maps [0,1] -> [-1,1] in PixelSpace
# ============================================================================

time python -m medgen.scripts.train --config-name=diffusion_3d \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model=default_3d \
    volume.height=128 \
    volume.width=128 \
    volume.pad_depth_to=160 \
    training.epochs=500 \
    training.learning_rate=1e-4 \
    training.rescale_data=true \
    training.name=exp1b_pixel_bravo_

conda deactivate
