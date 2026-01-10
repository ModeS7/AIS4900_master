#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp10_1_dcae3d"
#SBATCH --output=IDUN/output/train/compression/exp10_1_dcae3d_default_%j.out
#SBATCH --error=IDUN/output/train/compression/exp10_1_dcae3d_default_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp10.1 3D DC-AE 256x256x160 (32x spatial, 4x depth -> 8x8x40x32 latent) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Exp10.1: 3D DC-AE with default settings
# - 6 stages = 32x spatial compression (256 -> 8)
# - depth_factors=[2,2,1,1,1] = 4x depth compression (160 -> 40)
# - 32 latent channels
# - Deterministic (no KL/VQ), L1 + 2.5D perceptual loss
# - Gradient checkpointing enabled by default
time python -m medgen.scripts.train_dcae_3d \
    paths=cluster \
    mode=multi_modality \
    training.name=exp10_1_

conda deactivate
