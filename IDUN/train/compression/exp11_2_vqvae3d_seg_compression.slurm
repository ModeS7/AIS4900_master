#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp11_2_vqvae3d_seg"
#SBATCH --output=IDUN/output/train/compression/exp11_2_vqvae3d_seg_%j.out
#SBATCH --error=IDUN/output/train/compression/exp11_2_vqvae3d_seg_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp11.2 3D VQ-VAE Segmentation Mask Compression ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Exp11.2: 3D VQ-VAE Segmentation Mask Compression
# Based on exp8_1 (3D VQ-VAE, 4 latent channels)
# Input: 256x256x160 seg volumes -> discrete latent space
# Augmentation: 3D flips + 90° rotations (MONAI)
# Loss: BCE + Dice + Boundary (seg_mode)
# Metrics: Dice, IoU (not PSNR/LPIPS)
time python -m medgen.scripts.train_vqvae_3d \
    paths=cluster \
    mode=seg \
    training.epochs=125 \
    training.batch_size=1 \
    training.augment=true \
    vqvae_3d.embedding_dim=4 \
    vqvae_3d.seg_mode=true \
    training.name=exp11_2_vqvae3d_seg_

conda deactivate
