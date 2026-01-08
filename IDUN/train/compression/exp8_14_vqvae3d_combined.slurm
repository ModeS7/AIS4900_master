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
#SBATCH --job-name="exp8_14_combo"
#SBATCH --output=IDUN/output/train/compression/exp8_14_vqvae3d_combined_%j.out
#SBATCH --error=IDUN/output/train/compression/exp8_14_vqvae3d_combined_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp8.14: 3D VQ-VAE COMBINED improvements ==="
echo "Combines: wider channels + larger codebook + higher perceptual"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Exp8.14: Combined best practices
# - Wider channels [128, 256]
# - Larger codebook (1024)
# - Higher perceptual weight (0.01)
time python -m medgen.scripts.train_vqvae_3d \
    paths=cluster \
    mode=multi_modality \
    training.epochs=125 \
    training.batch_size=1 \
    'vqvae_3d.channels=[128,256]' \
    'vqvae_3d.num_res_channels=[128,256]' \
    vqvae_3d.num_embeddings=1024 \
    vqvae_3d.perceptual_weight=0.01 \
    training.name=exp8_14_

conda deactivate
