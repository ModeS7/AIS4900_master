#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp9_dcae_f64"
#SBATCH --output=IDUN/output/train/compression/exp9_2_dcae_f64_%j.out
#SBATCH --error=IDUN/output/train/compression/exp9_2_dcae_f64_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp9.2 DC-AE f64c128 (64x compression) Multi-Modality ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Exp9.2: DC-AE f64c128 - 64x spatial compression
# Input: 256x256x1 -> Latent: 4x4x128
# Model: 362M parameters, train from scratch
# Phase 1 training: L1 + Perceptual loss (no GAN)
time python -m medgen.scripts.train_compression --config-name=dcae \
    paths=cluster \
    mode=multi_modality \
    dcae=f64 \
    training.epochs=125 \
    training.batch_size=16 \
    training.learning_rate=1e-4 \
    training.use_compile=false \
    training.name=exp9_f64_

conda deactivate
