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
#SBATCH --job-name="exp9_4_dcae_f128_lpips"
#SBATCH --output=IDUN/output/train/compression/exp9_4_dcae_f128_lpips_%j.out
#SBATCH --error=IDUN/output/train/compression/exp9_4_dcae_f128_lpips_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-[01-07]

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp9.4 DC-AE f128c512 with LPIPS Loss ==="
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

# Exp9.4: DC-AE f128c512 with LPIPS loss (DC-AE paper setting)
# Compare with exp9.3 which uses RadImageNet perceptual loss
# Input: 256x256x1 -> Latent: 2x2x512
# Phase 1 training: L1 + LPIPS loss (no GAN)
time python -m medgen.scripts.train_compression --config-name=dcae \
    paths=cluster \
    mode=multi_modality \
    dcae=f128 \
    dcae.perceptual_loss_type=lpips \
    training.epochs=125 \
    training.batch_size=16 \
    training.learning_rate=1e-4 \
    training.use_compile=false \
    training.name=exp9_4_f128_lpips_

conda deactivate
