#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp11_1_seg_compress"
#SBATCH --output=IDUN/output/train/compression/exp11_1_dcae_seg_compression_%j.out
#SBATCH --error=IDUN/output/train/compression/exp11_1_dcae_seg_compression_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-[01-07]

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp11.1 DC-AE Segmentation Mask Compression ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Exp11.1: DC-AE Segmentation Mask Compression
# Input: 256x256x1 seg masks -> Latent: 8x8x32
# Loss: BCE + Dice + Boundary (no perceptual, no GAN)
# Metrics: Dice, IoU (not PSNR/LPIPS/MS-SSIM)
# Regional: Per-tumor Dice by size category (RANO-BM thresholds)
# Augmentation: Aggressive spatial + Mosaic + CutMix + CopyPaste
time python -m medgen.scripts.train_dcae \
    paths=cluster \
    mode=seg \
    dcae=f32 \
    dcae.seg_mode=true \
    training.epochs=125 \
    training.batch_size=16 \
    training.learning_rate=1e-4 \
    training.use_compile=false \
    training.logging.regional_losses=true \
    training.name=exp11_1_seg_compress_

conda deactivate
