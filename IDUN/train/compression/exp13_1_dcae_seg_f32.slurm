#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp13_1_seg_f32"
#SBATCH --output=IDUN/output/train/compression/exp13_1_dcae_seg_f32_%j.out
#SBATCH --error=IDUN/output/train/compression/exp13_1_dcae_seg_f32_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-[01-07]

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp13.1 DC-AE Seg Compression: f32 (8x8x32 latent) ==="
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

# Exp13.1: DC-AE Segmentation Mask Compression (8x8x32 latent)
# Input: 256x256x1 seg masks -> Latent: 8x8x32 (32x compression)
# Loss: BCE + Dice + Boundary (no perceptual, no GAN)
# Tokens for transformer: 8*8 = 64 per slice
# Augmentation: Aggressive spatial + Mosaic + CutMix + CopyPaste
time python -m medgen.scripts.train_compression --config-name=dcae \
    paths=cluster \
    mode=seg \
    dcae=f32 \
    dcae.seg_mode=true \
    training.epochs=500 \
    training.batch_size=16 \
    training.learning_rate=1e-4 \
    training.use_compile=false \
    training.augment=true \
    training.batch_augment.enabled=true \
    training.batch_augment.mosaic_prob=0.2 \
    training.batch_augment.cutmix_prob=0.2 \
    training.batch_augment.copy_paste_prob=0.3 \
    training.logging.regional_losses=true \
    training.name=exp13_1_seg_f32_

conda deactivate
