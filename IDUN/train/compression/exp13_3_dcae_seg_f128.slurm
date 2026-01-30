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
#SBATCH --job-name="exp13_3_seg_f128"
#SBATCH --output=IDUN/output/train/compression/exp13_3_dcae_seg_f128_%j.out
#SBATCH --error=IDUN/output/train/compression/exp13_3_dcae_seg_f128_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-[01-07]

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp13.3 DC-AE Seg Compression: f128 (2x2x512 latent) ==="
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

# Exp13.3: DC-AE Segmentation Mask Compression (2x2x512 latent)
# Input: 256x256x1 seg masks -> Latent: 2x2x512 (128x spatial, 32x total)
# Loss: BCE + Dice + Boundary (no perceptual, no GAN)
# Tokens for transformer: 2*2 = 4 (extremely efficient!)
# Near-perfect reconstruction observed in previous experiments
time python -m medgen.scripts.train_dcae \
    paths=cluster \
    mode=seg \
    dcae=f128 \
    dcae.seg_mode=true \
    training.epochs=500 \
    training.batch_size=16 \
    training.learning_rate=1e-4 \
    training.use_compile=false \
    training.logging.regional_losses=true \
    training.name=exp13_3_seg_f128_

conda deactivate
