#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="eval_maisi"
#SBATCH --output=IDUN/output/train/compression/exp12_1_eval_maisi_vae_%j.out
#SBATCH --error=IDUN/output/train/compression/exp12_1_eval_maisi_vae_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# =============================================================================
# exp12_1: Evaluate Pretrained MAISI VAE Compression
# =============================================================================
# MAISI VAE from NVIDIA (pretrained on 39K CT + 19K MRI volumes)
# Architecture:
#   - 4x spatial compression per dimension (not 8x like standard VAE)
#   - 4 latent channels
#   - Total compression: 16x
#   - Uses sliding window inference for memory efficiency
#
# This evaluates reconstruction quality (PSNR, SSIM, L1) on:
#   - Test set (full evaluation)
#   - Validation set (full evaluation)
#   - Train set (sampled, 100 samples)
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "exp12_1: Evaluate Pretrained MAISI VAE"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory
OUTPUT_DIR="runs/compression_3d/maisi_eval"
mkdir -p $OUTPUT_DIR

# Check if MAISI bundle exists, download if not
if [ ! -d "bundles/maisi_ct_generative" ]; then
    echo "Downloading MAISI bundle..."
    python -c "from monai.bundle import download; download(name='maisi_ct_generative', bundle_dir='bundles/')"
fi

echo ""
echo "=== Evaluating on TEST set ==="
time python scripts/evaluate_maisi_vae.py \
    --bundle_path bundles/maisi_ct_generative \
    --data_dir /cluster/home/modlrwtn/data/brainmetshare-3 \
    --split test \
    --output $OUTPUT_DIR/maisi_test_results.json

echo ""
echo "=== Evaluating on VAL set ==="
time python scripts/evaluate_maisi_vae.py \
    --bundle_path bundles/maisi_ct_generative \
    --data_dir /cluster/home/modlrwtn/data/brainmetshare-3 \
    --split val \
    --output $OUTPUT_DIR/maisi_val_results.json

echo ""
echo "=== Evaluating on TRAIN set (100 samples) ==="
time python scripts/evaluate_maisi_vae.py \
    --bundle_path bundles/maisi_ct_generative \
    --data_dir /cluster/home/modlrwtn/data/brainmetshare-3 \
    --split train \
    --max_samples 100 \
    --output $OUTPUT_DIR/maisi_train_results.json

echo ""
echo "=== All Results ==="
echo "Test:"
cat $OUTPUT_DIR/maisi_test_results.json
echo ""
echo "Val:"
cat $OUTPUT_DIR/maisi_val_results.json
echo ""
echo "Train:"
cat $OUTPUT_DIR/maisi_train_results.json

conda deactivate
