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
#SBATCH --job-name="exp9_dcae_f128_1_5"
#SBATCH --output=IDUN/output/train/compression/exp9_3_dcae_f128_1_5_%j.out
#SBATCH --error=IDUN/output/train/compression/exp9_3_dcae_f128_1_5_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp9.3 DC-AE 1.5 f128c512 (128x compression) Multi-Modality ==="
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

# Exp9.3 DC-AE 1.5: f128c512 with Structured Latent Space
# Input: 256x256x1 -> Latent: 2x2x512
# Structured latent: channel_steps=[16, 20, 24, ..., 512] (125 options)
# Per paper: DC-AE 1.5 recommended for large channel counts
time python -m medgen.scripts.train_compression --config-name=dcae \
    paths=cluster \
    mode=multi_modality \
    dcae=f128 \
    dcae.structured_latent.enabled=true \
    training.epochs=125 \
    training.batch_size=16 \
    training.learning_rate=1e-4 \
    training.use_compile=false \
    training.name=exp9_f128_1_5_

conda deactivate
