#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp5_vae_bf16"
#SBATCH --output=IDUN/output/train/vae/exp5_multimodal_bf16_%j.out
#SBATCH --error=IDUN/output/train/vae/exp5_multimodal_bf16_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp5 VAE Multi-Modality 256x256 Pure BF16 Training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Exp5: Multi-modality VAE 256x256 with Pure BF16 Training
# - Pools all modalities (bravo, flair, t1_pre, t1_gd) = 4x training data
# - Pure BF16: model weights stored in BF16 (~50% memory savings)
# - Prepares for future NVIDIA 2:4 structured sparsity
# - Single H100 GPU with torch.compile
# - 125 epochs, 20 validations
time python -m medgen.scripts.train_vae \
    paths=cluster \
    mode=multi_modality \
    model.image_size=256 \
    training.epochs=125 \
    training.batch_size=16 \
    training.augment=false \
    'training.precision.dtype=bf16' \
    'training.precision.pure_weights=true' \
    training.name=exp5_

conda deactivate
