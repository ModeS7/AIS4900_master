#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp10_9_multi_film"
#SBATCH --output=IDUN/output/train/diffusion/exp10_9_multi_film_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp10_9_multi_film_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp10.9: Multi-modality Diffusion + FiLM Baseline ==="
echo "FiLM mode embedding: learns scale (gamma) and shift (beta) per modality"
echo "More expressive than additive embedding: output = gamma * features + beta"
echo "4x training data: bravo, flair, t1_pre, t1_gd (all conditioned on seg)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp10_9: Multi-modality diffusion with FiLM mode embedding
# - mode=multi uses mode embedding (model learns modality from mode_id)
# - mode_embedding_strategy=film: FiLM conditioning instead of additive
# - FiLM learns gamma (scale) and beta (shift) per modality
# - 4x training data from all modalities
# - No ScoreAug - baseline for comparison with exp10_4
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=multi \
    mode.mode_embedding_strategy=film \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.epochs=125 \
    training.name=exp10_9_

conda deactivate
