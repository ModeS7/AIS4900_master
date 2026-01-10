#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp10_6_multi_dropout"
#SBATCH --output=IDUN/output/train/diffusion/exp10_6_multi_dropout_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp10_6_multi_dropout_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp10.6: Multi-modality Diffusion with Mode Embedding Dropout ==="
echo "Mode embedding with 20% dropout - forces partial shared learning"
echo "Like classifier-free guidance training: sometimes model doesn't know modality"
echo "4x training data: bravo, flair, t1_pre, t1_gd (all conditioned on seg)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp10_6: Multi-modality diffusion with mode embedding DROPOUT
# - 20% of batches drop mode embedding (use zeros)
# - Forces model to learn features that work without mode info
# - At inference, mode embedding is always used
# - Tests if CFG-style dropout helps regularization
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=multi \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.epochs=125 \
    mode.mode_embedding_strategy=dropout \
    mode.mode_embedding_dropout=0.2 \
    training.name=exp10_6_

conda deactivate
