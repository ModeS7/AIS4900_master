#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp2_vae_ft"
#SBATCH --output=IDUN/output/train/compression/exp2_finetune_%j.out
#SBATCH --error=IDUN/output/train/compression/exp2_finetune_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp2: Fine-tune VAE from progressive pretrained weights at 256x256
# Loads weights from exp1 progressive VAE, trains with:
#   - GAN loss (discriminator enabled)
#   - Full LR schedule (warmup + cosine annealing)
#   - VAE augmentation + batch augmentation (mixup/cutmix)
#
# NOTE: Uses mode=bravo (1 channel) to match progressive VAE architecture.
# Progressive VAE trains on 1-channel images (bravo, flair, t1_pre, t1_gd separately).
# For dual mode (2 channels), need to train progressive VAE with dual mode first.

# Progressive VAE saves checkpoints in phase subdirectories:
#   phase_64/best.pt, phase_128/best.pt, phase_256/best.pt
#   final_model.pt (copy of phase_256/best.pt)

PRETRAINED_CHECKPOINT="/cluster/work/modestas/AIS4900_master/runs/vae_2d/progressive/exp1_20251221-234218/phase_256/best.pt"

# Verify checkpoint exists
if [ ! -f "$PRETRAINED_CHECKPOINT" ]; then
    echo "ERROR: Pretrained checkpoint not found: $PRETRAINED_CHECKPOINT"
    echo "Available exp1 phase_256 checkpoints:"
    ls -la /cluster/work/modestas/AIS4900_master/runs/vae_2d/progressive/exp1_*/phase_256/best.pt 2>/dev/null || echo "No exp1 phase_256 checkpoints found"
    echo "Or use final_model.pt:"
    ls -la /cluster/work/modestas/AIS4900_master/runs/vae_2d/progressive/exp1_*/final_model.pt 2>/dev/null || echo "No final_model.pt found"
    exit 1
fi

echo "Loading pretrained weights from: $PRETRAINED_CHECKPOINT"

time python -m medgen.scripts.train_vae \
    paths=cluster \
    mode=bravo \
    model.image_size=256 \
    pretrained_checkpoint="$PRETRAINED_CHECKPOINT" \
    training.epochs=100 \
    training.batch_size=16 \
    training.name=exp2_

conda deactivate
