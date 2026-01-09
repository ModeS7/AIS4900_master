#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp10_4_mode_scale"
#SBATCH --output=IDUN/output/train/diffusion/exp10_4_mode_scale_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp10_4_mode_scale_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp10.4: Multi-modality + ScoreAug + Mode Intensity Scaling ==="
echo "Mode intensity scaling forces model to use mode conditioning"
echo "Scales: bravo=0.85, flair=1.15, t1_pre=0.92, t1_gd=1.08"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# exp10_4: ScoreAug single-transform mode + mode intensity scaling
# - Single transform sampled per batch (uniform probability)
# - Mode intensity scaling: input *= mode_scale, target unchanged
# - Model MUST learn mode to predict correct velocity
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=multi_modality \
    strategy.num_train_timesteps=100 \
    model.image_size=128 \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.epochs=125 \
    training.score_aug.enabled=true \
    training.score_aug.rotation=true \
    training.score_aug.flip=true \
    training.score_aug.translation=true \
    training.score_aug.cutout=true \
    training.score_aug.brightness=false \
    training.score_aug.use_omega_conditioning=true \
    training.score_aug.mode_intensity_scaling=true \
    mode.use_mode_embedding=true \
    training.name=exp10_4_

conda deactivate
