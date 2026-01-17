#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp27_seg_cond"
#SBATCH --output=IDUN/output/train/diffusion/exp27_seg_cond_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp27_seg_cond_%j.err
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

# Segmentation mask generation conditioned on tumor size distribution
# - Generates binary segmentation masks
# - Conditioning: 7-bin tumor size histogram (RANO-BM aligned)
# - CFG dropout: 10% (enables classifier-free guidance at inference)
# - Augmentation: H/V flip, ±15° rotation, ±5% translation, 0.9-1.1x scale
# - Metrics: MSE (training), Dice/IoU (validation)
# - No perceptual loss (binary masks don't work with VGG features)
time python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=seg_conditioned \
    model.image_size=256 \
    training.epochs=500 \
    training.batch_size=16 \
    training.use_ema=true \
    training.ema.decay=0.9999 \
    training.augment=true \
    training.use_min_snr=true \
    training.name=exp27_1_

conda deactivate
