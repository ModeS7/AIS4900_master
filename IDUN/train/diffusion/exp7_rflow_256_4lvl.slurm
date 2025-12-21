#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="exp7_rflow_256_4lvl"
#SBATCH --output=IDUN/output/train/diffusion/exp7_rflow_256_4lvl_%j.out
#SBATCH --error=IDUN/output/train/diffusion/exp7_rflow_256_4lvl_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

cd ${SLURM_SUBMIT_DIR}

echo "=== Exp7 RFlow Bravo 256 (4-level extended network) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Exp7: 256px with extended 4-level network [128, 256, 256, 512]
# DDP: 4 GPUs, no augmentation, no EMA, no Min-SNR
time srun python -m medgen.scripts.train \
    paths=cluster \
    strategy=rflow \
    mode=bravo \
    model.image_size=256 \
    model.channels=[128,256,256,512] \
    model.attention_levels=[false,false,true,true] \
    model.num_res_blocks=1 \
    training.use_multi_gpu=true \
    training.augment=false \
    training.use_ema=false \
    training.use_min_snr=false \
    training.batch_size=16 \
    training.name=exp7_

conda deactivate
