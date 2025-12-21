#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="exp1_vae_prog"
#SBATCH --output=IDUN/output/train/vae/exp1_progressive_%j.out
#SBATCH --error=IDUN/output/train/vae/exp1_progressive_%j.err
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

# Progressive VAE: 64 -> 128 -> 256, all modalities (bravo, flair, t1_pre, t1_gd, seg)
# Features: VAE augmentation (aggressive), batch augmentation (mixup/cutmix), no GAN
time python -m medgen.scripts.train_vae_progressive \
    paths=cluster \
    training.name=exp1_

conda deactivate
