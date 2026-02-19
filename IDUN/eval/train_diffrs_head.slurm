#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="train_diffrs"
#SBATCH --output=IDUN/output/eval/train_diffrs_%j.out
#SBATCH --error=IDUN/output/eval/train_diffrs_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== DiffRS Head Training ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# === PATHS ===
BRAVO_CHECKPOINT="runs/diffusion_3d/bravo/exp1_1_pixel_bravo_rflow_256x160_20260119-212710/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="runs/diffrs"

# Train DiffRS head for the 3D bravo model
# Uses paired (bravo, seg) volumes for conditioned model
# train split: 105 volumes, test split: 51 volumes, 100 epochs
time python -m medgen.scripts.train_diffrs_discriminator --config-name=diffrs \
    diffusion_checkpoint="${BRAVO_CHECKPOINT}" \
    paths=cluster \
    data_mode=bravo \
    spatial_dims=3 \
    image_size=256 \
    depth=160 \
    generation_num_steps=25 \
    num_epochs=100 \
    batch_size=1 \
    val_split=test \
    output_dir="${OUTPUT_DIR}"

echo "=== Head training complete ==="
ls -la "${OUTPUT_DIR}/"

conda deactivate
