#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="morph_exp14_1"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp14_1_morph_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp14_1_morph_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# exp14_1: 3D Pixel Seg (unconditional seg generation, 256x256x160)
# Morphological metric: optimize for tumor distribution similarity
# (volume, Feret diameter, spatial location, count)

cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp14_1 (Morphological Metric) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

CHECKPOINT="runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp14_1_morph"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --mode seg \
    --ref-modality seg \
    --image-size 256 \
    --num-volumes 25 \
    --lo 15 --hi 60 \
    --metric morphological \
    --ref-split all \
    --seed 42

echo "=== Search complete ==="
cat "${OUTPUT_DIR}/search_results.json"

conda deactivate
