#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="pca_steps23"
#SBATCH --output=IDUN/output/eval/eval_steps_pca_exp23_%j.out
#SBATCH --error=IDUN/output/eval/eval_steps_pca_exp23_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Evaluate how step count affects brain shape quality for exp23
# Uses PCA reconstruction error as the metric

cd ${SLURM_SUBMIT_DIR}

echo "=== Eval Steps vs PCA Error: exp23 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp23_pixel_bravo_scoreaug_256_20260309-220559/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
PCA_MODEL="${CLUSTER_BASE}/AIS4900_master/data/brain_pca_256x256x160.npz"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/eval_steps_pca_exp23"

time python -m medgen.scripts.eval_steps_pca \
    --checkpoint "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --pca-model "${PCA_MODEL}" \
    --output-dir "${OUT_DIR}" \
    --image-size 256 \
    --depth 160 \
    --num-volumes 25 \
    --lo 10 --hi 100 \
    --shift-ratio 1.0 \
    --seed 42

echo "--- Results ---"
cat "${OUT_DIR}/pca_step_results.json" 2>/dev/null

conda deactivate
