#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="refine_42"
#SBATCH --output=IDUN/output/eval/refine_exp42_%j.out
#SBATCH --error=IDUN/output/eval/refine_exp42_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Apply trained exp42 refinement generator to existing exp1_1_1000 outputs
# (compare_imagenet_optima_20260421-165759). One forward pass per volume —
# fast. Output saved for spectrum/Frangi/visual comparison against exp41a/b/c.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Apply exp42 refinement to exp1_1_1000 outputs ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
INPUT_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp1_1_1000"

# Auto-detect most recent exp42 run (chain-resumable runs all share the base name)
CKPT_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/refinement/exp42_pix2pix_refinement_* 2>/dev/null | head -1)
# Prefer best checkpoint if available, else latest
CKPT="${CKPT_DIR}/checkpoint_best.pt"
if [ ! -f "${CKPT}" ]; then
    CKPT="${CKPT_DIR}/checkpoint_latest.pt"
fi
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/refinement_eval_exp42"

if [ ! -f "${CKPT}" ]; then
    echo "ERROR: No exp42 checkpoint found (looked in ${CKPT_DIR})"
    exit 1
fi
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: No input volumes at ${INPUT_DIR}"
    exit 1
fi

echo "Checkpoint: ${CKPT}"
echo "Input:      ${INPUT_DIR}"
echo "Output:     ${OUT_DIR}"

mkdir -p "${OUT_DIR}"

time python -m medgen.scripts.restore_volumes \
    --restoration-model "${CKPT}" \
    --strategy refinement \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUT_DIR}" \
    --depth 160 \
    --image-size 256 \
    --eval \
    --data-root "${DATA_ROOT}"

count=$(find "${OUT_DIR}" -name "*.nii.gz" 2>/dev/null | wc -l)
echo "=== Done: ${count} refined volumes at ${OUT_DIR} ==="
