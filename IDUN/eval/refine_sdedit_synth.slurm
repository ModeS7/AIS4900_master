#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="sdedit"
#SBATCH --output=IDUN/output/eval/sdedit_%j.out
#SBATCH --error=IDUN/output/eval/sdedit_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# T1A: SDEdit refinement using exp1_1_1000 as the in-distribution refiner.
# Sweep t₀ ∈ {0.02, 0.05, 0.10, 0.20, 0.30}. No new training.
# Cost: ~5 t₀ × 25 vols × 24 steps ≈ 3000 forward passes per dir, ~1h.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== T1A: SDEdit refinement ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_ALLOC_CONF=expandable_segments:True

CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt"
DIR_525="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
DIR_OPT="${CLUSTER_BASE}/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp1_1_1000"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/test_new"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/sdedit_$(date +%Y%m%d-%H%M%S)"

for p in "${CKPT}" "${DIR_525}" "${DIR_OPT}" "${REAL_DIR}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: missing ${p}"
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"

time python -m medgen.scripts.refine_sdedit_synth \
    --checkpoint "${CKPT}" \
    --synth-dirs "${DIR_525}" "${DIR_OPT}" \
    --real-dir   "${REAL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --t0-values 0.02 0.05 0.10 0.20 0.30 \
    --num-steps 24 \
    --num-volumes 25 \
    --num-real 25 \
    --num-real-lpips 5

echo "=== Done ==="
ls -la "${OUT_DIR}"
