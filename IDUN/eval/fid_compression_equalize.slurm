#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="fid_eq"
#SBATCH --output=IDUN/output/eval/fid_eq_%j.out
#SBATCH --error=IDUN/output/eval/fid_eq_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Test: does VQ-VAE compression equalize synth & real distributions?
#   FID(real, synth)         — raw
#   FID(real, synth_rt)      — one-sided
#   FID(real_rt, synth_rt)   — equalized (hypothesis)
#   FID(real, real_rt)       — ceiling
# Also saves all 525 synth_rt + all real_rt volumes for downstream training.
# High mem because we hold ~550 full 256x256x160 volumes on CPU during FID.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== FID compression equalization test ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_ALLOC_CONF=expandable_segments:True

VQVAE_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt"
INPUT_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/test_new"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/fid_compression_eq_$(date +%Y%m%d-%H%M%S)"

for p in "${VQVAE_CKPT}" "${INPUT_DIR}" "${REAL_DIR}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: missing ${p}"
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"
echo "VQ-VAE:  ${VQVAE_CKPT}"
echo "Input:   ${INPUT_DIR}"
echo "Real:    ${REAL_DIR}"
echo "Output:  ${OUT_DIR}"

time python -m medgen.scripts.fid_compression_equalize \
    --compression-checkpoint "${VQVAE_CKPT}" \
    --compression-type vqvae \
    --input-dir "${INPUT_DIR}" \
    --real-dir "${REAL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --fid-network radimagenet

echo "=== Done ==="
echo "--- fid_results.json ---"
cat "${OUT_DIR}/fid_results.json"
