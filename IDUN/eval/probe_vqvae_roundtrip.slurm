#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="vqvae_probe"
#SBATCH --output=IDUN/output/eval/vqvae_probe_%j.out
#SBATCH --error=IDUN/output/eval/vqvae_probe_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Cheap VQ-VAE roundtrip probe (≤3 synthetic volumes through the 3D VQ-VAE).
# Goal: see whether vector-quantization manifold projection moves the mid/high
# spectrum band toward real. If no movement → kill the hypothesis.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== VQ-VAE roundtrip probe ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

# 4x VQ-VAE — most reliable reference in the project (latent 4ch, sf=4)
VQVAE_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt"
INPUT_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp1_1_1000"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/test_new"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/vqvae_roundtrip_probe_$(date +%Y%m%d-%H%M%S)"

for p in "${VQVAE_CKPT}" "${INPUT_DIR}" "${REAL_DIR}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: missing ${p}"
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"
echo "VQ-VAE:    ${VQVAE_CKPT}"
echo "Synthetic: ${INPUT_DIR}"
echo "Real:      ${REAL_DIR}"
echo "Output:    ${OUT_DIR}"

time python -m medgen.scripts.probe_vqvae_roundtrip \
    --compression-checkpoint "${VQVAE_CKPT}" \
    --compression-type vqvae \
    --input-dir "${INPUT_DIR}" \
    --real-dir "${REAL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --num-volumes 3 \
    --num-real 10

echo "=== Done ==="
ls -la "${OUT_DIR}"
