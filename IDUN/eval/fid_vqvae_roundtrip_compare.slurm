#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="fid_rt"
#SBATCH --output=IDUN/output/eval/fid_vqvae_rt_%j.out
#SBATCH --error=IDUN/output/eval/fid_vqvae_rt_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# FID compare: original synth vs VQ-VAE-roundtripped synth, against real.
# Roundtrips first 20 of each dir, computes FID over baseline 100 + subset 20.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== FID: VQ-VAE roundtrip vs original ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

VQVAE_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt"
DIR_A="${CLUSTER_BASE}/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759/exp1_1_1000"
DIR_B="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/test_new"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/fid_vqvae_rt_$(date +%Y%m%d-%H%M%S)"

for p in "${VQVAE_CKPT}" "${DIR_A}" "${DIR_B}" "${REAL_DIR}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: missing ${p}"
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"
echo "VQ-VAE:  ${VQVAE_CKPT}"
echo "Dir A:   ${DIR_A}"
echo "Dir B:   ${DIR_B}"
echo "Real:    ${REAL_DIR}"
echo "Output:  ${OUT_DIR}"

time python -m medgen.scripts.fid_vqvae_roundtrip_compare \
    --compression-checkpoint "${VQVAE_CKPT}" \
    --compression-type vqvae \
    --input-dirs "${DIR_A}" "${DIR_B}" \
    --real-dir "${REAL_DIR}" \
    --output-dir "${OUT_DIR}" \
    --num-roundtrip 20 \
    --num-baseline 100 \
    --fid-network radimagenet

echo "=== Done ==="
ls -la "${OUT_DIR}"
echo "--- fid_results.json ---"
cat "${OUT_DIR}/fid_results.json"
