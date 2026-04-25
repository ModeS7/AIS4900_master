#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="vqvae_pairs"
#SBATCH --output=IDUN/output/eval/precompute_vqvae_pairs_%j.out
#SBATCH --error=IDUN/output/eval/precompute_vqvae_pairs_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Precompute (real, real_rt) training pairs for exp43 VQ-VAE deblur.
# Creates symlinks to clean + seg, writes degraded_000.nii.gz per subject.
# Output: <CLUSTER_BASE>/MedicalDataSets/vqvae_pairs/{train,val,test,test_new}/
# Runtime: ~207 subjects × 1s each ≈ 5 min.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Precompute VQ-VAE pairs for exp43 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_ALLOC_CONF=expandable_segments:True

VQVAE_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
OUT_DIR="${CLUSTER_BASE}/MedicalDataSets/vqvae_pairs"

for p in "${VQVAE_CKPT}" "${DATA_ROOT}"; do
    if [ ! -e "${p}" ]; then
        echo "ERROR: missing ${p}"
        exit 1
    fi
done

mkdir -p "${OUT_DIR}"
echo "VQ-VAE: ${VQVAE_CKPT}"
echo "Input:  ${DATA_ROOT}"
echo "Output: ${OUT_DIR}"

time python -m medgen.scripts.precompute_vqvae_pairs \
    --compression-checkpoint "${VQVAE_CKPT}" \
    --compression-type vqvae \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --splits train val test_new \
    --skip-existing

echo "=== Done ==="
echo "Summary:"
for split in train val test_new; do
    count=$(find "${OUT_DIR}/${split}" -name "degraded_000.nii.gz" 2>/dev/null | wc -l)
    echo "  ${split}: ${count} pairs"
done
