#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="dbg_rt_exp9_1"
#SBATCH --output=IDUN/output/debug/ldm_roundtrip_exp9_1_%j.out
#SBATCH --error=IDUN/output/debug/ldm_roundtrip_exp9_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# Debug round-trip test for exp9_1 (LDM 4x UNet, bravo_seg_cond)
# Tests denoising at various noise levels to isolate pipeline vs model issues

cd ${SLURM_SUBMIT_DIR}

echo "=== Debug LDM Round-trip: exp9_1 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === PATHS ===
CHECKPOINT=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_latent/exp9_1_ldm_4x_bravo_*/checkpoint_latest.pt 2>/dev/null | head -1)
COMPRESSION_CHECKPOINT=/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets/debug_ldm_roundtrip_exp9_1"

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: No checkpoint found"
    exit 1
fi

echo "Diffusion: $CHECKPOINT"
echo "Compression: $COMPRESSION_CHECKPOINT"
echo "Latent stats: read from diffusion checkpoint"
echo ""

time python -m medgen.scripts.debug_ldm_roundtrip \
    --checkpoint "${CHECKPOINT}" \
    --compression-checkpoint "${COMPRESSION_CHECKPOINT}" \
    --compression-type vqvae \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-volumes 3 \
    --num-steps 25 \
    --seed 42

echo "=== Done ==="
cat "${OUTPUT_DIR}/roundtrip_results.json"

conda deactivate
