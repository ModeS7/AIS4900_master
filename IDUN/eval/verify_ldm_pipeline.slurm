#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="verify_ldm"
#SBATCH --output=IDUN/output/eval/verify_ldm_pipeline_%j.out
#SBATCH --error=IDUN/output/eval/verify_ldm_pipeline_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# Verify LDM pipeline consistency
# Only needs: diffusion checkpoint + compression checkpoint + data root

cd ${SLURM_SUBMIT_DIR}

echo "=== Verify LDM Pipeline ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === PATHS ===
COMPRESSION_CHECKPOINT=/cluster/work/modestas/AIS4900_master/runs/compression_3d/multi_modality/exp8_1_256x160_20260107-031153/checkpoint_latest.pt
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"

# Auto-find LDM diffusion checkpoint (exp9_1)
DIFFUSION_CHECKPOINT=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo_seg_cond/exp9_1_*/checkpoint_latest.pt 2>/dev/null | head -1)

if [ -z "$DIFFUSION_CHECKPOINT" ]; then
    echo "ERROR: No LDM diffusion checkpoint found for exp9_1_*"
    exit 1
fi

echo "Diffusion: $DIFFUSION_CHECKPOINT"
echo "Compression: $COMPRESSION_CHECKPOINT"
echo "Data root: $DATA_ROOT"
echo ""

python -m medgen.scripts.verify_ldm_pipeline \
    --diffusion-checkpoint "${DIFFUSION_CHECKPOINT}" \
    --compression-checkpoint "${COMPRESSION_CHECKPOINT}" \
    --compression-type vqvae \
    --data-root "${DATA_ROOT}"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== ALL CHECKS PASSED — safe to retrain ==="
else
    echo "=== CHECKS FAILED — DO NOT retrain until fixed ==="
fi

conda deactivate
exit $EXIT_CODE
