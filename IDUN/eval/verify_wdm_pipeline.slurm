#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="verify_wdm"
#SBATCH --output=IDUN/output/eval/verify_wdm_pipeline_%j.out
#SBATCH --error=IDUN/output/eval/verify_wdm_pipeline_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# Verify WDM pipeline consistency after normalization persistence fix
# Must PASS all checks before retraining

cd ${SLURM_SUBMIT_DIR}

echo "=== Verify WDM Pipeline ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === PATHS ===
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"

# Auto-find WDM checkpoint (exp12b_2)
CHECKPOINT=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp12b_2_*/checkpoint_latest.pt 2>/dev/null | head -1)

echo "Data root: $DATA_ROOT"
if [ -n "$CHECKPOINT" ]; then
    echo "Checkpoint: $CHECKPOINT"
else
    echo "No WDM checkpoint found — running data-only checks"
fi
echo ""

# exp12b_2 trains at 128x128x160 with wavelet normalize (no rescale)
# Use --max-volumes 100 for good stats estimate while keeping runtime short
CMD="python -m medgen.scripts.verify_wdm_pipeline \
    --data-root ${DATA_ROOT} \
    --image-size 128 \
    --depth 160 \
    --max-volumes 100"

if [ -n "$CHECKPOINT" ]; then
    CMD="${CMD} --checkpoint ${CHECKPOINT}"
fi

eval $CMD

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== ALL CHECKS PASSED — safe to retrain ==="
else
    echo "=== CHECKS FAILED — DO NOT retrain until fixed ==="
fi

conda deactivate
exit $EXIT_CODE
