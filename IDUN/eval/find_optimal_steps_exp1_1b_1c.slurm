#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --job-name="optim_exp1_1b_1c"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp1_1b_1c_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp1_1b_1c_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# Optimal step search for three pixel-space normalization experiments:
#   exp1:  raw [0,1] baseline
#   exp1b: [-1,1] rescaled (training.rescale_data=true)
#   exp1c: brain-only N(0,1) (pixel_norm=bravo, shift=0.2033, scale=0.0832)
# All: UNet 5L 270M, rflow, LR=1e-4, 128x128x160

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp1 + exp1b + exp1c ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"

# Common args
COMMON_ARGS="--data-root ${DATA_ROOT} --image-size 128 --num-volumes 25 --lo 10 --hi 50 --metric fid --seed 42"

# ============================================================
# exp1: Pixel bravo [0,1] baseline
# ============================================================
echo ""
echo "=========================================="
echo "  exp1: Pixel bravo [0,1] baseline"
echo "=========================================="

CKPT_EXP1=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_pixel_bravo_*/checkpoint_best.pt 2>/dev/null | head -1)
OUT_EXP1="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp1"

if [ -z "$CKPT_EXP1" ]; then
    echo "WARNING: No checkpoint found for exp1_pixel_bravo_* — skipping"
else
    echo "Checkpoint: $CKPT_EXP1"
    time python -m medgen.scripts.find_optimal_steps \
        --checkpoint "${CKPT_EXP1}" \
        --output-dir "${OUT_EXP1}" \
        ${COMMON_ARGS}
    echo "--- exp1 results ---"
    cat "${OUT_EXP1}/search_results.json"
fi

# ============================================================
# exp1b: Pixel bravo [-1,1] rescaled
# ============================================================
echo ""
echo "=========================================="
echo "  exp1b: Pixel bravo [-1,1] rescaled"
echo "=========================================="

CKPT_EXP1B=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1b_*/checkpoint_best.pt 2>/dev/null | head -1)
OUT_EXP1B="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp1b"

if [ -z "$CKPT_EXP1B" ]; then
    echo "WARNING: No checkpoint found for exp1b_* — skipping"
else
    echo "Checkpoint: $CKPT_EXP1B"
    time python -m medgen.scripts.find_optimal_steps \
        --checkpoint "${CKPT_EXP1B}" \
        --output-dir "${OUT_EXP1B}" \
        ${COMMON_ARGS}
    echo "--- exp1b results ---"
    cat "${OUT_EXP1B}/search_results.json"
fi

# ============================================================
# exp1c: Pixel bravo brain-only N(0,1) normalized
# ============================================================
echo ""
echo "=========================================="
echo "  exp1c: Pixel bravo brain N(0,1)"
echo "=========================================="

CKPT_EXP1C=$(ls -d /cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1c_*/checkpoint_best.pt 2>/dev/null | head -1)
OUT_EXP1C="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp1c"

if [ -z "$CKPT_EXP1C" ]; then
    echo "WARNING: No checkpoint found for exp1c_* — skipping"
else
    echo "Checkpoint: $CKPT_EXP1C"
    time python -m medgen.scripts.find_optimal_steps \
        --checkpoint "${CKPT_EXP1C}" \
        --output-dir "${OUT_EXP1C}" \
        ${COMMON_ARGS}
    echo "--- exp1c results ---"
    cat "${OUT_EXP1C}/search_results.json"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  ALL RESULTS SUMMARY"
echo "=========================================="
for EXP in exp1 exp1b exp1c; do
    RESULT="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_${EXP}/search_results.json"
    if [ -f "$RESULT" ]; then
        echo "--- ${EXP} ---"
        cat "$RESULT"
        echo ""
    fi
done

conda deactivate
