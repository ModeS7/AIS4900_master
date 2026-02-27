#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_exp1_1"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp1_1_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp1_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp1_1: Pixel-space RFlow bravo (256x256x160)
# Architecture: UNet 270M (default_3d), channels [16,32,64,256,512,512]
# Mode: bravo (in=2 [noisy_bravo, seg], out=1), raw [0,1] normalization
# Strategy: RFlow (continuous timesteps)
# Two checkpoints: earlier run (0119) and later run (0129)

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp1_1 (Pixel RFlow bravo 256x256x160) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
COMMON_ARGS="--data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --lo 10 --hi 50 --metric fid --ref-split test --seed 42"

# ============================================================
# exp1_1 (run 0119): earlier training run
# ============================================================
echo ""
echo "=========================================="
echo "  exp1_1 (run 0119): Pixel RFlow 256x160"
echo "=========================================="

CKPT_0119="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_rflow_256x160_20260119-212710/checkpoint_best.pt"
if [ ! -f "$CKPT_0119" ]; then
    CKPT_0119="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_rflow_256x160_20260119-212710/checkpoint_latest.pt"
fi
OUT_0119="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp1_1_0119"

if [ -f "$CKPT_0119" ]; then
    echo "Checkpoint: $CKPT_0119"
    time python -m medgen.scripts.find_optimal_steps \
        --checkpoint "${CKPT_0119}" \
        --output-dir "${OUT_0119}" \
        ${COMMON_ARGS}
    echo "--- exp1_1 (0119) results ---"
    cat "${OUT_0119}/search_results.json"
else
    echo "WARNING: No checkpoint found for exp1_1 run 0119, skipping"
fi

# ============================================================
# exp1_1 (run 0129): later training run
# ============================================================
echo ""
echo "=========================================="
echo "  exp1_1 (run 0129): Pixel RFlow 256x160"
echo "=========================================="

CKPT_0129="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_rflow_256x160_20260129-233435/checkpoint_best.pt"
if [ ! -f "$CKPT_0129" ]; then
    CKPT_0129="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_pixel_bravo_rflow_256x160_20260129-233435/checkpoint_latest.pt"
fi
OUT_0129="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp1_1_0129"

if [ -f "$CKPT_0129" ]; then
    echo "Checkpoint: $CKPT_0129"
    time python -m medgen.scripts.find_optimal_steps \
        --checkpoint "${CKPT_0129}" \
        --output-dir "${OUT_0129}" \
        ${COMMON_ARGS}
    echo "--- exp1_1 (0129) results ---"
    cat "${OUT_0129}/search_results.json"
else
    echo "WARNING: No checkpoint found for exp1_1 run 0129, skipping"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  ALL RESULTS SUMMARY"
echo "=========================================="
for RUN in 0119 0129; do
    RESULT="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp1_1_${RUN}/search_results.json"
    if [ -f "$RESULT" ]; then
        echo "--- exp1_1 (${RUN}) ---"
        cat "$RESULT"
        echo ""
    fi
done

conda deactivate
