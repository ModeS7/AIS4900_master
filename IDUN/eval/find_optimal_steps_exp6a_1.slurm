#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_exp6a_1"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp6a_1_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp6a_1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp6a_1: Pixel-space RFlow bravo UNCONDITIONAL (256x256x160)
# Architecture: UNet 270M (default_3d), channels [16,32,64,256,512,512]
# Mode: bravo unconditional (in=1, out=1) — ControlNet Stage 1
# Strategy: RFlow (continuous timesteps)
# Runs: ImageNet FID sweep, then RadImageNet FID sweep

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp6a_1 (Pixel RFlow bravo unconditional 256x256x160) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
BASE_ARGS="--data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --lo 10 --hi 50 --ref-split test --seed 42"

CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_best.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp6a_1_pixel_bravo_controlnet_stage1_20260311-003305/checkpoint_latest.pt"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found for exp6a_1, aborting"
    exit 1
fi

echo "Checkpoint: $CKPT"

# ============================================================
# 1) Optimize by ImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  exp6a_1: Optimal steps by ImageNet FID"
echo "=========================================="

OUT_FID="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp6a_1_fid"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_FID}" \
    --metric fid \
    ${BASE_ARGS}

echo "--- ImageNet FID results ---"
cat "${OUT_FID}/search_results.json"

# ============================================================
# 2) Optimize by RadImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  exp6a_1: Optimal steps by RadImageNet FID"
echo "=========================================="

OUT_FID_RIN="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp6a_1_fid_rin"

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_FID_RIN}" \
    --metric fid_radimagenet \
    ${BASE_ARGS}

echo "--- RadImageNet FID results ---"
cat "${OUT_FID_RIN}/search_results.json"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp6a_1"
echo "=========================================="
echo "--- ImageNet FID ---"
cat "${OUT_FID}/search_results.json" 2>/dev/null
echo ""
echo "--- RadImageNet FID ---"
cat "${OUT_FID_RIN}/search_results.json" 2>/dev/null
echo ""

conda deactivate
