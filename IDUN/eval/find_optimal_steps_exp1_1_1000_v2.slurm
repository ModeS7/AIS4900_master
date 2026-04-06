#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="optim_1_1_v2"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp1_1_1000_v2_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp1_1_1000_v2_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp1_1_1000 (repeat run): Pixel-space RFlow bravo 256x256x160, 1000 epochs
# Find optimal steps for: ImageNet FID, RadImageNet FID, PCA
# Uses LATEST checkpoint only (not best)

cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp1_1_1000 v2 (repeat run) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
CKPT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
PCA_MODEL="/cluster/work/modestas/AIS4900_master/data/brain_pca_256x256x160.npz"
BASE_ARGS="--data-root ${DATA_ROOT} --image-size 256 --num-volumes 25 --lo 10 --hi 100 --ref-split test --seed 42"
OUT_BASE="/cluster/work/modestas/MedicalDataSets/eval_optimal_steps_exp1_1_1000_v2"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi
echo "Checkpoint: $CKPT"

# ============================================================
# 1) Golden section search: ImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  Step 1: Optimal steps by ImageNet FID"
echo "=========================================="

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_BASE}_fid" \
    --metric fid \
    ${BASE_ARGS}

echo "--- ImageNet FID results ---"
cat "${OUT_BASE}_fid/search_results.json" 2>/dev/null

# ============================================================
# 2) Golden section search: RadImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  Step 2: Optimal steps by RadImageNet FID"
echo "=========================================="

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_BASE}_fid_rin" \
    --metric fid_radimagenet \
    ${BASE_ARGS}

echo "--- RadImageNet FID results ---"
cat "${OUT_BASE}_fid_rin/search_results.json" 2>/dev/null

# ============================================================
# 3) Golden section search: PCA brain shape error
# ============================================================
echo ""
echo "=========================================="
echo "  Step 3: Optimal steps by PCA error"
echo "=========================================="

time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT}" \
    --output-dir "${OUT_BASE}_pca" \
    --metric pca \
    ${BASE_ARGS}

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: exp1_1_1000 v2"
echo "=========================================="
echo "--- ImageNet FID (golden section) ---"
cat "${OUT_BASE}_fid/search_results.json" 2>/dev/null
echo ""
echo "--- RadImageNet FID (golden section) ---"
cat "${OUT_BASE}_fid_rin/search_results.json" 2>/dev/null
echo ""
echo "--- PCA (golden section) ---"
cat "${OUT_BASE}_pca/search_results.json" 2>/dev/null
echo ""
