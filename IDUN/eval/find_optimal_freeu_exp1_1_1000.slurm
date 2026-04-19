#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="freeu_exp1_1_1k"
#SBATCH --output=IDUN/output/eval/find_optimal_freeu_exp1_1_1000_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_freeu_exp1_1_1000_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# FreeU grid search for exp1_1_1000
# Pixel-space RFlow bravo (256x256x160), 1000 epochs, UNet 270M
# Grid: b=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5] x s=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
# Total: 36 configs x 25 volumes each
# Euler 25 steps (proven optimal)

# === Environment setup ===
cd ${SLURM_SUBMIT_DIR}

echo "=== FreeU Grid Search: exp1_1_1000 (Pixel RFlow bravo 256x256x160, 1000 epochs) ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"

CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt"
if [ ! -f "$CKPT" ]; then
    CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_best.pt"
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: No checkpoint found for exp1_1_1000, aborting"
    exit 1
fi

echo "Checkpoint: $CKPT"

# ============================================================
# 1) FreeU grid search by ImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  FreeU Grid Search by ImageNet FID"
echo "=========================================="

OUT_FID="${CLUSTER_BASE}/MedicalDataSets/eval_freeu_exp1_1_1000_fid"

time python -m medgen.scripts.find_optimal_freeu \
    --checkpoint "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_FID}" \
    --image-size 256 \
    --num-volumes 25 \
    --num-steps 25 \
    --metric fid \
    --ref-split test \
    --seed 42 \
    --b-values 1.0 1.1 1.2 1.3 1.4 1.5 \
    --s-values 0.2 0.4 0.6 0.8 0.9 1.0

echo "--- ImageNet FID results ---"
cat "${OUT_FID}/freeu_results.json"

# ============================================================
# 2) FreeU grid search by RadImageNet FID
# ============================================================
echo ""
echo "=========================================="
echo "  FreeU Grid Search by RadImageNet FID"
echo "=========================================="

OUT_FID_RIN="${CLUSTER_BASE}/MedicalDataSets/eval_freeu_exp1_1_1000_fid_rin"

time python -m medgen.scripts.find_optimal_freeu \
    --checkpoint "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_FID_RIN}" \
    --image-size 256 \
    --num-volumes 25 \
    --num-steps 25 \
    --metric fid_radimagenet \
    --ref-split test \
    --seed 42 \
    --b-values 1.0 1.1 1.2 1.3 1.4 1.5 \
    --s-values 0.2 0.4 0.6 0.8 0.9 1.0

echo "--- RadImageNet FID results ---"
cat "${OUT_FID_RIN}/freeu_results.json"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: FreeU exp1_1_1000"
echo "=========================================="
echo "--- ImageNet FID ---"
python3 -c "
import json
with open('${OUT_FID}/freeu_results.json') as f:
    d = json.load(f)
if d.get('best'):
    print(f\"Best: b={d['best']['backbone_scale']}, s={d['best']['skip_scale']}\")
for r in sorted(d['results'], key=lambda x: x['fid'])[:5]:
    print(f\"  b={r['backbone_scale']:.2f} s={r['skip_scale']:.2f} -> FID={r['fid']:.2f} FID_RIN={r['fid_radimagenet']:.2f}\")
" 2>/dev/null

echo ""
echo "--- RadImageNet FID ---"
python3 -c "
import json
with open('${OUT_FID_RIN}/freeu_results.json') as f:
    d = json.load(f)
if d.get('best'):
    print(f\"Best: b={d['best']['backbone_scale']}, s={d['best']['skip_scale']}\")
for r in sorted(d['results'], key=lambda x: x['fid_radimagenet'])[:5]:
    print(f\"  b={r['backbone_scale']:.2f} s={r['skip_scale']:.2f} -> FID={r['fid']:.2f} FID_RIN={r['fid_radimagenet']:.2f}\")
" 2>/dev/null

conda deactivate
