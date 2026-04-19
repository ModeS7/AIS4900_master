#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="pca_dualtrip"
#SBATCH --output=IDUN/output/eval/eval_pca_dual_triple_%j.out
#SBATCH --error=IDUN/output/eval/eval_pca_dual_triple_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Evaluate PCA + FID/KID/CMMD for dual and triple mode experiments
# 25 volumes, 20 steps, 128x128x160

cd ${SLURM_SUBMIT_DIR}

echo "=== PCA + FID Evaluation: Dual & Triple Modes ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
PCA_MODEL="${CLUSTER_BASE}/AIS4900_master/data/brain_pca_256x256x160.npz"
OUT_BASE="${CLUSTER_BASE}/MedicalDataSets/eval_pca_dual_triple"

MODELS=(
    "exp1v2:${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_pixel_dual_20260326-164147/checkpoint_latest.pt"
    "exp1v2_1:${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1v2_1_pixel_dual_joint_norm_20260327-031630/checkpoint_latest.pt"
    "exp1v3:${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_pixel_triple_20260326-164117/checkpoint_latest.pt"
    "exp1v3_1:${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1v3_1_pixel_triple_joint_norm_20260327-033611/checkpoint_latest.pt"
)

for entry in "${MODELS[@]}"; do
    NAME="${entry%%:*}"
    CKPT="${entry#*:}"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP $NAME: checkpoint not found at $CKPT"
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  $NAME (20 steps, 128x128x160)"
    echo "=========================================="

    time python -m medgen.scripts.eval_steps_pca \
        --checkpoint "${CKPT}" \
        --data-root "${DATA_ROOT}" \
        --pca-model "${PCA_MODEL}" \
        --output-dir "${OUT_BASE}/${NAME}" \
        --image-size 128 --depth 160 \
        --num-volumes 25 \
        --step-list 20 \
        --shift-ratio 1.0 \
        --seed 42 \
        --compute-fid \
        --ref-split test1

    echo "--- $NAME done ---"
done

# === Summary ===
echo ""
echo "=========================================="
echo "  RESULTS SUMMARY: Dual & Triple"
echo "=========================================="
for dir in ${OUT_BASE}/*/; do
    NAME=$(basename "$dir")
    JSON="${dir}pca_step_results.json"
    if [ -f "$JSON" ]; then
        python3 -c "
import json
d = json.load(open('$JSON'))
r = d['results'][0]
pca = f\"PCA={r['mean_error']:.6f} pass={r['pass_rate']:.0%}\"
fid = f\"FID={r.get('FID', 'N/A'):.2f}\" if 'FID' in r else 'FID=N/A'
kid = f\"KID={r.get('KID_mean', 'N/A'):.4f}\" if 'KID_mean' in r else 'KID=N/A'
cmmd = f\"CMMD={r.get('CMMD', 'N/A'):.4f}\" if 'CMMD' in r else 'CMMD=N/A'
print(f'  ${NAME}: {pca} | {fid} | {kid} | {cmmd}')
"
    fi
done

conda deactivate
