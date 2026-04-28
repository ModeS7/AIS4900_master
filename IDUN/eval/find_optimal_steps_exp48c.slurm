#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="opt_48c"
#SBATCH --output=IDUN/output/eval/find_optimal_steps_exp48c_%j.out
#SBATCH --error=IDUN/output/eval/find_optimal_steps_exp48c_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp48c — three optima tests:
#   1. exp48c standalone
#   2. exp1_1_1000  → exp48c handoff at t=0.25
#   3. exp32_2_1000 → exp48c handoff at t=0.25

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== Find Optimal Steps: exp48c (3 tests) ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
EXP48_RUN=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp48c_lowt_only_lpips_huber_* 2>/dev/null | head -1)
CKPT_LOW="${EXP48_RUN}/checkpoint_latest.pt"
CKPT_EXP1_1_1000="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260301-020650/checkpoint_latest.pt"
CKPT_EXP32_2_1000="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt"
OUT_BASE="${CLUSTER_BASE}/MedicalDataSets/eval_optimal_steps_exp48c"

for ck in "$CKPT_LOW" "$CKPT_EXP1_1_1000" "$CKPT_EXP32_2_1000"; do
    [ ! -f "$ck" ] && { echo "ERROR: ckpt not found: $ck"; exit 1; }
done
echo "exp48c:        $CKPT_LOW"
echo "exp1_1_1000:   $CKPT_EXP1_1_1000"
echo "exp32_2_1000:  $CKPT_EXP32_2_1000"

echo ""
echo "=== [1/3] exp48c standalone ==="
time python -m medgen.scripts.find_optimal_steps \
    --checkpoint "${CKPT_LOW}" \
    --output-dir "${OUT_BASE}/standalone" \
    --metric fid_radimagenet \
    --data-root "${DATA_ROOT}" \
    --image-size 256 --num-volumes 25 \
    --lo 10 --hi 100 --ref-split test --seed 42

echo ""
echo "=== [2/3] exp1_1_1000 → exp48c @ t=0.25 ==="
time python -m medgen.scripts.find_optimal_steps \
    --high-t-checkpoint "${CKPT_EXP1_1_1000}" \
    --low-t-checkpoint "${CKPT_LOW}" \
    --handoff-t 0.25 \
    --output-dir "${OUT_BASE}/handoff_exp1_1_1000" \
    --metric fid_radimagenet \
    --data-root "${DATA_ROOT}" \
    --image-size 256 --num-volumes 25 \
    --lo 10 --hi 100 --ref-split test --seed 42

echo ""
echo "=== [3/3] exp32_2_1000 → exp48c @ t=0.25 ==="
time python -m medgen.scripts.find_optimal_steps \
    --high-t-checkpoint "${CKPT_EXP32_2_1000}" \
    --low-t-checkpoint "${CKPT_LOW}" \
    --handoff-t 0.25 \
    --output-dir "${OUT_BASE}/handoff_exp32_2_1000" \
    --metric fid_radimagenet \
    --data-root "${DATA_ROOT}" \
    --image-size 256 --num-volumes 25 \
    --lo 10 --hi 100 --ref-split test --seed 42

echo ""
echo "=== Summary: exp48c ==="
for sub in standalone handoff_exp1_1_1000 handoff_exp32_2_1000; do
    echo ""
    echo "--- ${sub} ---"
    cat "${OUT_BASE}/${sub}/search_results.json" 2>/dev/null
done
