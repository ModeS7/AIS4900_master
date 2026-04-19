#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --job-name="pca_ablation"
#SBATCH --output=IDUN/output/eval/pca_ablation_clean_%j.out
#SBATCH --error=IDUN/output/eval/pca_ablation_clean_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Clean PCA filter ablation on the two nofilter datasets.
#   - exp1_1_1000: known-good generator (should have no bad brains → filter dormant)
#   - exp23:       known-problematic generator (ScoreAug leak → filter catches tail)
# Both generated with ImageNet-FID optima (32 steps / 27 + shift=1.22 respectively),
# real test1 seg masks, NO PCA filter applied at generation — raw distribution.
#
# Fits ablation PCAs (train+val only) if missing, then runs ablation on both sets.
# CPU queue — no GPU needed (numpy SVD + matplotlib, torch only for interpolate).

cd ${SLURM_SUBMIT_DIR}

echo "=== Clean PCA filter ablation ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

# Cluster paths
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
ABLATION_DIR="/cluster/work/modestas/AIS4900_master/data/ablation"
GEN_ROOT="/cluster/work/modestas/MedicalDataSets/generated"
OUTPUT_ROOT="/cluster/work/modestas/AIS4900_master/runs/eval"

mkdir -p "${ABLATION_DIR}"
mkdir -p "${OUTPUT_ROOT}"

# --- Step 1: fit ablation PCAs if missing (train+val only, 100 components) ---
if [ ! -f "${ABLATION_DIR}/brain_pca_64x64x40.npz" ]; then
    echo ""
    echo "[1/3] Fitting coarse PCA (64x64x40)..."
    time python -m medgen.scripts.compute_brain_pca \
        --data-root "${DATA_ROOT}" \
        --output "${ABLATION_DIR}/brain_pca_64x64x40.npz" \
        --splits train val \
        --n-components 100 \
        --pca-size 64 --pca-depth 40
else
    echo "[1/3] Coarse PCA already fitted: ${ABLATION_DIR}/brain_pca_64x64x40.npz"
fi

if [ ! -f "${ABLATION_DIR}/brain_pca_128x128x80.npz" ]; then
    echo ""
    echo "[2/3] Fitting fine PCA (128x128x80)..."
    time python -m medgen.scripts.compute_brain_pca \
        --data-root "${DATA_ROOT}" \
        --output "${ABLATION_DIR}/brain_pca_128x128x80.npz" \
        --splits train val \
        --n-components 100 \
        --pca-size 128 --pca-depth 80
else
    echo "[2/3] Fine PCA already fitted: ${ABLATION_DIR}/brain_pca_128x128x80.npz"
fi

# --- Step 3: run ablation on both generated sets ---
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo ""
echo "[3/3a] Ablation: exp1_1_1000 (known-good generator)"
OUT_A="${OUTPUT_ROOT}/pca_ablation_clean_exp1_1_1000_${TIMESTAMP}"
time python -m medgen.scripts.ablate_pca_filter \
    --pca-coarse "${ABLATION_DIR}/brain_pca_64x64x40.npz" \
    --pca-fine   "${ABLATION_DIR}/brain_pca_128x128x80.npz" \
    --real-dir   "${DATA_ROOT}/test1" \
    --gen-dir    "${GEN_ROOT}/exp1_1_1000_bravo_real_seg_test1_nofilter" \
    --train-root "${DATA_ROOT}" \
    --output-dir "${OUT_A}" \
    --thesis-dir ""
echo "  -> ${OUT_A}"

echo ""
echo "[3/3b] Ablation: exp23 (known-problematic ScoreAug generator)"
OUT_B="${OUTPUT_ROOT}/pca_ablation_clean_exp23_${TIMESTAMP}"
time python -m medgen.scripts.ablate_pca_filter \
    --pca-coarse "${ABLATION_DIR}/brain_pca_64x64x40.npz" \
    --pca-fine   "${ABLATION_DIR}/brain_pca_128x128x80.npz" \
    --real-dir   "${DATA_ROOT}/test1" \
    --gen-dir    "${GEN_ROOT}/exp23_bravo_real_seg_test1_nofilter" \
    --train-root "${DATA_ROOT}" \
    --output-dir "${OUT_B}" \
    --thesis-dir ""
echo "  -> ${OUT_B}"

echo ""
echo "=== Ablation complete ==="
echo "exp1_1_1000: ${OUT_A}"
echo "exp23:       ${OUT_B}"
