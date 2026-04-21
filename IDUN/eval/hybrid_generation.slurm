#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="hybrid_gen"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/hybrid_generation_%j.out
#SBATCH --error=IDUN/output/eval/hybrid_generation_%j.err

# Phase 1 #5 — hybrid generation ablation.
# Compares 8 configs per fine-tune (baseline, ft, 3 splits, 3 reverses) on 5 seeds.
# Primary comparison: baseline vs exp37_3 (spectrum-analysis winner).

cd ${SLURM_SUBMIT_DIR}

echo "=== Hybrid generation: baseline vs exp37_3 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE_MIB" -lt 60000 ]; then
    echo "ERROR: Only ${FREE_MIB} MiB free on GPU. Aborting."
    exit 78
fi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
RUNS_ROOT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUT_ROOT="/cluster/work/modestas/AIS4900_master/runs/eval"

BASELINE="${RUNS_ROOT}/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
FT_EXP37_3="${RUNS_ROOT}/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt"

# ---- Primary: baseline vs exp37_3 (the spectrum-analysis winner) ----
if [ -f "${FT_EXP37_3}" ]; then
    OUT_DIR="${OUT_ROOT}/hybrid_generation_exp37_3_${TIMESTAMP}"
    echo ""
    echo "=== Running: baseline vs exp37_3 ==="
    time python -m medgen.scripts.analyze_hybrid_generation \
        --baseline "${BASELINE}" \
        --fine-tune "${FT_EXP37_3}" \
        --ft-label exp37_3 \
        --data-root "${DATA_ROOT}" \
        --real-dir "${DATA_ROOT}/test1" \
        --output-dir "${OUT_DIR}" \
        --num-seeds 5 \
        --num-steps 32 \
        --boundaries 0.3 0.5 0.7 \
        --depth 160 \
        --seg-split test1 \
        --seed-base 42
    echo "Saved: ${OUT_DIR}"
else
    echo "SKIP: exp37_3 checkpoint not found (training may still be running)"
fi

# ---- Optional secondary: baseline vs exp37_2 (regressed variant, for comparison) ----
FT_EXP37_2="${RUNS_ROOT}/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt"
if [ -f "${FT_EXP37_2}" ]; then
    OUT_DIR="${OUT_ROOT}/hybrid_generation_exp37_2_${TIMESTAMP}"
    echo ""
    echo "=== Running: baseline vs exp37_2 (secondary comparison) ==="
    time python -m medgen.scripts.analyze_hybrid_generation \
        --baseline "${BASELINE}" \
        --fine-tune "${FT_EXP37_2}" \
        --ft-label exp37_2 \
        --data-root "${DATA_ROOT}" \
        --real-dir "${DATA_ROOT}/test1" \
        --output-dir "${OUT_DIR}" \
        --num-seeds 5 \
        --num-steps 32 \
        --boundaries 0.3 0.5 0.7 \
        --depth 160 \
        --seg-split test1 \
        --seed-base 42
    echo "Saved: ${OUT_DIR}"
fi

echo ""
echo "=== Hybrid generation analysis complete ==="
