#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="traj_all"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/trajectory_emergence_all_%j.out
#SBATCH --error=IDUN/output/eval/trajectory_emergence_all_%j.err

# Trajectory emergence analysis (Phase 1 #3) for all 7 models in one job.
# Each model runs 3 seeds × 50 Euler steps. Per-model runtime ~50-60 min.
# Total: ~6-7 hours.

cd ${SLURM_SUBMIT_DIR}

echo "=== Trajectory Emergence: all 7 models ==="
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
OUT_ROOT="/cluster/work/modestas/AIS4900_master/runs/eval"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Checkpoint paths — one per model
CKPT_EXP1_1_1000="${RUNS_ROOT}/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
CKPT_EXP32_1_1000="${RUNS_ROOT}/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt"
CKPT_EXP32_2_1000="${RUNS_ROOT}/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt"
CKPT_EXP32_3_1000="${RUNS_ROOT}/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt"
CKPT_EXP37_1="${RUNS_ROOT}/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt"
CKPT_EXP37_2="${RUNS_ROOT}/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt"
CKPT_EXP37_3="${RUNS_ROOT}/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt"

run_model() {
    local label="$1"
    local ckpt="$2"
    local out_dir="${OUT_ROOT}/trajectory_emergence_${label}_${TIMESTAMP}"

    echo ""
    echo "=========================================================="
    echo "  [${label}] Starting trajectory analysis"
    echo "  checkpoint: ${ckpt}"
    echo "  output:     ${out_dir}"
    echo "=========================================================="

    if [ ! -f "${ckpt}" ]; then
        echo "  SKIP: checkpoint not found"
        return
    fi

    time python -m medgen.scripts.analyze_generation_trajectory \
        --bravo-model "${ckpt}" \
        --data-root "${DATA_ROOT}" \
        --output-dir "${out_dir}" \
        --num-seeds 3 \
        --num-steps 50 \
        --depth 160 \
        --seg-split test1 \
        --seed-base 42 \
        --vessel-every-n-steps 5

    echo "  [${label}] done"
}

run_model "exp1_1_1000"  "${CKPT_EXP1_1_1000}"
run_model "exp32_1_1000" "${CKPT_EXP32_1_1000}"
run_model "exp32_2_1000" "${CKPT_EXP32_2_1000}"
run_model "exp32_3_1000" "${CKPT_EXP32_3_1000}"
run_model "exp37_1"      "${CKPT_EXP37_1}"
run_model "exp37_2"      "${CKPT_EXP37_2}"
run_model "exp37_3"      "${CKPT_EXP37_3}"

echo ""
echo "=== All 7 models complete ==="
echo "Output directories:"
for label in exp1_1_1000 exp32_1_1000 exp32_2_1000 exp32_3_1000 exp37_1 exp37_2 exp37_3; do
    echo "  ${label}: ${OUT_ROOT}/trajectory_emergence_${label}_${TIMESTAMP}"
done
