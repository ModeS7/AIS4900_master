#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --exclude=idun-01-05
#SBATCH --job-name="vel_div"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/velocity_divergence_all_%j.out
#SBATCH --error=IDUN/output/eval/velocity_divergence_all_%j.err

# Velocity divergence map (Phase 1 #4) — compare all 6 fine-tunes against
# exp1_1_1000 baseline velocity field, at a grid of t values.
#
# Skips fine-tune labels whose checkpoint path does not exist (e.g. if exp37_3
# hasn't finished yet), so it can be re-run as checkpoints become available.

cd ${SLURM_SUBMIT_DIR}

echo "=== Velocity Divergence: all 6 fine-tunes vs exp1_1_1000 baseline ==="
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
OUTPUT_DIR="/cluster/work/modestas/AIS4900_master/runs/eval/velocity_divergence_${TIMESTAMP}"

BASELINE="${RUNS_ROOT}/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"

# Build --fine-tunes arg dynamically — skip missing checkpoints so this SLURM is
# resumable as more fine-tunes come online.
FT_ARGS=""
for spec in \
    "exp32_1_1000=${RUNS_ROOT}/exp32_1_1000_pixel_bravo_ffl_20260412-151416/checkpoint_latest.pt" \
    "exp32_2_1000=${RUNS_ROOT}/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt" \
    "exp32_3_1000=${RUNS_ROOT}/exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057/checkpoint_latest.pt" \
    "exp37_1=${RUNS_ROOT}/exp37_1_pixel_bravo_lpips_hight_20260418-002202/checkpoint_latest.pt" \
    "exp37_2=${RUNS_ROOT}/exp37_2_pixel_bravo_lpips_ffl_hight_20260418-005105/checkpoint_latest.pt" \
    "exp37_3=${RUNS_ROOT}/exp37_3_pixel_bravo_lpips_ffl_short_20260420-010042/checkpoint_latest.pt"; do
    label="${spec%%=*}"
    path="${spec#*=}"
    if [ -f "${path}" ]; then
        FT_ARGS="${FT_ARGS} ${spec}"
        echo "  will compare: ${label}"
    else
        echo "  SKIP (checkpoint missing): ${label}"
    fi
done

if [ -z "${FT_ARGS}" ]; then
    echo "ERROR: no fine-tune checkpoints available"
    exit 1
fi

time python -m medgen.scripts.analyze_velocity_divergence \
    --baseline "${BASELINE}" \
    --fine-tunes ${FT_ARGS} \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-volumes 5 \
    --depth 160 \
    --seg-split test1 \
    --seed 42

echo "Results saved to: ${OUTPUT_DIR}"
