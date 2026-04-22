#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="traj_dense"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/trajectory_emergence_dense_%j.out
#SBATCH --error=IDUN/output/eval/trajectory_emergence_dense_%j.err

# Dense trajectory emergence for exp1_1_1000 only — 4× the t-resolution
# (200 Euler steps, Δt=0.005) and 3.3× the seeds (10) for tighter
# emergence_t confidence. Feeds analyze_emergence_timeline.py.

cd ${SLURM_SUBMIT_DIR}

echo "=== Dense Trajectory Emergence: exp1_1_1000 ==="
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
CKPT="${RUNS_ROOT}/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
# FIXED output dir (no timestamp) — enables seed-level resume from a prior
# timed-out run. The script detects existing emergence_metrics.json in this
# directory and skips seeds already completed.
OUT_DIR="/cluster/work/modestas/AIS4900_master/runs/eval/trajectory_emergence_dense_exp1_1_1000"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

time python -m medgen.scripts.analyze_generation_trajectory \
    --bravo-model "${CKPT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUT_DIR}" \
    --num-seeds 10 \
    --num-steps 200 \
    --depth 160 \
    --seg-split test1 \
    --seed-base 42 \
    --vessel-every-n-steps 2

echo "Results saved to: ${OUT_DIR}"

# Post-process into emergence timeline once the job completes
time python -m medgen.scripts.analyze_emergence_timeline \
    --input-dirs "${OUT_DIR}" \
    --output-dir "${OUT_DIR}/timeline" \
    --thesis-dir /home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/emergence_timeline_dense_exp1_1_1000

echo "Timeline saved to: ${OUT_DIR}/timeline"
