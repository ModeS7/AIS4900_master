#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="phema"
#SBATCH --output=IDUN/output/eval/phema_sweep_%j.out
#SBATCH --error=IDUN/output/eval/phema_sweep_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

# Post-hoc EMA sweep: evaluate all sigma_rel values for one experiment.
#
# Usage:
#   RUN_DIR=/cluster/work/.../exp1o_1_... sbatch eval_phema_sweep.slurm
#
# For seg models:
#   RUN_DIR=/cluster/work/.../exp14_2_... IS_SEG=1 sbatch eval_phema_sweep.slurm
#
# The script finds phema_checkpoints/ inside the run dir, synthesizes
# models at 11 sigma_rel values, generates 25 volumes each, and computes
# FID/KID/CMMD/FWD/PCA metrics. Also evaluates raw (non-EMA) model.

cd ${SLURM_SUBMIT_DIR}

if [ -z "${RUN_DIR}" ]; then
    echo "ERROR: Set RUN_DIR environment variable."
    echo "Usage: RUN_DIR=/path/to/run sbatch eval_phema_sweep.slurm"
    exit 1
fi

EXP_BASENAME=$(basename "${RUN_DIR}")

echo "=== Post-hoc EMA Sweep: ${EXP_BASENAME} ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
echo "Run dir: ${RUN_DIR}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"

SEG_FLAG=""
if [ "${IS_SEG:-0}" = "1" ]; then
    SEG_FLAG="--is-seg"
fi

time python -m medgen.scripts.synthesize_phema \
    --run-dir "${RUN_DIR}" \
    --data-root "${DATA_ROOT}" \
    --num-volumes 25 \
    --num-steps 25 \
    --depth 160 \
    --image-size 256 \
    --seed 42 \
    --include-raw \
    ${SEG_FLAG}

echo ""
echo "=== Results ==="
cat "${RUN_DIR}/phema_sweep/phema_sweep_results.json" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for name, r in data.items():
        if 'error' in r:
            print(f'{name}: ERROR - {r[\"error\"]}')
        else:
            fid = r.get('fid', 0)
            fwd = r.get('fwd', 0)
            pca = r.get('pca_pass_rate', 0) * 100
            print(f'{name}: FID={fid:.2f}  FWD={fwd:.4f}  PCA={pca:.0f}%')
except: pass
" 2>/dev/null

conda deactivate
