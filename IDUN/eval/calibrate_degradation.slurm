#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="calib_deg"
#SBATCH --output=IDUN/output/eval/calibrate_degradation_%j.out
#SBATCH --error=IDUN/output/eval/calibrate_degradation_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Calibrate SDEdit degradation strength for restoration training.
# Compares power spectra of real, generated, and SDEdit volumes at various t₀.
# Output: recommended t₀ range + spectrum plots.

cd ${SLURM_SUBMIT_DIR}

echo "=== Calibrate Degradation Strength for Restoration ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

BRAVO_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN_DIR_IN="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
GEN_DIR_RIN="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525"
OUT_DIR="${CLUSTER_BASE}/AIS4900_master/eval/calibrate_degradation"

time python -m medgen.scripts.calibrate_degradation \
    --bravo-model "${BRAVO_CKPT}" \
    --data-root "${DATA_ROOT}" \
    --generated-dir "${GEN_DIR_IN}" \
    --generated-dir-rin "${GEN_DIR_RIN}" \
    --output-dir "${OUT_DIR}" \
    --num-volumes 5 \
    --num-steps 32 \
    --depth 160 \
    --image-size 256

echo ""
echo "=== Results ==="
cat "${OUT_DIR}/calibration_results.json" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('Best t0:', json.dumps(data.get('best_t0', {}), indent=2))
print('Recommended range:', data.get('recommended_range', 'N/A'))
" 2>/dev/null

conda deactivate
