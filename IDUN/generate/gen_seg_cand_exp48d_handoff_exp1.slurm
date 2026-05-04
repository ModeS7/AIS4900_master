#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_e48dh"
#SBATCH --output=IDUN/output/generate/gen_seg_cand_exp48d_handoff_exp1_%j.out
#SBATCH --error=IDUN/output/generate/gen_seg_cand_exp48d_handoff_exp1_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --signal=B:SIGUSR1@120

# Generate 525 BRAVO volumes for segmentation candidate: exp48d_handoff_exp1
#   low-t : exp48d_lowt_only_huber_lpips_*
#   high-t: exp1_1_1000_pixel_bravo_*  (pure-MSE seed @ t=0.25)
#   steps : 100 (RadImageNet-FID dense-grid optimum)
# Highest band_very_high (sharpest HF) variant.
# Filters: atlas, brain mask, brain-PCA, seg-PCA, single component.

cd ${SLURM_SUBMIT_DIR}

CANDIDATE="exp48d_handoff_exp1"
GLOB="exp48d_lowt_only_huber_lpips_*"
HIGH_T_GLOB="exp1_1_1000_pixel_bravo_*"
STEPS=100
NUM=525
CHAIN_MAX=10
CHAIN_COUNT=${CHAIN_COUNT:-0}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
OUTPUT_SUBDIR="seg_candidates_525/${CANDIDATE}"

handle_timeout() {
    echo "=== Wall time approaching. Resubmitting [${CANDIDATE}]... ==="
    kill $PID 2>/dev/null
    wait $PID 2>/dev/null
    NEXT=$((CHAIN_COUNT + 1))
    if [ "$NEXT" -ge "$CHAIN_MAX" ]; then
        echo "Chain limit reached (${CHAIN_MAX}). Stopping."
        exit 1
    fi
    sbatch --export=ALL,CHAIN_COUNT=${NEXT} "$0"
    echo "Resubmitted [${CANDIDATE}] segment ${NEXT}/${CHAIN_MAX}"
    exit 0
}
trap handle_timeout SIGUSR1

echo "=== Generate 525 vols [${CANDIDATE}] | Chain ${CHAIN_COUNT}/${CHAIN_MAX} ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

SEG_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"

RUN_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${GLOB} 2>/dev/null | head -1)
CKPT="${RUN_DIR}/checkpoint_latest.pt"
[ ! -f "${CKPT}" ] && CKPT="${RUN_DIR}/checkpoint_best.pt"
if [ ! -f "${CKPT}" ]; then
    echo "ERROR: low-t ckpt not found for ${CANDIDATE} (glob=${GLOB})"
    exit 3
fi

HIGH_RUN_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo/${HIGH_T_GLOB} 2>/dev/null | head -1)
HIGH_CKPT="${HIGH_RUN_DIR}/checkpoint_latest.pt"
[ ! -f "${HIGH_CKPT}" ] && HIGH_CKPT="${HIGH_RUN_DIR}/checkpoint_best.pt"
if [ ! -f "${HIGH_CKPT}" ]; then
    echo "ERROR: high-t ckpt not found for ${CANDIDATE} (glob=${HIGH_T_GLOB})"
    exit 3
fi

GEN_DIR="${CLUSTER_BASE}/MedicalDataSets/generated/${OUTPUT_SUBDIR}"
mkdir -p "${GEN_DIR}"
CURRENT=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
echo "Resuming [${CANDIDATE}] from sample ${CURRENT}/${NUM}"
echo "  low-t  ckpt : ${CKPT}"
echo "  high-t ckpt : ${HIGH_CKPT}"
echo "  bravo steps : ${STEPS}"
echo "  handoff t   : 0.25"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    image_size=256 \
    depth=160 \
    seg_model="${SEG_CKPT}" \
    image_model="${CKPT}" \
    +image_model_high_t="${HIGH_CKPT}" \
    handoff_t=0.25 \
    num_images=${NUM} \
    current_image=${CURRENT} \
    num_steps_seg=40 \
    num_steps_bravo=${STEPS} \
    validate_size_bins=false \
    validate_brain_mask=true \
    brain_atlas_path=auto \
    +brain_pca_path=auto \
    +seg_pca_path=auto \
    output_subdir="${OUTPUT_SUBDIR}" \
    verbose=true &

PID=$!
wait $PID

FINAL=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
echo "[${CANDIDATE}] Generated ${FINAL}/${NUM} volumes"
[ "$FINAL" -ge "$NUM" ] && echo "Generation complete for ${CANDIDATE}." && exit 0

conda deactivate
