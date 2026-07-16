#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="eval_exp47d"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/eval_exp47d_%j.out
#SBATCH --error=IDUN/output/generate/eval_exp47d_%j.err
#SBATCH --signal=B:SIGUSR1@120

# ============================================================================
# Source-selection eval for exp47d (leak-free redo).
# 1) Generate 105 BRAVO volumes conditioned on the 105 REAL TRAIN masks,
#    fixed 100 steps, NO filters, bravo-only -> MedicalDataSets/evalModels/exp47d/
# 2) Compute the full metric panel vs TRAIN-105 ONLY: ImageNet + RadImageNet FID/KID,
#    CMMD (BiomedCLIP), Med3D-MMD (true-3D), intra-set MS-SSIM diversity.
# Standalone (full t-range, single model).
# NOTE: Med3D + BiomedCLIP weights auto-download from HF; pre-cache on the LOGIN node.
# ============================================================================

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

MODEL="exp47d"
GLOB="exp47d_compression_recipe_*"
HIGH_T_GLOB=""
NUM=105
STEPS=100
HANDOFF_T=0.25

BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
TRAIN_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"    # 105 real train (*/seg.nii.gz + bravo.nii.gz)
OUT_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels"
GEN_DIR="${OUT_ROOT}/${MODEL}"
RESULT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/eval_models_metrics"
mkdir -p "${GEN_DIR}" "${RESULT_DIR}"

# Resolve low-t checkpoint (latest, fall back to best).
RUN_DIR=$(ls -td ${BRAVO_ROOT}/${GLOB} 2>/dev/null | head -1)
CKPT="${RUN_DIR}/checkpoint_latest.pt"; [ ! -f "${CKPT}" ] && CKPT="${RUN_DIR}/checkpoint_best.pt"
if [ ! -f "${CKPT}" ]; then echo "FATAL: no ckpt for ${MODEL} (glob=${GLOB} under ${BRAVO_ROOT})"; exit 3; fi

# Optional high-t handoff partner (exp48 curriculum models only).
HANDOFF_ARGS=""
if [ -n "${HIGH_T_GLOB}" ]; then
    HIGH_RUN_DIR=$(ls -td ${BRAVO_ROOT}/${HIGH_T_GLOB} 2>/dev/null | head -1)
    HIGH_CKPT="${HIGH_RUN_DIR}/checkpoint_latest.pt"; [ ! -f "${HIGH_CKPT}" ] && HIGH_CKPT="${HIGH_RUN_DIR}/checkpoint_best.pt"
    if [ ! -f "${HIGH_CKPT}" ]; then echo "FATAL: no high-t ckpt (glob=${HIGH_T_GLOB})"; exit 3; fi
    HANDOFF_ARGS="+image_model_high_t=${HIGH_CKPT} handoff_t=${HANDOFF_T}"
fi

# Chain/resume (105 vols usually fit one job; kept for safety).
CHAIN_MAX=10
CHAIN_COUNT=${CHAIN_COUNT:-0}
CURRENT=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)

handle_timeout() {
    echo "=== Wall time approaching (SIGUSR1). Saving and resubmitting... ==="
    kill -USR1 $PID 2>/dev/null; wait $PID 2>/dev/null
    DONE=$(find "${GEN_DIR}" -maxdepth 1 -type d -name "[0-9]*" | wc -l)
    if [ "${DONE}" -ge "${NUM}" ]; then echo "Generation complete (${DONE}/${NUM})."; exit 0; fi
    NEXT=$((CHAIN_COUNT + 1))
    if [ "${NEXT}" -ge "${CHAIN_MAX}" ]; then echo "Chain limit reached."; exit 1; fi
    sbatch --export=ALL,CHAIN_COUNT=${NEXT} "$0"
    echo "Resubmitted chain ${NEXT}/${CHAIN_MAX}"; exit 0
}
trap handle_timeout SIGUSR1

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

echo "=== [${MODEL}] STEP 1/2 : generate 105 BRAVO @ ${STEPS} steps, real train masks, NO filters ==="
echo "  low-t  : ${CKPT}"
[ -n "${HIGH_T_GLOB}" ] && echo "  high-t : ${HIGH_CKPT}  (handoff_t=${HANDOFF_T})"
echo "  masks  : ${TRAIN_DIR}"
echo "  out    : ${GEN_DIR}  (resume from ${CURRENT}/${NUM})"

time python -m medgen.scripts.generate \
    paths=cluster \
    spatial_dims=3 \
    gen_mode=bravo \
    image_size=256 \
    depth=160 \
    image_model="${CKPT}" \
    ${HANDOFF_ARGS} \
    real_seg_dir="${TRAIN_DIR}" \
    num_images=${NUM} \
    current_image=${CURRENT} \
    num_steps_bravo=${STEPS} \
    validate_size_bins=false \
    validate_brain_mask=false \
    brain_atlas_path=null \
    brain_pca_path=null \
    paths.generated_dir="${OUT_ROOT}" \
    output_subdir="${MODEL}" \
    verbose=true &
PID=$!
wait $PID

# Bravo-only: drop any seg.nii.gz the pipeline copied in.
find "${GEN_DIR}" -name "seg.nii.gz" -delete 2>/dev/null

DONE=$(find "${GEN_DIR}" -name "bravo.nii.gz" | wc -l)
echo "Generated ${DONE}/${NUM} bravo volumes in ${GEN_DIR}."
if [ "${DONE}" -lt "${NUM}" ]; then
    echo "Generation incomplete (${DONE}/${NUM}) — skipping metrics; a resubmit will finish it."
    exit 0
fi

echo "=== [${MODEL}] STEP 2/2 : metric panel vs TRAIN-105 (FID/KID ImageNet+RadImageNet, CMMD, Med3D-MMD, diversity) ==="
time python -m medgen.scripts.eval_all_metrics \
    --reference train:"${TRAIN_DIR}" \
    --dataset ${MODEL}:"${GEN_DIR}" \
    --pattern bravo.nii.gz \
    --pool-cap 0 \
    --pca-model none \
    --output "${RESULT_DIR}/${MODEL}.json"

echo "Done [${MODEL}]: panel -> ${RESULT_DIR}/${MODEL}.json"
