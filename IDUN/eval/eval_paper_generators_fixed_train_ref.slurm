#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="paper_trref"
#SBATCH --output=IDUN/output/eval/eval_paper_generators_fixed_train_ref_%j.out
#SBATCH --error=IDUN/output/eval/eval_paper_generators_fixed_train_ref_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Fixed-configuration train-reference sensitivity panel for the paper.
#
# Phase 1 uses find_optimal_steps.py with a ONE-VALUE step grid. Each entry:
#   - loads an existing checkpoint/configuration at its historical selected step count;
#   - uses the first 25 TRAIN segmentation masks as conditioning;
#   - generates the same 25 seed-42 noise realizations;
#   - scores ImageNet/RadImageNet FID+KID, CMMD, and PCA vs TRAIN;
#   - saves the 25 generated volumes for whole-volume Med3D evaluation.
#
# Phase 2 evaluates all saved sets with the established native-resolution MedicalNet
# ResNet-50 Gaussian-MMD implementation against the 105-volume TRAIN reference.
# This is a sensitivity analysis only: it does not choose new optima or create new
# downstream datasets.

set -euo pipefail

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd "${SLURM_SUBMIT_DIR}"

echo "=== Paper generator fixed-step train-reference panel ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
export PYTORCH_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
RUN_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/eval_paper_generators_fixed_train_ref"

# Exact checkpoints recovered from the completed candidate-generation logs.
CKPT_EXP1="${RUN_ROOT}/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
CKPT_EXP1PLUS="${RUN_ROOT}/exp1_1_1000plus_pixel_bravo_20260411-235425/checkpoint_latest.pt"
CKPT_EXP32="${RUN_ROOT}/exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027/checkpoint_latest.pt"
CKPT_EXP48C="${RUN_ROOT}/exp48c_lowt_only_lpips_huber_20260425-162615/checkpoint_latest.pt"
CKPT_EXP48D="${RUN_ROOT}/exp48d_lowt_only_huber_lpips_20260425-172756/checkpoint_latest.pt"

for ckpt in \
    "${CKPT_EXP1}" \
    "${CKPT_EXP1PLUS}" \
    "${CKPT_EXP32}" \
    "${CKPT_EXP48C}" \
    "${CKPT_EXP48D}"; do
    [ -f "${ckpt}" ] || { echo "ERROR: checkpoint not found: ${ckpt}"; exit 1; }
done

mkdir -p "${PANEL_ROOT}"

METRICS="fid,kid,cmmd,fid_radimagenet,kid_radimagenet,pca"
REF_CACHE=""
MED3D_DATASETS=()

prepare_output() {
    local label="$1"
    local steps="$2"
    CURRENT_OUT="${PANEL_ROOT}/${label}"
    CURRENT_VOLS="${CURRENT_OUT}/generated/euler_steps$(printf '%03d' "${steps}")"
    mkdir -p "${CURRENT_OUT}"

    # All configurations use identical real reference features. Reuse the cache
    # built by the first entry instead of extracting 105 train volumes repeatedly.
    if [ -n "${REF_CACHE}" ] && [ ! -e "${CURRENT_OUT}/reference_features" ]; then
        ln -s "${REF_CACHE}" "${CURRENT_OUT}/reference_features"
    fi
}

finish_output() {
    local label="$1"
    if [ -z "${REF_CACHE}" ]; then
        REF_CACHE="${CURRENT_OUT}/reference_features"
    fi
    MED3D_DATASETS+=(--dataset "${label}:${CURRENT_VOLS}")
}

run_standalone() {
    local label="$1"
    local checkpoint="$2"
    local steps="$3"
    prepare_output "${label}" "${steps}"

    echo ""
    echo "=== ${label}: standalone, ${steps} Euler steps ==="
    time python -m medgen.scripts.find_optimal_steps \
        --checkpoint "${checkpoint}" \
        --output-dir "${CURRENT_OUT}" \
        --metric "${METRICS}" \
        --data-root "${DATA_ROOT}" \
        --image-size 256 \
        --depth 160 \
        --num-volumes 25 \
        --step-grid "${steps}" \
        --num-seeds 1 \
        --cond-split train \
        --ref-split train \
        --seed 42 \
        --save-volumes

    finish_output "${label}"
}

run_handoff() {
    local label="$1"
    local low_checkpoint="$2"
    local high_checkpoint="$3"
    local steps="$4"
    prepare_output "${label}" "${steps}"

    echo ""
    echo "=== ${label}: handoff t=0.25, ${steps} Euler steps ==="
    time python -m medgen.scripts.find_optimal_steps \
        --high-t-checkpoint "${high_checkpoint}" \
        --low-t-checkpoint "${low_checkpoint}" \
        --handoff-t 0.25 \
        --output-dir "${CURRENT_OUT}" \
        --metric "${METRICS}" \
        --data-root "${DATA_ROOT}" \
        --image-size 256 \
        --depth 160 \
        --num-volumes 25 \
        --step-grid "${steps}" \
        --num-seeds 1 \
        --cond-split train \
        --ref-split train \
        --seed 42 \
        --save-volumes

    finish_output "${label}"
}

# exp1_1_1000 is the only model intentionally compared at both its ImageNet-
# and RadImageNet-selected settings.
run_standalone "exp1_1_1000_img_s032" "${CKPT_EXP1}" 32
run_standalone "exp1_1_1000_rin_s079" "${CKPT_EXP1}" 79

# Remaining entries reproduce configurations actually used to generate the
# segmentation-candidate datasets. They are evaluated only at their deployed
# RadImageNet-selected setting; no secondary ImageNet minima are introduced.
run_standalone "exp1_1_1000plus_rin_s079" "${CKPT_EXP1PLUS}" 79
run_standalone "exp32_2_1000_rin_s100" "${CKPT_EXP32}" 100
run_standalone "exp48c_standalone_rin_s100" "${CKPT_EXP48C}" 100
run_handoff "exp48c_handoff_exp32_rin_s100" "${CKPT_EXP48C}" "${CKPT_EXP32}" 100
run_handoff "exp48d_handoff_exp1_rin_s100" "${CKPT_EXP48D}" "${CKPT_EXP1}" 100

echo ""
echo "=== Med3D-MMD: MedicalNet ResNet-50 vs train reference ==="
time python -m medgen.scripts.eval_med3d_mmd_networks \
    --reference "train:${DATA_ROOT}/train" \
    "${MED3D_DATASETS[@]}" \
    --networks 50 \
    --pattern '*_bravo.nii.gz' \
    --target-shape 150 256 256 \
    --pool-cap 0 \
    --subsample-n 25 \
    --bootstrap 0 \
    --batch-size 2 \
    --seed 42 \
    --output "${PANEL_ROOT}/med3d_mmd_r50_train_ref.json"

echo ""
echo "=== Completed paper generator train-reference panel ==="
echo "Per-configuration metrics: ${PANEL_ROOT}/<label>/search_results.json"
echo "Med3D-R50 MMD panel:       ${PANEL_ROOT}/med3d_mmd_r50_train_ref.json"
