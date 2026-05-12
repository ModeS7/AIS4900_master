#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="irsde_anatomy"
#SBATCH --output=IDUN/output/eval/eval_irsde_anatomy_%j.out
#SBATCH --error=IDUN/output/eval/eval_irsde_anatomy_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# R7 — IR-SDE anatomy quantification.
# Quantifies the "right statistics, wrong content" claim for exp43_irsde:
#   - brain-mask Dice (x > 0.02) against the matched real volume
#   - cortex-slab L1 (top 50 axial slices)
#   - intensity-band per-region L1 (ventricle / parenchyma / cortex)
# Atlas: data/brain_atlas_256x256x160.nii.gz (binary brain mask; intensity
# bands carve up the inside-brain content for the per-region partition).
#
# Strategy: prefer the existing refined volumes from a prior compare_refinement
# run; if absent (the v2 SLURM deletes ${OUT_ROOT}/refined/ after eval), the
# IR-SDE refinement is regenerated on synth_rt. This stays well under the 4h
# budget — IR-SDE 25-step inference on 525 volumes runs in ~30-60 min on H100.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== R7 IR-SDE anatomy quantification ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

NUM_VOLS=525
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
REAL_DIR="${DATA_ROOT}/test_new"
RAW_SYNTH="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
ATLAS="${SLURM_SUBMIT_DIR}/data/brain_atlas_256x256x160.nii.gz"

OUT_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/eval/irsde_anatomy_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT_ROOT}"

# Locate IR-SDE refined volumes. Prefer the most recent v2 run (override with
# IRSDE_REFINED env var if needed).
if [ -z "${IRSDE_REFINED:-}" ]; then
    _LATEST_V2=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/eval/compare_refinement_v2_* 2>/dev/null | head -1)
    if [ -n "${_LATEST_V2}" ] && [ -d "${_LATEST_V2}/refined/exp43_irsde" ]; then
        IRSDE_REFINED="${_LATEST_V2}/refined/exp43_irsde"
    fi
fi

if [ -n "${IRSDE_REFINED:-}" ] && [ -d "${IRSDE_REFINED}" ]; then
    echo "Reusing existing IR-SDE refined volumes: ${IRSDE_REFINED}"
else
    echo "No existing IR-SDE refined volumes found — regenerating."
    SYNTH_RT_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/eval/fid_compression_eq_* 2>/dev/null | head -1)/volumes/synth_rt
    [ ! -d "${SYNTH_RT_DIR}" ] && { echo "ERROR: synth_rt not found at ${SYNTH_RT_DIR}"; exit 1; }

    CKPT_IRSDE_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/restoration/exp43_irsde_vqvae_deblur_* 2>/dev/null | head -1)
    if [ -f "${CKPT_IRSDE_DIR}/checkpoint_best.pt" ]; then
        CKPT_IRSDE="${CKPT_IRSDE_DIR}/checkpoint_best.pt"
    else
        CKPT_IRSDE="${CKPT_IRSDE_DIR}/checkpoint_latest.pt"
    fi
    [ ! -f "${CKPT_IRSDE}" ] && { echo "ERROR: IR-SDE checkpoint missing: ${CKPT_IRSDE}"; exit 1; }

    IRSDE_REFINED="${OUT_ROOT}/refined/exp43_irsde"
    mkdir -p "${IRSDE_REFINED}"
    echo "Generating IR-SDE refined volumes from ${SYNTH_RT_DIR}"
    echo "  ckpt: ${CKPT_IRSDE}"
    time python -m medgen.scripts.restore_volumes \
        --restoration-model "${CKPT_IRSDE}" --strategy irsde \
        --input-dir "${SYNTH_RT_DIR}" --output-dir "${IRSDE_REFINED}" \
        --num-steps 25 --max-volumes ${NUM_VOLS} --depth 160 --image-size 256
fi

EVAL_DIR="${OUT_ROOT}/comparison"
mkdir -p "${EVAL_DIR}"

echo ""
echo "=== Running eval_blur_attack_compare with --per-region-l1 ==="
# raw_synth included as the apples-to-apples baseline ("unrefined") so the
# anatomy delta from IR-SDE is interpretable.
time python -m medgen.scripts.eval_blur_attack_compare \
    --methods \
        "raw_synth:${RAW_SYNTH}" \
        "exp43_irsde:${IRSDE_REFINED}" \
    --real-dir "${REAL_DIR}" \
    --output-dir "${EVAL_DIR}" \
    --num-volumes ${NUM_VOLS} \
    --num-real 51 \
    --num-real-lpips 10 \
    --diversity-pairs 100 \
    --grid-subjects 5 \
    --per-region-l1 \
    --brain-threshold 0.02 \
    --cortex-slices 50 \
    --atlas "${ATLAS}"

echo ""
echo "=== Done ==="
echo "Results at: ${EVAL_DIR}"
echo ""
echo "--- results.csv ---"
cat "${EVAL_DIR}/results.csv" 2>/dev/null
