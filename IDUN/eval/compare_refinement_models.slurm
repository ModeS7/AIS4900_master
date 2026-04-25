#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="cmp_ref"
#SBATCH --output=IDUN/output/eval/compare_refinement_%j.out
#SBATCH --error=IDUN/output/eval/compare_refinement_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Quick comparison of all refinement experiments on 10 synth volumes:
#   1. baseline       — raw synth (no refinement)
#   2. synth_rt       — VQ-VAE roundtrip alone (no refinement)
#   3. exp42b         — Pix2Pix GAN (input: raw synth)
#   4. exp43_pix2pix  — Pix2Pix on VQ-VAE pairs (input: synth_rt)
#   5. exp43_irsde    — IR-SDE on VQ-VAE pairs (input: synth_rt)
#   6. exp44          — Pix2Pix + residual + FM (input: raw synth)
# Then runs eval_blur_attack_compare.py to produce CSV + bar chart + 5×6 grid.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Comparing refinement models on 10 synth volumes ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_ALLOC_CONF=expandable_segments:True

NUM_VOLS=525    # Use full set (525-cap; will use however many are available)
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
REAL_DIR="${DATA_ROOT}/test_new"
RAW_SYNTH="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"

# synth_rt comes from the most recent fid_compression_eq run that saved them
SYNTH_RT_DIR=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/eval/fid_compression_eq_* 2>/dev/null | head -1)/volumes/synth_rt
if [ ! -d "${SYNTH_RT_DIR}" ]; then
    echo "ERROR: synth_rt volumes not found at ${SYNTH_RT_DIR}"
    echo "Run fid_compression_equalize.slurm first."
    exit 1
fi
echo "synth_rt source: ${SYNTH_RT_DIR}"

# Output root
OUT_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/eval/compare_refinement_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT_ROOT}"

# Auto-detect each experiment's checkpoint dir
CKPT_42B=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/refinement/exp42b_pix2pix_refinement_* 2>/dev/null | head -1)
CKPT_43=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/refinement/exp43_vqvae_deblur_* 2>/dev/null | head -1)
CKPT_43_IRSDE=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/restoration/exp43_irsde_vqvae_deblur_* 2>/dev/null | head -1)
CKPT_44=$(ls -td ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/refinement/exp44_pix2pix_residual_fm_* 2>/dev/null | head -1)

pick_ckpt () {
    local d="$1"
    if [ -f "${d}/checkpoint_best.pt" ]; then echo "${d}/checkpoint_best.pt"
    elif [ -f "${d}/checkpoint_latest.pt" ]; then echo "${d}/checkpoint_latest.pt"
    else echo ""
    fi
}

CKPT_42B_PT=$(pick_ckpt "${CKPT_42B}")
CKPT_43_PT=$(pick_ckpt "${CKPT_43}")
CKPT_43_IRSDE_PT=$(pick_ckpt "${CKPT_43_IRSDE}")
CKPT_44_PT=$(pick_ckpt "${CKPT_44}")

for ck in "${CKPT_42B_PT}" "${CKPT_43_PT}" "${CKPT_43_IRSDE_PT}" "${CKPT_44_PT}"; do
    if [ -z "${ck}" ]; then
        echo "ERROR: missing checkpoint — found: 42b=${CKPT_42B_PT} 43=${CKPT_43_PT} 43_irsde=${CKPT_43_IRSDE_PT} 44=${CKPT_44_PT}"
        exit 1
    fi
done

echo ""
echo "Checkpoints:"
echo "  exp42b:        ${CKPT_42B_PT}"
echo "  exp43_pix2pix: ${CKPT_43_PT}"
echo "  exp43_irsde:   ${CKPT_43_IRSDE_PT}"
echo "  exp44:         ${CKPT_44_PT}"
echo ""

# ── Run inference for each model on the same 10 volumes ───────────
DIR_42B="${OUT_ROOT}/refined/exp42b"
DIR_43="${OUT_ROOT}/refined/exp43_pix2pix"
DIR_43_IRSDE="${OUT_ROOT}/refined/exp43_irsde"
DIR_44="${OUT_ROOT}/refined/exp44"
mkdir -p "${DIR_42B}" "${DIR_43}" "${DIR_43_IRSDE}" "${DIR_44}"

echo "=== [1/4] exp42b (Pix2Pix GAN, input=raw synth) ==="
time python -m medgen.scripts.restore_volumes \
    --restoration-model "${CKPT_42B_PT}" \
    --strategy refinement \
    --input-dir "${RAW_SYNTH}" \
    --output-dir "${DIR_42B}" \
    --max-volumes ${NUM_VOLS} \
    --depth 160 --image-size 256

echo ""
echo "=== [2/4] exp43_pix2pix (Pix2Pix on VQ-VAE pairs, input=synth_rt) ==="
time python -m medgen.scripts.restore_volumes \
    --restoration-model "${CKPT_43_PT}" \
    --strategy refinement \
    --input-dir "${SYNTH_RT_DIR}" \
    --output-dir "${DIR_43}" \
    --max-volumes ${NUM_VOLS} \
    --depth 160 --image-size 256

echo ""
echo "=== [3/4] exp43_irsde (IR-SDE on VQ-VAE pairs, input=synth_rt, 25 steps) ==="
time python -m medgen.scripts.restore_volumes \
    --restoration-model "${CKPT_43_IRSDE_PT}" \
    --strategy irsde \
    --input-dir "${SYNTH_RT_DIR}" \
    --output-dir "${DIR_43_IRSDE}" \
    --num-steps 25 \
    --max-volumes ${NUM_VOLS} \
    --depth 160 --image-size 256

echo ""
echo "=== [4/4] exp44 (Pix2Pix + residual + FM, input=raw synth) ==="
# exp44 was trained with residual learning: G predicts (clean - degraded).
# At inference we must add the input back: refined = degraded + G(degraded).
time python -m medgen.scripts.restore_volumes \
    --restoration-model "${CKPT_44_PT}" \
    --strategy refinement \
    --use-residual \
    --input-dir "${RAW_SYNTH}" \
    --output-dir "${DIR_44}" \
    --max-volumes ${NUM_VOLS} \
    --depth 160 --image-size 256

# ── Diagnostic: per-method volume stats (range, mean, NaN check) ──
echo ""
echo "=== Volume diagnostics ==="
DIAG_ROOT="${OUT_ROOT}/refined" python3 <<'EOF'
import os
from pathlib import Path
import nibabel as nib, numpy as np
root = Path(os.environ['DIAG_ROOT'])
for d in sorted(root.iterdir()):
    files = list(d.glob('**/bravo.nii.gz'))[:3]
    if not files: continue
    n_total = len(list(d.glob('**/bravo.nii.gz')))
    print(f"\n{d.name}: {n_total} volumes")
    for f in files:
        v = nib.load(str(f)).get_fdata()
        nan_pct = 100 * np.isnan(v).sum() / v.size
        print(f"  {f.parent.name}: shape={v.shape} min={v.min():.3f} max={v.max():.3f} "
              f"mean={v.mean():.3f} std={v.std():.3f} nan%={nan_pct:.1f}")
EOF

# ── Final 6-way comparison ────────────────────────────────────────
echo ""
echo "=== Running eval_blur_attack_compare.py on 6 methods ==="
EVAL_DIR="${OUT_ROOT}/comparison"
mkdir -p "${EVAL_DIR}"

time python -m medgen.scripts.eval_blur_attack_compare \
    --methods \
        "raw_synth:${RAW_SYNTH}" \
        "synth_rt:${SYNTH_RT_DIR}" \
        "exp42b:${DIR_42B}" \
        "exp43_pix2pix:${DIR_43}" \
        "exp43_irsde:${DIR_43_IRSDE}" \
        "exp44:${DIR_44}" \
    --real-dir "${REAL_DIR}" \
    --output-dir "${EVAL_DIR}" \
    --num-volumes ${NUM_VOLS} \
    --num-real 26 \
    --num-real-lpips 10 \
    --diversity-pairs 100 \
    --grid-subjects 5

# ── Cleanup: delete intermediate refined volumes (40MB × 525 × 4 ≈ 80GB) ──
echo ""
echo "=== Cleanup: deleting intermediate refined volumes ==="
du -sh "${OUT_ROOT}/refined" 2>/dev/null || true
rm -rf "${OUT_ROOT}/refined"
echo "Cleanup done. Only comparison/ outputs retained."

echo ""
echo "=== Done ==="
echo "All outputs at: ${OUT_ROOT}"
echo ""
echo "--- Comparison summary ---"
cat "${EVAL_DIR}/results.csv" 2>/dev/null
echo ""
echo "Visual grid: ${EVAL_DIR}/visual_grid.png"
echo "Spectrum overlay: ${EVAL_DIR}/spectrum_overlay.png"
echo "Bar chart: ${EVAL_DIR}/bar_chart.png"
