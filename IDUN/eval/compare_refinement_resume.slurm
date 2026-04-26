#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="cmp_ref_rs"
#SBATCH --output=IDUN/output/eval/compare_refinement_resume_%j.out
#SBATCH --error=IDUN/output/eval/compare_refinement_resume_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Resume of compare_refinement_models.slurm: the inference phase finished
# (~3.5h) but the eval phase ran out of wall time at exp43_pix2pix. The 4
# refined volume directories are already written to disk under OUT_ROOT/refined.
# This job ONLY re-runs eval_blur_attack_compare on the existing dirs, then
# cleans up the intermediate refined volumes.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

cd ${SLURM_SUBMIT_DIR}

echo "=== Resuming compare_refinement: eval-only phase ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

export PYTORCH_ALLOC_CONF=expandable_segments:True

NUM_VOLS=525

# Point at the existing run from the timed-out job:
OUT_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/eval/compare_refinement_20260426-021542"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/test_new"
RAW_SYNTH="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
SYNTH_RT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/fid_compression_eq_20260424-160608/volumes/synth_rt"

DIR_42B="${OUT_ROOT}/refined/exp42b"
DIR_43="${OUT_ROOT}/refined/exp43_pix2pix"
DIR_43_IRSDE="${OUT_ROOT}/refined/exp43_irsde"
DIR_44="${OUT_ROOT}/refined/exp44"

echo "Verifying all refined dirs exist..."
for d in "${DIR_42B}" "${DIR_43}" "${DIR_43_IRSDE}" "${DIR_44}" "${SYNTH_RT_DIR}" "${RAW_SYNTH}" "${REAL_DIR}"; do
    if [ ! -d "${d}" ]; then
        echo "ERROR: missing directory ${d}"
        exit 1
    fi
    n=$(find "${d}" -maxdepth 2 -name 'bravo.nii.gz' | wc -l)
    echo "  ${d}: ${n} volumes"
done

EVAL_DIR="${OUT_ROOT}/comparison"
mkdir -p "${EVAL_DIR}"

echo ""
echo "=== Running eval_blur_attack_compare.py on 6 methods ==="
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
echo "Visual grid:      ${EVAL_DIR}/visual_grid.png"
echo "Spectrum overlay: ${EVAL_DIR}/spectrum_overlay.png"
echo "Bar chart:        ${EVAL_DIR}/bar_chart.png"
