#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="irsde_anat_o4"
#SBATCH --output=IDUN/output/eval/eval_irsde_anatomy_option4_%j.out
#SBATCH --error=IDUN/output/eval/eval_irsde_anatomy_option4_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# R7 — Option 4: matched-pair anatomy on raw_synth ↔ exp43_irsde (filename-paired).
# Re-uses the 525 IR-SDE refined volumes produced by job 24453825
# (runs/eval/irsde_anatomy_20260512-224255/refined/exp43_irsde/) — NO refinement rerun.
# Each generated synth file has the same name in raw_synth and exp43_irsde dirs, so the
# new --anatomy-pair-with raw_synth matches them by filename, giving 525 paired
# anatomy comparisons that directly test "did IR-SDE preserve the input brain envelope".

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
cd ${SLURM_SUBMIT_DIR}

echo "=== R7 Option 4: IR-SDE anatomy (matched-pair vs raw_synth input) ==="
nvidia-smi
module purge && module load Anaconda3/2024.02-1 && conda activate AIS4900
export PYTORCH_ALLOC_CONF=expandable_segments:True

NUM_VOLS=525
DATA_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
REAL_DIR="${DATA_ROOT}/test_new"
RAW_SYNTH="${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525"
ATLAS="${SLURM_SUBMIT_DIR}/data/brain_atlas_256x256x160.nii.gz"

# Reuse the IR-SDE volumes from the prior run that hit walltime mid-eval.
IRSDE_REFINED="${CLUSTER_BASE}/AIS4900_master/runs/eval/irsde_anatomy_20260512-224255/refined/exp43_irsde"
if [ ! -d "${IRSDE_REFINED}" ]; then
    echo "ERROR: expected IR-SDE refined volumes at ${IRSDE_REFINED}"
    echo "Override with IRSDE_REFINED=<path> if they were moved."
    exit 1
fi
N_REFINED=$(find "${IRSDE_REFINED}" -name 'bravo.nii.gz' -o -name '*.nii.gz' 2>/dev/null | wc -l)
echo "Reusing ${N_REFINED} IR-SDE refined volumes from ${IRSDE_REFINED}"

OUT_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/eval/irsde_anatomy_option4_$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT_ROOT}"

echo ""
echo "=== Running eval_blur_attack_compare with --anatomy-pair-with raw_synth ==="
# raw_synth listed first so it's the matched-pair source for exp43_irsde.
# --per-region-l1 + --anatomy-pair-with raw_synth matches by filename within each
# method dir (both have gen_NNNN.nii.gz layouts). 525 paired anatomy comparisons.
time python -m medgen.scripts.eval_blur_attack_compare \
    --methods \
        "raw_synth:${RAW_SYNTH}" \
        "exp43_irsde:${IRSDE_REFINED}" \
    --real-dir "${REAL_DIR}" \
    --output-dir "${OUT_ROOT}" \
    --num-volumes ${NUM_VOLS} \
    --num-real 51 \
    --num-real-lpips 10 \
    --diversity-pairs 100 \
    --grid-subjects 5 \
    --per-region-l1 \
    --anatomy-pair-with raw_synth \
    --brain-threshold 0.02 \
    --cortex-slices 50 \
    --atlas "${ATLAS}"

echo ""
echo "=== Done ==="
echo "Results at: ${OUT_ROOT}"
echo ""
echo "--- results.csv ---"
cat "${OUT_ROOT}/results.csv" 2>/dev/null
