#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="gen_qual_testmask"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/gen_qual_testmask_%j.out
#SBATCH --error=IDUN/output/generate/gen_qual_testmask_%j.err
#
# Qualitative-figure volumes for the seven downstream sources, conditioned on REAL
# **TEST** segmentation masks.
#
# WHY TEST MASKS: the existing qualitative material is conditioned on the 105 real
# *training* masks, so a reader cannot tell realism from memorisation — the generators
# were fitted on those (image, mask) pairs. The 51 official test cases were never seen
# by any generator, so conditioning on them shows generalisation, and the paired real
# test volume gives a legitimate reference column.
#
# Sampling matches the paper exactly: 100 Euler updates (dt = 0.01), seed 42, handoff at
# t = 0.25 for the two handoff routes, same brain-support PCA containment (2.323 mm) and
# the same 3-attempt cap. The ONLY change from the paper's generation procedure is the
# conditioning mask source (real test instead of the frozen synthetic bank).
#
# Output: $CLUSTER_BASE/MedicalDataSets/evalModels/qualitative_testmask/<route>/<idx>/
# Copy back per route: <route>/00000/{bravo.nii.gz,seg.nii.gz}
set -Eeuo pipefail

cd "${SLURM_SUBMIT_DIR}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly REPO_ROOT="${SLURM_SUBMIT_DIR}"
readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly TEST_SEGS="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/test1"
readonly OUT_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/qualitative_testmask"
readonly BRAIN_SUPPORT_PCA="${REPO_ROOT}/data/brain_support_pca_train105_var95_256x256x160.npz"
readonly SEED=42
readonly STEPS=100
readonly BRAIN_MARGIN_MM=2.323
readonly MAX_IMAGE_ATTEMPTS=3

# WHICH TEST CASES. Chosen deliberately, not by directory order: all 51 test cases were
# screened locally on lesion size, local contrast on post-contrast T1, and compactness.
#   Mets_098 - 2501 vox, contrast +0.201, compactness 0.84. Clean ring-enhancing lesion with a
#              necrotic centre, well inside the frontal lobe and obvious at whole-slice scale.
#              PRIMARY.
#   Mets_041 - 6643 vox, contrast +0.240. Larger, even more striking ring, but sits inferiorly
#              near the brain edge so the slice reads less like a classic view. BACKUP.
#   Mets_149 - 948 vox, highest contrast (+0.265) but the lesion abuts the skull base. SPARE.
# Taking the first N cases in directory order would have given Mets_009 (178 vox, contrast
# +0.134) — a small, low-contrast lesion that would not showcase anything.
CASES=(Mets_098 Mets_041 Mets_149)

# route label | low-t (or single) run | high-t run for handoff ("-" if none)
ROUTES=(
  "exp1_1_1000|exp1_1_1000_pixel_bravo_20260402-121556|-"
  "exp1_1_1000plus|exp1_1_1000plus_pixel_bravo_20260411-235425|-"
  "exp32_2_1000|exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027|-"
  "exp47a|exp47a_lpips_strong_20260425-055252|-"
  "exp47c|exp47c_lpips_huber_builtin_20260425-223819|-"
  "exp1_to_exp48c_t025|exp48c_lowt_only_lpips_huber_20260425-162615|exp1_1_1000_pixel_bravo_20260402-121556"
  "exp1_to_exp48d_t025|exp48d_lowt_only_huber_lpips_20260425-172756|exp1_1_1000_pixel_bravo_20260402-121556"
)

echo "=== Qualitative figure volumes | REAL TEST masks | ${STEPS} Euler steps | seed ${SEED} ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
nvidia-smi
module purge
module load Anaconda3/2024.02-1
conda activate AIS4900
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA GPU: {torch.cuda.get_device_name(0)}')"

# Stage only the chosen cases so generation targets them exactly, regardless of the order
# generate.py walks a directory in.
readonly STAGE_SEGS="${OUT_ROOT}/_conditioning_cases"
mkdir -p "$STAGE_SEGS"
for c in "${CASES[@]}"; do
  [[ -d "${TEST_SEGS}/${c}" ]] || { echo "FATAL: test case not found: ${TEST_SEGS}/${c}"; exit 1; }
  rm -rf "${STAGE_SEGS:?}/${c}"
  cp -r "${TEST_SEGS}/${c}" "${STAGE_SEGS}/${c}"
done
echo "Staged conditioning cases: ${CASES[*]}"
readonly NUM_IMAGES="${#CASES[@]}"

# Preflight: fail before burning GPU time on a missing checkpoint.
[[ -d "$TEST_SEGS" ]]            || { echo "FATAL: test segs not found: $TEST_SEGS"; exit 1; }
[[ -s "$BRAIN_SUPPORT_PCA" ]]    || { echo "FATAL: brain-support PCA not found"; exit 1; }
for entry in "${ROUTES[@]}"; do
  IFS='|' read -r label low high <<<"$entry"
  [[ -f "${BRAVO_ROOT}/${low}/checkpoint_latest.pt" ]] || { echo "FATAL: missing $low"; exit 1; }
  if [[ "$high" != "-" ]]; then
    [[ -f "${BRAVO_ROOT}/${high}/checkpoint_latest.pt" ]] || { echo "FATAL: missing $high"; exit 1; }
  fi
done
echo "Preflight OK: ${#ROUTES[@]} routes x ${NUM_IMAGES} chosen test cases"

for entry in "${ROUTES[@]}"; do
  IFS='|' read -r label low high <<<"$entry"
  echo ""
  echo "=== ${label} ==="
  GEN_ARGS=(
    paths=cluster
    spatial_dims=3
    gen_mode=bravo
    image_size=256
    depth=160
    trim_slices=10
    fov_mm=240.0
    image_model="${BRAVO_ROOT}/${low}/checkpoint_latest.pt"
    image_model_high_t=null
    real_seg_dir="$STAGE_SEGS"
    expected_real_depth=150
    seed="$SEED"
    num_images="$NUM_IMAGES"
    current_image=0
    num_steps_bravo="$STEPS"
    ode_solver=euler
    shift_ratio_bravo=1.0
    cfg_scale_bravo=1.0
    validate_size_bins=false
    validate_real_seg_masks=true
    require_real_bravo_pairs=false
    validate_brain_mask=true
    conditioning_brain_qc_mode=reject
    brain_threshold=0.05
    brain_containment_margin_mm="$BRAIN_MARGIN_MM"
    max_image_attempts_per_mask="$MAX_IMAGE_ATTEMPTS"
    skip_failed_fixed_masks=true
    brain_atlas_path=null
    brain_support_pca_path="$BRAIN_SUPPORT_PCA"
    brain_pca_path=null
    seg_pca_path=null
    diffrs_checkpoint=null
    mask_outside_brain=false
    paths.generated_dir="$OUT_ROOT"
    output_subdir="$label"
    verbose=true
  )
  if [[ "$high" != "-" ]]; then
    GEN_ARGS+=(image_model_high_t="${BRAVO_ROOT}/${high}/checkpoint_latest.pt" handoff_t=0.25)
    echo "  handoff: high-t ${high} for t > 0.25 -> specialist ${low}"
  fi
  time python -m medgen.scripts.generate "${GEN_ARGS[@]}"
  echo "  produced: $(find "${OUT_ROOT}/${label}" -name 'bravo.nii.gz' 2>/dev/null | wc -l) volume(s)"
done

echo ""
echo "=== DONE. Copy back with: ==="
echo "  for r in exp1_1_1000 exp1_1_1000plus exp32_2_1000 exp47a exp47c \\"
echo "           exp1_to_exp48c_t025 exp1_to_exp48d_t025; do"
echo "    mkdir -p ~/NTNU/MedicalDataSets/evalModels/qual_test/\$r/00000"
echo "    scp <cluster>:${OUT_ROOT}/\$r/00000/{bravo,seg}.nii.gz \\"
echo "        ~/NTNU/MedicalDataSets/evalModels/qual_test/\$r/00000/"
echo "  done"
