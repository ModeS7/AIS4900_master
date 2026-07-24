#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_cv_exp17_d663"
#SBATCH --output=IDUN/output/train/downstream/nnunet/convert_exp17_whh210_d663_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/convert_exp17_whh210_d663_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Convert the frozen 210-pair Weighted Huber handoff source into Dataset663.
# The source is 151 canonical accepted pairs followed by 59 accepted extension
# pairs. Dataset663 contains 105 real and 210 synthetic training cases. It is
# never merged with or substituted for an existing nnU-Net dataset.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

require_marker_value() {
    local marker="$1"
    local key="$2"
    local expected="$3"
    local -a values=()

    mapfile -t values < <(
        awk -F= -v key="$key" \
            '$1 == key { print substr($0, index($0, "=") + 1) }' "$marker"
    )
    [[ ${#values[@]} -eq 1 ]] || fatal \
        "$marker must contain exactly one ${key}=... entry"
    [[ "${values[0]}" == "$expected" ]] || fatal \
        "$marker has ${key}=${values[0]}; expected ${key}=${expected}"
}

marker_value() {
    local marker="$1"
    local key="$2"
    local -a values=()

    mapfile -t values < <(
        awk -F= -v key="$key" \
            '$1 == key { print substr($0, index($0, "=") + 1) }' "$marker"
    )
    [[ ${#values[@]} -eq 1 && -n "${values[0]}" ]] || fatal \
        "$marker must contain exactly one nonempty ${key}=... entry"
    printf '%s\n' "${values[0]}"
}

readonly EXPERIMENT_FAMILY="exp17"
readonly GENERATOR="exp1_to_exp48c_t025"
readonly GENERATOR_DESCRIPTION="weighted_huber_handoff"
readonly DATASET_ID=663
readonly EXPECTED_REAL_CASES=105
readonly EXPECTED_SYNTHETIC_CASES=210
readonly EXPECTED_ORIGINAL_PAIRS=151
readonly EXPECTED_EXTENSION_PAIRS=59
readonly EXPECTED_TRAINING_CASES=315
readonly EXPECTED_TEST_CASES=51
readonly ORIGINAL_CANDIDATES=525
readonly EXTENSION_START=525
readonly EXTENSION_STOP=825
readonly EXPECTED_EXTENSION_MASKS=300
readonly BINS_HEADER='id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors'

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly MASK_PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/source_selection_train105_seed42_euler100"
readonly ORIGINAL_MASKS="${EVAL_ROOT}/ordered_common_seg_masks_seed42/candidates525"
readonly COMMON105_IDS="${EVAL_ROOT}/matched_common105/common_candidate_ids.txt"
readonly ORIGINAL_PAIRS="${EVAL_ROOT}/${GENERATOR}"
readonly ORIGINAL_MASK_MARKER="${ORIGINAL_PAIRS}/.brain_mask_stage3_applied"

readonly EXTENSION_ROOT="${EVAL_ROOT}/weighted_huber_handoff_extension_to210"
readonly NEW_MASK_DIR="${MASK_PANEL_ROOT}/common_seg_masks_extension_00525_00824"
readonly NEW_MASK_MARKER="${NEW_MASK_DIR}/.extension_complete"
readonly EXPANDED_MASKS="${EXTENSION_ROOT}/candidates825"
readonly EXPANDED_MASKS_MARKER="${EXPANDED_MASKS}/.extension_view_complete"
readonly NEW_PAIRS="${EXTENSION_ROOT}/exp1_to_exp48c_t025__extension_00525_00824"
readonly NEW_PAIRS_MARKER="${NEW_PAIRS}/.extension_complete"
readonly SYNTHETIC_VIEW="${EXTENSION_ROOT}/hybrid_pairs_210"
readonly SOURCE_ORDER="${EXTENSION_ROOT}/hybrid_candidate_ids_210.txt"
readonly SOURCE_COMPLETE="${SYNTHETIC_VIEW}/.hybrid_view_complete"
readonly GENERATION_LOCK="${EXTENSION_ROOT}/.generation.lock"

readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"
readonly SEG_CKPT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp14_1_pixel_seg_20260217-040309/checkpoint_latest.pt"
readonly LOW_CKPT="${BRAVO_ROOT}/exp48c_lowt_only_lpips_huber_20260425-162615/checkpoint_latest.pt"
readonly HIGH_CKPT="${BRAVO_ROOT}/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"

readonly REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

readonly RAW_DIR="${nnUNet_raw}/Dataset${DATASET_ID}_BrainMet"
readonly REFERENCE_RAW="${nnUNet_raw}/Dataset600_BrainMet"
readonly RAW_ORDER="${RAW_DIR}/synthetic_candidate_order.txt"
readonly SOURCE_SELECTION="${RAW_DIR}/source_selection.txt"
readonly CONVERSION_COMPLETE="${RAW_DIR}/.exp17_d663_conversion_complete"

[[ -f "$REPO_ROOT/pyproject.toml" ]] || fatal "repository root not found: $REPO_ROOT"
[[ -d "$REAL_DIR/train" && -d "$REAL_DIR/val" && -d "$REAL_DIR/test_new" ]] || \
    fatal "real train/val/test_new dataset is incomplete: $REAL_DIR"
[[ -d "$EXTENSION_ROOT" ]] || fatal "extension root not found: $EXTENSION_ROOT"
[[ -d "$ORIGINAL_MASKS" ]] || fatal "original 525-mask view not found: $ORIGINAL_MASKS"
[[ -s "$COMMON105_IDS" ]] || fatal "common-105 ID list not found: $COMMON105_IDS"
[[ -d "$ORIGINAL_PAIRS" ]] || fatal "canonical source not found: $ORIGINAL_PAIRS"
[[ -s "$ORIGINAL_MASK_MARKER" ]] || fatal "canonical source is not brain-masked"
[[ -d "$NEW_MASK_DIR" && -s "$NEW_MASK_MARKER" ]] || \
    fatal "completed 300-mask extension not found: $NEW_MASK_DIR"
[[ -d "$EXPANDED_MASKS" && -s "$EXPANDED_MASKS_MARKER" ]] || \
    fatal "completed 825-mask view not found: $EXPANDED_MASKS"
[[ -d "$NEW_PAIRS" && -s "$NEW_PAIRS_MARKER" ]] || \
    fatal "completed 59-pair extension not found: $NEW_PAIRS"
[[ -d "$SYNTHETIC_VIEW" && -s "$SOURCE_COMPLETE" ]] || \
    fatal "completed 210-pair source view not found: $SYNTHETIC_VIEW"
[[ -s "$SOURCE_ORDER" ]] || fatal "frozen 210-case order not found: $SOURCE_ORDER"
[[ -f "$SEG_CKPT" && -f "$LOW_CKPT" && -f "$HIGH_CKPT" ]] || \
    fatal "one or more generation checkpoints are missing"
[[ -s "$REFERENCE_RAW/case_info.json" ]] || \
    fatal "Dataset600 case provenance not found: $REFERENCE_RAW/case_info.json"

# The generation job and this conversion job share this lock. Conversion cannot
# inspect a source while it is being generated or rebuilt.
exec 9>"$GENERATION_LOCK"
flock -n 9 || fatal "the Weighted Huber handoff source is being generated or rebuilt"

require_marker_value "$ORIGINAL_MASK_MARKER" method medgen.scripts.brain_mask_existing
require_marker_value "$ORIGINAL_MASK_MARKER" mode in_place_atomic
require_marker_value "$ORIGINAL_MASK_MARKER" threshold 0.05
require_marker_value "$ORIGINAL_MASK_MARKER" fill_holes true
require_marker_value "$ORIGINAL_MASK_MARKER" dilate_pixels 2
require_marker_value "$ORIGINAL_MASK_MARKER" accepted_cases "$EXPECTED_ORIGINAL_PAIRS"

require_marker_value "$NEW_MASK_MARKER" global_start "$EXTENSION_START"
require_marker_value "$NEW_MASK_MARKER" global_stop "$EXTENSION_STOP"
require_marker_value "$NEW_MASK_MARKER" local_seed 567
require_marker_value "$NEW_MASK_MARKER" base_seed 42
require_marker_value "$NEW_MASK_MARKER" num_masks "$EXPECTED_EXTENSION_MASKS"

require_marker_value "$EXPANDED_MASKS_MARKER" original_candidates "$ORIGINAL_CANDIDATES"
require_marker_value "$EXPANDED_MASKS_MARKER" extension_start "$EXTENSION_START"
require_marker_value "$EXPANDED_MASKS_MARKER" extension_stop "$EXTENSION_STOP"
require_marker_value "$EXPANDED_MASKS_MARKER" total_candidates "$EXTENSION_STOP"

require_marker_value "$NEW_PAIRS_MARKER" source weighted_huber_handoff
require_marker_value "$NEW_PAIRS_MARKER" candidate_start "$EXTENSION_START"
require_marker_value "$NEW_PAIRS_MARKER" candidate_stop "$EXTENSION_STOP"
require_marker_value "$NEW_PAIRS_MARKER" accepted_pairs "$EXPECTED_EXTENSION_PAIRS"
require_marker_value "$NEW_PAIRS_MARKER" max_image_attempts 3
require_marker_value "$NEW_PAIRS_MARKER" brain_margin_mm 2.323
require_marker_value "$NEW_PAIRS_MARKER" brain_mask_threshold 0.05
require_marker_value "$NEW_PAIRS_MARKER" brain_mask_dilate_pixels 2

require_marker_value "$SOURCE_COMPLETE" target_pairs "$EXPECTED_SYNTHETIC_CASES"
require_marker_value "$SOURCE_COMPLETE" original_pairs "$EXPECTED_ORIGINAL_PAIRS"
require_marker_value "$SOURCE_COMPLETE" extension_pairs "$EXPECTED_EXTENSION_PAIRS"
require_marker_value "$SOURCE_COMPLETE" ordered_ids "$SOURCE_ORDER"

expected_seg_sha256="$(marker_value "$NEW_MASK_MARKER" seg_checkpoint_sha256)"
expected_low_sha256="$(marker_value "$NEW_PAIRS_MARKER" low_checkpoint_sha256)"
expected_high_sha256="$(marker_value "$NEW_PAIRS_MARKER" high_checkpoint_sha256)"
[[ "$expected_seg_sha256" =~ ^[0-9a-f]{64}$ ]] || fatal \
    "invalid segmentation checkpoint hash in $NEW_MASK_MARKER"
[[ "$expected_low_sha256" =~ ^[0-9a-f]{64}$ ]] || fatal \
    "invalid low-t checkpoint hash in $NEW_PAIRS_MARKER"
[[ "$expected_high_sha256" =~ ^[0-9a-f]{64}$ ]] || fatal \
    "invalid high-t checkpoint hash in $NEW_PAIRS_MARKER"
[[ "$(sha256sum "$SEG_CKPT" | awk '{print $1}')" == "$expected_seg_sha256" ]] || \
    fatal "segmentation checkpoint differs from the completed mask extension"
[[ "$(sha256sum "$LOW_CKPT" | awk '{print $1}')" == "$expected_low_sha256" ]] || \
    fatal "low-t checkpoint differs from the completed pair extension"
[[ "$(sha256sum "$HIGH_CKPT" | awk '{print $1}')" == "$expected_high_sha256" ]] || \
    fatal "high-t checkpoint differs from the completed pair extension"

# Dataset IDs are global in nnU-Net. Refuse any raw or preprocessed Dataset663,
# including an incompletely created one, so stale cases can never be mixed in.
for root in "$nnUNet_raw" "$nnUNet_preprocessed"; do
    if compgen -G "$root/Dataset${DATASET_ID}_*" > /dev/null; then
        fatal "Dataset${DATASET_ID} already exists under $root"
    fi
done

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet

# Validate the frozen nested order, all symlink mappings, and NIfTI geometry
# before the converter is allowed to create Dataset663.
python - \
    "$COMMON105_IDS" \
    "$ORIGINAL_MASKS" \
    "$NEW_MASK_DIR" \
    "$EXPANDED_MASKS" \
    "$ORIGINAL_PAIRS" \
    "$NEW_PAIRS" \
    "$SYNTHETIC_VIEW" \
    "$SOURCE_ORDER" \
    "$BINS_HEADER" <<'PYEOF'
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

(
    common_path,
    original_masks,
    new_masks,
    expanded_masks,
    original_pairs,
    new_pairs,
    synthetic_view,
    order_path,
    bins_header,
) = [Path(value) for value in sys.argv[1:9]] + [sys.argv[9]]

candidate_pattern = re.compile(r"[0-9]{5}")


def ids_in(root: Path) -> list[str]:
    return sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and candidate_pattern.fullmatch(path.name)
    )


def resolve_file(path: Path) -> Path:
    if not path.is_file():
        raise RuntimeError(f"required source file is missing: {path}")
    return path.resolve(strict=True)


common = common_path.read_text().splitlines()
original = ids_in(original_pairs)
extension = ids_in(new_pairs)
ordered = order_path.read_text().splitlines()

if len(common) != 105 or len(set(common)) != 105:
    raise RuntimeError("common-105 manifest must contain 105 unique IDs")
if any(candidate_pattern.fullmatch(value) is None for value in common):
    raise RuntimeError("common-105 manifest contains a malformed ID")
if len(original) != 151:
    raise RuntimeError(f"canonical source has {len(original)} pairs; expected 151")
if len(extension) != 59:
    raise RuntimeError(f"extension source has {len(extension)} pairs; expected 59")
if not set(common).issubset(original):
    raise RuntimeError("common-105 IDs are not a subset of the canonical source")
if any(not 525 <= int(value) < 825 for value in extension):
    raise RuntimeError("pair extension contains a candidate outside 00525--00824")

expected_order = common + [value for value in original if value not in set(common)] + extension
if ordered != expected_order:
    raise RuntimeError(
        "hybrid_candidate_ids_210.txt is not common105 + canonical remainder + extension"
    )
if len(ordered) != 210 or len(set(ordered)) != 210:
    raise RuntimeError("hybrid_candidate_ids_210.txt must contain 210 unique IDs")
if ids_in(synthetic_view) != sorted(ordered):
    raise RuntimeError("hybrid_pairs_210 does not exactly match its frozen manifest")

if ids_in(new_masks) != [f"{index:05d}" for index in range(300)]:
    raise RuntimeError("continuation mask tranche is not the complete local range 00000--00299")
if ids_in(expanded_masks) != [f"{index:05d}" for index in range(825)]:
    raise RuntimeError("expanded mask view is not the complete range 00000--00824")
for index in range(825):
    candidate_id = f"{index:05d}"
    linked = expanded_masks / candidate_id / "seg.nii.gz"
    if not linked.is_symlink():
        raise RuntimeError(f"expanded mask is not a symlink: {linked}")
    if index < 525:
        expected = original_masks / candidate_id / "seg.nii.gz"
    else:
        expected = new_masks / f"{index - 525:05d}" / "seg.nii.gz"
    if linked.resolve(strict=True) != resolve_file(expected):
        raise RuntimeError(f"expanded mask points to the wrong source: {linked}")

expected_bins = [bins_header] + [f"{value},0,0,0,0,0,0,0,0" for value in ordered]
if (synthetic_view / "bins.csv").read_text().splitlines() != expected_bins:
    raise RuntimeError("hybrid_pairs_210/bins.csv does not exactly match the frozen order")

canonical_affine = np.diag([0.9375, 0.9375, 1.0, 1.0])
for candidate_id in ordered:
    source = original_pairs if int(candidate_id) < 525 else new_pairs
    for name in ("bravo.nii.gz", "seg.nii.gz"):
        linked = synthetic_view / candidate_id / name
        expected = source / candidate_id / name
        if not linked.is_symlink():
            raise RuntimeError(f"hybrid source file is not a symlink: {linked}")
        if linked.resolve(strict=True) != resolve_file(expected):
            raise RuntimeError(f"hybrid source file points to the wrong file: {linked}")
        image = nib.load(linked)
        if image.shape != (256, 256, 150):
            raise RuntimeError(f"unexpected shape for {linked}: {image.shape}")
        if not np.allclose(image.affine, canonical_affine, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"unexpected affine for {linked}")
        data = np.asanyarray(image.dataobj)
        if not np.isfinite(data).all():
            raise RuntimeError(f"non-finite values in {linked}")
        if name == "seg.nii.gz":
            values = set(np.unique(data).tolist())
            if not values.issubset({0, 1}):
                raise RuntimeError(
                    f"segmentation is not binary in {linked}: {sorted(values)[:10]}"
                )

print("Validated frozen source: 151 canonical + 59 extension = 210 pairs")
PYEOF

echo "=== Convert exp17 Weighted Huber handoff dataset ==="
echo "Dataset ID:  $DATASET_ID"
echo "Training:    $EXPECTED_REAL_CASES real + $EXPECTED_SYNTHETIC_CASES synthetic"
echo "Test:        $EXPECTED_TEST_CASES held-out real"
echo "Synthetic:   $SYNTHETIC_VIEW"
echo "Output:      $RAW_DIR"

python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "$REAL_DIR" \
    --synthetic-dir "$SYNTHETIC_VIEW" \
    --nnunet-raw "$nnUNet_raw" \
    --dataset-id "$DATASET_ID" \
    --modality bravo

# Preserve the scientifically frozen order even though nnU-Net stores cases by
# identifier. The training split builder consumes this manifest explicitly.
order_tmp="${RAW_ORDER}.tmp_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
cp -- "$SOURCE_ORDER" "$order_tmp"
mv -f -- "$order_tmp" "$RAW_ORDER"

python - "$RAW_DIR" "$REFERENCE_RAW" "$SYNTHETIC_VIEW" "$RAW_ORDER" <<'PYEOF'
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

raw, reference, synthetic_view, order_path = map(Path, sys.argv[1:5])
case_info = json.loads((raw / "case_info.json").read_text())
reference_info = json.loads((reference / "case_info.json").read_text())
dataset_info = json.loads((raw / "dataset.json").read_text())
ordered_ids = order_path.read_text().splitlines()

expected_synthetic = [f"BrainMetSyn_{value}" for value in sorted(ordered_ids)]
if case_info["real_train_cases"] != reference_info["real_train_cases"]:
    raise RuntimeError("Dataset663 real training cases differ from Dataset600")
if case_info["test_cases"] != reference_info["test_cases"]:
    raise RuntimeError("Dataset663 held-out cases differ from Dataset600")
if len(case_info["real_train_cases"]) != 105:
    raise RuntimeError("Dataset663 must contain exactly 105 real training cases")
if case_info["synthetic_cases"] != expected_synthetic:
    raise RuntimeError("Dataset663 does not contain exactly the frozen 210-case source")
if len(case_info["test_cases"]) != 51:
    raise RuntimeError("Dataset663 must contain exactly 51 held-out real cases")
if case_info["modalities"] != ["bravo"]:
    raise RuntimeError(f"unexpected modalities: {case_info['modalities']!r}")
if dataset_info["numTraining"] != 315:
    raise RuntimeError(f"unexpected numTraining: {dataset_info['numTraining']!r}")
if dataset_info["channel_names"] != {"0": "BRAVO"}:
    raise RuntimeError(f"unexpected channel_names: {dataset_info['channel_names']!r}")

training_cases = case_info["real_train_cases"] + case_info["synthetic_cases"]
expected = {
    "imagesTr": {f"{value}_0000.nii.gz" for value in training_cases},
    "labelsTr": {f"{value}.nii.gz" for value in training_cases},
    "imagesTs": {f"{value}_0000.nii.gz" for value in case_info["test_cases"]},
    "labelsTs": {f"{value}.nii.gz" for value in case_info["test_cases"]},
}
for directory, expected_names in expected.items():
    actual_names = {path.name for path in (raw / directory).glob("*.nii.gz")}
    if actual_names != expected_names:
        raise RuntimeError(
            f"{directory} differs from case_info.json: actual={len(actual_names)}, "
            f"expected={len(expected_names)}, "
            f"missing={sorted(expected_names - actual_names)[:5]}, "
            f"extra={sorted(actual_names - expected_names)[:5]}"
        )

for candidate_id in ordered_ids:
    raw_image = raw / "imagesTr" / f"BrainMetSyn_{candidate_id}_0000.nii.gz"
    source_image = synthetic_view / candidate_id / "bravo.nii.gz"
    if not raw_image.is_symlink():
        raise RuntimeError(f"synthetic raw image is not a symlink: {raw_image}")
    if raw_image.resolve(strict=True) != source_image.resolve(strict=True):
        raise RuntimeError(f"synthetic raw image points to the wrong source: {raw_image}")


def require_same_volume(current_path: Path, reference_path: Path) -> None:
    current = nib.load(current_path)
    baseline = nib.load(reference_path)
    if current.shape != baseline.shape:
        raise RuntimeError(
            f"real-data shape mismatch: {current_path} has {current.shape}, "
            f"Dataset600 has {baseline.shape}"
        )
    if not np.allclose(
        current.affine,
        baseline.affine,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError(
            f"real-data affine mismatch between {current_path} and {reference_path}"
        )
    current_data = np.asanyarray(current.dataobj)
    baseline_data = np.asanyarray(baseline.dataobj)
    if not np.array_equal(current_data, baseline_data, equal_nan=True):
        raise RuntimeError(
            f"real-data voxel mismatch between {current_path} and {reference_path}"
        )


for case_id in case_info["real_train_cases"]:
    require_same_volume(
        raw / "imagesTr" / f"{case_id}_0000.nii.gz",
        reference / "imagesTr" / f"{case_id}_0000.nii.gz",
    )
    require_same_volume(
        raw / "labelsTr" / f"{case_id}.nii.gz",
        reference / "labelsTr" / f"{case_id}.nii.gz",
    )
for case_id in case_info["test_cases"]:
    require_same_volume(
        raw / "imagesTs" / f"{case_id}_0000.nii.gz",
        reference / "imagesTs" / f"{case_id}_0000.nii.gz",
    )
    require_same_volume(
        raw / "labelsTs" / f"{case_id}.nii.gz",
        reference / "labelsTs" / f"{case_id}.nii.gz",
    )

print(
    "Validated Dataset663: 105 real + 210 synthetic; 51 held-out real cases; "
    "all real images and labels numerically match Dataset600"
)
PYEOF

synthetic_order_sha256="$(sha256sum "$RAW_ORDER" | awk '{print $1}')"
source_complete_sha256="$(sha256sum "$SOURCE_COMPLETE" | awk '{print $1}')"
original_marker_sha256="$(sha256sum "$ORIGINAL_MASK_MARKER" | awk '{print $1}')"
new_mask_marker_sha256="$(sha256sum "$NEW_MASK_MARKER" | awk '{print $1}')"
expanded_marker_sha256="$(sha256sum "$EXPANDED_MASKS_MARKER" | awk '{print $1}')"
new_pair_marker_sha256="$(sha256sum "$NEW_PAIRS_MARKER" | awk '{print $1}')"
conversion_script_sha256="$(sha256sum "$REPO_ROOT/src/medgen/downstream/nnunet/convert_dataset.py" | awk '{print $1}')"
repository_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"

selection_tmp="${SOURCE_SELECTION}.tmp_${SLURM_JOB_ID}"
{
    echo "experiment_family=${EXPERIMENT_FAMILY}"
    echo "dataset_id=${DATASET_ID}"
    echo "generator=${GENERATOR}"
    echo "generator_description=${GENERATOR_DESCRIPTION}"
    echo "real_cases=${EXPECTED_REAL_CASES}"
    echo "synthetic_cases=${EXPECTED_SYNTHETIC_CASES}"
    echo "original_synthetic_cases=${EXPECTED_ORIGINAL_PAIRS}"
    echo "extension_synthetic_cases=${EXPECTED_EXTENSION_PAIRS}"
    echo "test_cases=${EXPECTED_TEST_CASES}"
    echo "synthetic_dir=${SYNTHETIC_VIEW}"
    echo "synthetic_candidate_order=${RAW_ORDER}"
    echo "synthetic_candidate_order_sha256=${synthetic_order_sha256}"
    echo "source_complete_marker=${SOURCE_COMPLETE}"
    echo "source_complete_marker_sha256=${source_complete_sha256}"
    echo "original_brain_mask_marker=${ORIGINAL_MASK_MARKER}"
    echo "original_brain_mask_marker_sha256=${original_marker_sha256}"
    echo "new_mask_marker=${NEW_MASK_MARKER}"
    echo "new_mask_marker_sha256=${new_mask_marker_sha256}"
    echo "expanded_mask_marker=${EXPANDED_MASKS_MARKER}"
    echo "expanded_mask_marker_sha256=${expanded_marker_sha256}"
    echo "new_pair_marker=${NEW_PAIRS_MARKER}"
    echo "new_pair_marker_sha256=${new_pair_marker_sha256}"
    echo "seg_checkpoint_sha256=${expected_seg_sha256}"
    echo "low_checkpoint_sha256=${expected_low_sha256}"
    echo "high_checkpoint_sha256=${expected_high_sha256}"
    echo "brain_mask_mode=in_place_atomic"
    echo "brain_mask_threshold=0.05"
    echo "brain_mask_fill_holes=true"
    echo "brain_mask_dilate_pixels=2"
    echo "conversion_script_sha256=${conversion_script_sha256}"
    echo "repository_commit=${repository_commit}"
} > "$selection_tmp"
mv -f -- "$selection_tmp" "$SOURCE_SELECTION"

source_selection_sha256="$(sha256sum "$SOURCE_SELECTION" | awk '{print $1}')"
marker_tmp="${CONVERSION_COMPLETE}.tmp_${SLURM_JOB_ID}"
{
    echo "experiment_family=${EXPERIMENT_FAMILY}"
    echo "dataset_id=${DATASET_ID}"
    echo "generator=${GENERATOR}"
    echo "generator_description=${GENERATOR_DESCRIPTION}"
    echo "real_cases=${EXPECTED_REAL_CASES}"
    echo "synthetic_cases=${EXPECTED_SYNTHETIC_CASES}"
    echo "original_synthetic_cases=${EXPECTED_ORIGINAL_PAIRS}"
    echo "extension_synthetic_cases=${EXPECTED_EXTENSION_PAIRS}"
    echo "training_cases=${EXPECTED_TRAINING_CASES}"
    echo "test_cases=${EXPECTED_TEST_CASES}"
    echo "synthetic_candidate_order=${RAW_ORDER}"
    echo "synthetic_candidate_order_sha256=${synthetic_order_sha256}"
    echo "source_selection=${SOURCE_SELECTION}"
    echo "source_selection_sha256=${source_selection_sha256}"
    echo "source_complete_marker_sha256=${source_complete_sha256}"
    echo "conversion_script_sha256=${conversion_script_sha256}"
    echo "repository_commit=${repository_commit}"
    echo "job_id=${SLURM_JOB_ID}"
} > "$marker_tmp"
mv -f -- "$marker_tmp" "$CONVERSION_COMPLETE"

echo "Raw Dataset${DATASET_ID} is complete: $RAW_DIR"
echo "Next: submit preprocess_exp17_weighted_huber_handoff_210_d663.slurm"
