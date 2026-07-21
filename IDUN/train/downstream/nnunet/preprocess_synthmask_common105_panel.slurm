#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --array=0-4%2
#SBATCH --job-name="nn_pp_common105"
#SBATCH --output=IDUN/output/train/downstream/nnunet/preprocess_common105_%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/preprocess_common105_%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Extract fresh fingerprints from the brain-masked common-105 panel, transfer
# Dataset600's exact ResEnc-L plan under a distinct plan/cache identifier, and
# preprocess the five exp16_2 datasets. Native plans and caches are not used.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

require_marker_value() {
    local marker="$1"
    local key="$2"
    local expected="$3"
    local -a values
    mapfile -t values < <(sed -n "s/^${key}=//p" "$marker")
    [[ ${#values[@]} -eq 1 && "${values[0]}" == "$expected" ]] || \
        fatal "$marker must contain exactly ${key}=${expected}"
}

readonly LABELS=(
    exp1_1_1000
    exp1_1_1000plus
    exp32_2_1000
    exp1_to_exp48c_t025
    exp1_to_exp48d_t025
)
readonly DATASET_IDS=(650 651 652 661 662)
readonly SOURCE_DATASET_ID=600
readonly SOURCE_PLANS="nnUNetResEncUNetLPlans"
readonly TARGET_PLANS="nnUNetResEncUNetLPlansD600"

if [[ -n "${EXP16_LABEL_OVERRIDE:-}" || -n "${EXP16_DATASET_ID_OVERRIDE:-}" ]]; then
    [[ -n "${EXP16_LABEL_OVERRIDE:-}" && -n "${EXP16_DATASET_ID_OVERRIDE:-}" ]] || \
        fatal "EXP16_LABEL_OVERRIDE and EXP16_DATASET_ID_OVERRIDE must be set together"
    [[ "$EXP16_DATASET_ID_OVERRIDE" =~ ^[0-9]{3}$ ]] || \
        fatal "EXP16_DATASET_ID_OVERRIDE must be a three-digit nnU-Net dataset ID"
    LABEL="$EXP16_LABEL_OVERRIDE"
    DATASET_ID="$EXP16_DATASET_ID_OVERRIDE"
else
    ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
    (( ARRAY_INDEX >= 0 && ARRAY_INDEX < ${#DATASET_IDS[@]} )) || \
        fatal "array index is outside 0--4: $ARRAY_INDEX"
    LABEL="${LABELS[$ARRAY_INDEX]}"
    DATASET_ID="${DATASET_IDS[$ARRAY_INDEX]}"
fi
readonly LABEL DATASET_ID

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly MASK_COMPLETE="${EVAL_ROOT}/.brain_mask_stage3_all14_complete"
readonly MATCHED_ROOT="${EVAL_ROOT}/matched_common105"
readonly COMMON_IDS_FILE="${MATCHED_ROOT}/common_candidate_ids.txt"
readonly CANONICAL_DATASET="${EVAL_ROOT}/${LABEL}"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

readonly RAW_DIR="${nnUNet_raw}/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset${DATASET_ID}_BrainMet"
readonly SOURCE_PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset${SOURCE_DATASET_ID}_BrainMet"
readonly SOURCE_PLAN_FILE="${SOURCE_PREPROCESSED_DIR}/${SOURCE_PLANS}.json"
readonly HISTORICAL_PLAN_FILE="${nnUNet_results}/exp3_baseline_v2_d600/Dataset600_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/plans.json"
readonly TARGET_PLAN_FILE="${PREPROCESSED_DIR}/${TARGET_PLANS}.json"
readonly COMPLETE_MARKER="${PREPROCESSED_DIR}/.exp16_2_d600_preprocess_complete"
readonly LOCK_FILE="${nnUNet_preprocessed}/.exp16_2_d600_preprocess_Dataset${DATASET_ID}.lock"

[[ -s "$MASK_COMPLETE" ]] || \
    fatal "all-14 brain masking is not complete: $MASK_COMPLETE"
require_marker_value "$MASK_COMPLETE" mode in_place_atomic
require_marker_value "$MASK_COMPLETE" threshold 0.05
require_marker_value "$MASK_COMPLETE" fill_holes true
require_marker_value "$MASK_COMPLETE" dilate_pixels 2

[[ -d "$RAW_DIR" ]] || fatal "raw Dataset${DATASET_ID} not found: $RAW_DIR"
[[ -s "$RAW_DIR/case_info.json" ]] || fatal "case_info.json not found: $RAW_DIR"
[[ -s "$RAW_DIR/dataset.json" ]] || fatal "dataset.json not found: $RAW_DIR"
[[ -s "$RAW_DIR/source_selection.txt" ]] || fatal "source provenance not found: $RAW_DIR"
[[ -s "$COMMON_IDS_FILE" ]] || fatal "common candidate IDs not found: $COMMON_IDS_FILE"
[[ -s "$SOURCE_PLAN_FILE" ]] || fatal "Dataset600 source plan not found: $SOURCE_PLAN_FILE"
[[ -s "$HISTORICAL_PLAN_FILE" ]] || \
    fatal "immutable exp3 training plan not found: $HISTORICAL_PLAN_FILE"

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet

python - "$SOURCE_PLAN_FILE" "$HISTORICAL_PLAN_FILE" <<'PYEOF'
import json
import sys
from pathlib import Path

live_path, historical_path = map(Path, sys.argv[1:3])
live = json.loads(live_path.read_text())
historical = json.loads(historical_path.read_text())
if live != historical:
    raise RuntimeError(
        "Dataset600 live plan differs from the plan actually saved by exp3: "
        f"{live_path} != {historical_path}"
    )
print("Dataset600 plan matches the immutable plan saved by exp3")
PYEOF

python - "$RAW_DIR" "$COMMON_IDS_FILE" "$CANONICAL_DATASET" "$LABEL" "$DATASET_ID" <<'PYEOF'
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
common_ids_file = Path(sys.argv[2])
canonical = Path(sys.argv[3])
expected_label = sys.argv[4]
expected_dataset_id = int(sys.argv[5])

case_info = json.loads((raw / "case_info.json").read_text())
dataset_info = json.loads((raw / "dataset.json").read_text())
source = dict(
    line.split("=", 1)
    for line in (raw / "source_selection.txt").read_text().splitlines()
    if "=" in line
)
common_ids = common_ids_file.read_text().splitlines()

assert source["generator"] == expected_label
assert int(source["dataset_id"]) == expected_dataset_id
assert len(common_ids) == len(set(common_ids)) == 105
assert case_info["synthetic_cases"] == [f"BrainMetSyn_{i}" for i in common_ids]
assert len(case_info["real_train_cases"]) == 105
assert len(case_info["test_cases"]) == 51
assert case_info["modalities"] == ["bravo"]
assert dataset_info["numTraining"] == 210
assert dataset_info["channel_names"] == {"0": "BRAVO"}

expected_tr = {
    *(f"{case_id}_0000.nii.gz" for case_id in case_info["real_train_cases"]),
    *(f"{case_id}_0000.nii.gz" for case_id in case_info["synthetic_cases"]),
}
expected_labels_tr = {
    *(f"{case_id}.nii.gz" for case_id in case_info["real_train_cases"]),
    *(f"{case_id}.nii.gz" for case_id in case_info["synthetic_cases"]),
}
expected_ts = {f"{case_id}_0000.nii.gz" for case_id in case_info["test_cases"]}
expected_labels_ts = {f"{case_id}.nii.gz" for case_id in case_info["test_cases"]}

actual_tr = {path.name for path in (raw / "imagesTr").glob("*.nii.gz")}
actual_labels_tr = {path.name for path in (raw / "labelsTr").glob("*.nii.gz")}
actual_ts = {path.name for path in (raw / "imagesTs").glob("*.nii.gz")}
actual_labels_ts = {path.name for path in (raw / "labelsTs").glob("*.nii.gz")}
for name, actual, expected in (
    ("imagesTr", actual_tr, expected_tr),
    ("labelsTr", actual_labels_tr, expected_labels_tr),
    ("imagesTs", actual_ts, expected_ts),
    ("labelsTs", actual_labels_ts, expected_labels_ts),
):
    if actual != expected:
        raise RuntimeError(
            f"{name} does not exactly match case_info.json: "
            f"actual={len(actual)}, expected={len(expected)}, "
            f"missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )

for candidate_id in common_ids:
    raw_image = raw / "imagesTr" / f"BrainMetSyn_{candidate_id}_0000.nii.gz"
    canonical_image = canonical / candidate_id / "bravo.nii.gz"
    assert raw_image.is_symlink(), raw_image
    assert raw_image.resolve(strict=True) == canonical_image.resolve(strict=True), (
        raw_image,
        canonical_image,
    )

print(
    f"Validated raw {raw.name}: 105 real + 105 brain-masked {expected_label}; "
    "51 held-out real cases"
)
PYEOF

mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"
exec 9>"$LOCK_FILE"
flock -n 9 || fatal "another controlled preprocessing job is using Dataset${DATASET_ID}"
rm -f -- "$COMPLETE_MARKER"

echo "=== Dataset600-controlled preprocessing ==="
echo "Generator:   $LABEL"
echo "Dataset:     $DATASET_ID"
echo "Source plan: Dataset600/$SOURCE_PLANS"
echo "Target plan: $TARGET_PLANS"

# --clean is required because these image files changed when Stage 3 was
# applied in place. It refreshes the target fingerprint without touching D600.
nnUNetv2_extract_fingerprint \
    -d "$DATASET_ID" \
    -np 4 \
    --verify_dataset_integrity \
    --clean \
    --no_pbar

# Fingerprint extraction does not install dataset.json in every nnU-Net
# version. Keep the target metadata paired with the refreshed fingerprint.
cp -- "$RAW_DIR/dataset.json" "$PREPROCESSED_DIR/dataset.json"

nnUNetv2_move_plans_between_datasets \
    -s "$SOURCE_DATASET_ID" \
    -t "$DATASET_ID" \
    -sp "$SOURCE_PLANS" \
    -tp "$TARGET_PLANS"

[[ -s "$TARGET_PLAN_FILE" ]] || fatal "transferred target plan was not created"

python - "$SOURCE_PLAN_FILE" "$TARGET_PLAN_FILE" "$SOURCE_PLANS" "$TARGET_PLANS" "$DATASET_ID" <<'PYEOF'
import copy
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
target_path = Path(sys.argv[2])
source_plans_name = sys.argv[3]
target_plans_name = sys.argv[4]
target_dataset_id = int(sys.argv[5])

source = json.loads(source_path.read_text())
target = json.loads(target_path.read_text())

if not target["dataset_name"].startswith(f"Dataset{target_dataset_id}_"):
    raise RuntimeError(
        f"dataset_name: expected Dataset{target_dataset_id}_*, "
        f"got {target['dataset_name']!r}"
    )

expected = copy.deepcopy(source)
expected["dataset_name"] = target["dataset_name"]
expected["plans_name"] = target_plans_name
# The official mover selects the target dataset's compatible reader.
expected["image_reader_writer"] = target["image_reader_writer"]
for configuration in expected["configurations"].values():
    source_identifier = configuration.get("data_identifier")
    if source_identifier is None:
        continue
    if source_identifier.startswith(source_plans_name):
        suffix = source_identifier[len(source_plans_name):]
        configuration["data_identifier"] = target_plans_name + suffix
    else:
        configuration["data_identifier"] = (
            f"{target_plans_name}_{source_identifier}"
        )

if target != expected:
    raise RuntimeError(
        "transferred plan is not an exact Dataset600 plan copy with only "
        "the official target metadata/data-identifier rewrites"
    )

configuration = target["configurations"]["3d_fullres"]
required = {
    "patch_size": [160, 192, 160],
    "batch_size": 3,
    "spacing": [1.0, 0.9375, 0.9375],
    "normalization_schemes": ["ZScoreNormalization"],
    "use_mask_for_norm": [True],
    "preprocessor_name": "DefaultPreprocessor",
    "batch_dice": False,
}
for key, expected_value in required.items():
    actual = configuration.get(key)
    if actual != expected_value:
        raise RuntimeError(
            f"{key}: expected {expected_value!r}, got {actual!r}"
        )

architecture = configuration["architecture"]
expected_network = (
    "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet"
)
if architecture["network_class_name"] != expected_network:
    raise RuntimeError(
        f"network_class_name: expected {expected_network!r}, "
        f"got {architecture['network_class_name']!r}"
    )
expected_features = [32, 64, 128, 256, 320, 320]
actual_features = architecture["arch_kwargs"]["features_per_stage"]
if actual_features != expected_features:
    raise RuntimeError(
        f"features_per_stage: expected {expected_features!r}, "
        f"got {actual_features!r}"
    )

print(
    "Transferred plan exactly matches Dataset600: patch 160x192x160, "
    "batch 3, masked Z-score"
)
PYEOF

nnUNetv2_preprocess \
    -d "$DATASET_ID" \
    -plans_name "$TARGET_PLANS" \
    -c 3d_fullres \
    -np 4 \
    --no_pbar

python - "$TARGET_PLAN_FILE" "$PREPROCESSED_DIR" "$RAW_DIR" "$MASK_COMPLETE" <<'PYEOF'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
preprocessed = Path(sys.argv[2])
raw = Path(sys.argv[3])
mask_complete = Path(sys.argv[4])
plans = json.loads(plan_path.read_text())
configuration = plans["configurations"]["3d_fullres"]
cache = preprocessed / configuration["data_identifier"]
if not cache.is_dir():
    raise RuntimeError(f"controlled cache is missing: {cache}")

image_suffix = "_0000.nii.gz"
case_ids = {
    path.name[:-len(image_suffix)]
    for path in (raw / "imagesTr").glob(f"*{image_suffix}")
}
label_ids = {
    path.name[:-len(".nii.gz")]
    for path in (raw / "labelsTr").glob("*.nii.gz")
}
if len(case_ids) != 210 or label_ids != case_ids:
    raise RuntimeError(
        "raw training cases are inconsistent: "
        f"images={len(case_ids)}, labels={len(label_ids)}, "
        f"missing_labels={sorted(case_ids - label_ids)[:5]}, "
        f"missing_images={sorted(label_ids - case_ids)[:5]}"
    )

b2nd_data = {
    path.name[:-len(".b2nd")]
    for path in cache.glob("*.b2nd")
    if not path.name.endswith("_seg.b2nd")
}
b2nd_seg = {
    path.name[:-len("_seg.b2nd")]
    for path in cache.glob("*_seg.b2nd")
}
npz_ids = {path.stem for path in cache.glob("*.npz")}
pkl_ids = {path.stem for path in cache.glob("*.pkl")}
if b2nd_data or b2nd_seg:
    if b2nd_data != case_ids or b2nd_seg != case_ids or pkl_ids != case_ids:
        raise RuntimeError(
            "Blosc2 controlled cache does not exactly match all 210 case IDs: "
            f"data={len(b2nd_data)}, seg={len(b2nd_seg)}, pkl={len(pkl_ids)}"
        )
elif npz_ids != case_ids or pkl_ids != case_ids:
    raise RuntimeError(
        "NPZ controlled cache does not exactly match all 210 case IDs: "
        f"npz={len(npz_ids)}, pkl={len(pkl_ids)}"
    )

gt_ids = {
    path.name[:-len(".nii.gz")]
    for path in (preprocessed / "gt_segmentations").glob("*.nii.gz")
}
if gt_ids != case_ids:
    raise RuntimeError(
        "gt_segmentations does not exactly match all 210 case IDs: "
        f"got {len(gt_ids)}"
    )

fingerprint = preprocessed / "dataset_fingerprint.json"
if not fingerprint.is_file():
    raise RuntimeError(f"dataset fingerprint is missing: {fingerprint}")
if fingerprint.stat().st_mtime_ns < mask_complete.stat().st_mtime_ns:
    raise RuntimeError("dataset fingerprint predates completed brain masking")
if cache.stat().st_mtime_ns < plan_path.stat().st_mtime_ns:
    raise RuntimeError("controlled cache predates its transferred plan")
print(f"Validated controlled cache {cache.name}: 210 training cases")
PYEOF

source_plan_sha256="$(sha256sum "$SOURCE_PLAN_FILE" | awk '{print $1}')"
target_plan_sha256="$(sha256sum "$TARGET_PLAN_FILE" | awk '{print $1}')"
marker_tmp="${COMPLETE_MARKER}.tmp_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
{
    echo "dataset_id=${DATASET_ID}"
    echo "generator=${LABEL}"
    echo "source_dataset_id=${SOURCE_DATASET_ID}"
    echo "source_plans=${SOURCE_PLANS}"
    echo "source_plan_sha256=${source_plan_sha256}"
    echo "target_plans=${TARGET_PLANS}"
    echo "target_plan_sha256=${target_plan_sha256}"
    echo "patch_size=160,192,160"
    echo "batch_size=3"
    echo "normalization=ZScoreNormalization"
    echo "use_mask_for_norm=true"
    echo "brain_mask_marker=${MASK_COMPLETE}"
    echo "brain_mask_mode=in_place_atomic"
    echo "brain_mask_threshold=0.05"
    echo "brain_mask_fill_holes=true"
    echo "brain_mask_dilate_pixels=2"
    echo "job_id=${SLURM_JOB_ID}"
} > "$marker_tmp"
mv -f -- "$marker_tmp" "$COMPLETE_MARKER"

echo "Dataset${DATASET_ID} controlled preprocessing complete."
