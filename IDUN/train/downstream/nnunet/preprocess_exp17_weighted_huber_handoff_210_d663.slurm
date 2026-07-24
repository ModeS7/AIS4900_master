#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_pp_exp17_d663"
#SBATCH --output=IDUN/output/train/downstream/nnunet/preprocess_exp17_d663_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/preprocess_exp17_d663_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Extract a fresh Dataset663 fingerprint, transfer Dataset600's exact
# ResEnc-L plan under the controlled D600 plan/cache identifier, and preprocess
# the exp17 training pool of 105 real plus 210 validation-locked Weighted Huber
# handoff cases. Native Dataset663 plans and caches are not used.

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

readonly EXPERIMENT_FAMILY="exp17"
readonly GENERATOR="exp1_to_exp48c_t025"
readonly GENERATOR_DESCRIPTION="weighted_huber_handoff"
readonly DATASET_ID=663
readonly EXPECTED_REAL_CASES=105
readonly EXPECTED_SYNTHETIC_CASES=210
readonly EXPECTED_ORIGINAL_SYNTHETIC_CASES=151
readonly EXPECTED_EXTENSION_SYNTHETIC_CASES=59
readonly EXPECTED_TRAINING_CASES=315
readonly EXPECTED_TEST_CASES=51
readonly SOURCE_DATASET_ID=600
readonly SOURCE_PLANS="nnUNetResEncUNetLPlans"
readonly TARGET_PLANS="nnUNetResEncUNetLPlansD600"

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly EXTENSION_ROOT="${EVAL_ROOT}/weighted_huber_handoff_extension_to210"
readonly SYNTHETIC_VIEW="${EXTENSION_ROOT}/hybrid_pairs_210"
readonly SOURCE_ORDER="${EXTENSION_ROOT}/hybrid_candidate_ids_210.txt"
readonly SOURCE_COMPLETE="${SYNTHETIC_VIEW}/.hybrid_view_complete"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

readonly RAW_DIR="${nnUNet_raw}/Dataset${DATASET_ID}_BrainMet"
readonly RAW_ORDER="${RAW_DIR}/synthetic_candidate_order.txt"
readonly CONVERSION_COMPLETE="${RAW_DIR}/.exp17_d663_conversion_complete"
readonly PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset${DATASET_ID}_BrainMet"
readonly SOURCE_PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset${SOURCE_DATASET_ID}_BrainMet"
readonly SOURCE_PLAN_FILE="${SOURCE_PREPROCESSED_DIR}/${SOURCE_PLANS}.json"
readonly HISTORICAL_PLAN_FILE="${nnUNet_results}/exp3_baseline_v2_d600/Dataset600_BrainMet/nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/plans.json"
readonly TARGET_PLAN_FILE="${PREPROCESSED_DIR}/${TARGET_PLANS}.json"
readonly COMPLETE_MARKER="${PREPROCESSED_DIR}/.exp17_d600_preprocess_complete"
readonly LOCK_FILE="${nnUNet_preprocessed}/.exp17_d600_preprocess_Dataset${DATASET_ID}.lock"

[[ -d "$SYNTHETIC_VIEW" ]] || fatal "frozen 210-pair source view not found: $SYNTHETIC_VIEW"
[[ -s "$SOURCE_ORDER" ]] || fatal "ordered 210-case source manifest not found: $SOURCE_ORDER"
[[ -s "$SOURCE_COMPLETE" ]] || fatal "210-pair source view is incomplete: $SOURCE_COMPLETE"
require_marker_value "$SOURCE_COMPLETE" target_pairs "$EXPECTED_SYNTHETIC_CASES"
require_marker_value "$SOURCE_COMPLETE" original_pairs "$EXPECTED_ORIGINAL_SYNTHETIC_CASES"
require_marker_value "$SOURCE_COMPLETE" extension_pairs "$EXPECTED_EXTENSION_SYNTHETIC_CASES"
require_marker_value "$SOURCE_COMPLETE" ordered_ids "$SOURCE_ORDER"

[[ -d "$RAW_DIR" ]] || fatal "raw Dataset${DATASET_ID} not found: $RAW_DIR"
[[ -s "$RAW_DIR/case_info.json" ]] || fatal "case_info.json not found: $RAW_DIR"
[[ -s "$RAW_DIR/dataset.json" ]] || fatal "dataset.json not found: $RAW_DIR"
[[ -s "$RAW_DIR/source_selection.txt" ]] || fatal "source provenance not found: $RAW_DIR"
[[ -s "$RAW_ORDER" ]] || fatal "raw ordered synthetic manifest not found: $RAW_ORDER"
[[ -s "$CONVERSION_COMPLETE" ]] || \
    fatal "exp17 Dataset663 conversion is incomplete: $CONVERSION_COMPLETE"
[[ -s "$SOURCE_PLAN_FILE" ]] || fatal "Dataset600 source plan not found: $SOURCE_PLAN_FILE"
[[ -s "$HISTORICAL_PLAN_FILE" ]] || \
    fatal "immutable exp3 training plan not found: $HISTORICAL_PLAN_FILE"
cmp -s -- "$SOURCE_ORDER" "$RAW_ORDER" || \
    fatal "raw synthetic order differs from the frozen 210-case source order"

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

python - \
    "$RAW_DIR" \
    "$RAW_ORDER" \
    "$SYNTHETIC_VIEW" \
    "$EXPERIMENT_FAMILY" \
    "$GENERATOR" \
    "$GENERATOR_DESCRIPTION" \
    "$DATASET_ID" <<'PYEOF'
import hashlib
import json
import re
import sys
from pathlib import Path

raw = Path(sys.argv[1])
order_path = Path(sys.argv[2])
synthetic_view = Path(sys.argv[3])
expected_family = sys.argv[4]
expected_generator = sys.argv[5]
expected_description = sys.argv[6]
expected_dataset_id = int(sys.argv[7])

case_info = json.loads((raw / "case_info.json").read_text())
dataset_info = json.loads((raw / "dataset.json").read_text())
source_path = raw / "source_selection.txt"
conversion_path = raw / ".exp17_d663_conversion_complete"


def parse_marker(path: Path) -> dict[str, str]:
    pairs = [
        line.split("=", 1)
        for line in path.read_text().splitlines()
        if "=" in line
    ]
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate keys in provenance marker: {path}")
    return dict(pairs)


source = parse_marker(source_path)
conversion = parse_marker(conversion_path)
ordered_ids = order_path.read_text().splitlines()

required_source = {
    "experiment_family": expected_family,
    "dataset_id": str(expected_dataset_id),
    "generator": expected_generator,
    "generator_description": expected_description,
    "real_cases": "105",
    "synthetic_cases": "210",
    "original_synthetic_cases": "151",
    "extension_synthetic_cases": "59",
    "test_cases": "51",
}
for key, expected in required_source.items():
    actual = source.get(key)
    if actual != expected:
        raise RuntimeError(
            f"source_selection.txt {key}: expected {expected!r}, got {actual!r}"
        )

required_conversion = {
    **required_source,
    "training_cases": "315",
    "synthetic_candidate_order": str(order_path),
    "source_selection": str(source_path),
}
for key, expected in required_conversion.items():
    actual = conversion.get(key)
    if actual != expected:
        raise RuntimeError(
            f"conversion marker {key}: expected {expected!r}, got {actual!r}"
        )

order_sha256 = hashlib.sha256(order_path.read_bytes()).hexdigest()
if source.get("synthetic_candidate_order_sha256") != order_sha256:
    raise RuntimeError(
        "source_selection.txt synthetic order SHA-256 does not match the raw manifest"
    )
if conversion.get("synthetic_candidate_order_sha256") != order_sha256:
    raise RuntimeError(
        "conversion marker synthetic order SHA-256 does not match the raw manifest"
    )
source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
if conversion.get("source_selection_sha256") != source_sha256:
    raise RuntimeError(
        "conversion marker source-selection SHA-256 does not match source_selection.txt"
    )

if len(ordered_ids) != 210 or len(set(ordered_ids)) != 210:
    raise RuntimeError("synthetic_candidate_order.txt must contain 210 unique IDs")
if any(re.fullmatch(r"[0-9]{5}", candidate_id) is None for candidate_id in ordered_ids):
    raise RuntimeError("synthetic_candidate_order.txt contains a malformed candidate ID")

expected_synthetic = {f"BrainMetSyn_{candidate_id}" for candidate_id in ordered_ids}
actual_synthetic = case_info["synthetic_cases"]
if len(actual_synthetic) != 210 or set(actual_synthetic) != expected_synthetic:
    raise RuntimeError(
        "case_info.json does not contain exactly the ordered 210-case source pool"
    )
if len(case_info["real_train_cases"]) != 105:
    raise RuntimeError("case_info.json does not contain exactly 105 real train cases")
if len(case_info["test_cases"]) != 51:
    raise RuntimeError("case_info.json does not contain exactly 51 held-out real cases")
if case_info["modalities"] != ["bravo"]:
    raise RuntimeError(f"unexpected modalities: {case_info['modalities']!r}")
if dataset_info["numTraining"] != 315:
    raise RuntimeError(
        f"dataset.json numTraining: expected 315, got {dataset_info['numTraining']!r}"
    )
if dataset_info["channel_names"] != {"0": "BRAVO"}:
    raise RuntimeError(
        f"unexpected channel_names: {dataset_info['channel_names']!r}"
    )

expected_tr = {
    *(f"{case_id}_0000.nii.gz" for case_id in case_info["real_train_cases"]),
    *(f"{case_id}_0000.nii.gz" for case_id in actual_synthetic),
}
expected_labels_tr = {
    *(f"{case_id}.nii.gz" for case_id in case_info["real_train_cases"]),
    *(f"{case_id}.nii.gz" for case_id in actual_synthetic),
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

for candidate_id in ordered_ids:
    raw_image = raw / "imagesTr" / f"BrainMetSyn_{candidate_id}_0000.nii.gz"
    source_image = synthetic_view / candidate_id / "bravo.nii.gz"
    if not raw_image.is_symlink():
        raise RuntimeError(f"synthetic raw image is not a symlink: {raw_image}")
    if raw_image.resolve(strict=True) != source_image.resolve(strict=True):
        raise RuntimeError(
            f"synthetic raw image points to the wrong source: {raw_image}"
        )

print(
    "Validated raw Dataset663: 105 real + 210 brain-masked Weighted Huber "
    "handoff cases; 51 held-out real cases"
)
PYEOF

mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"
exec 9>"$LOCK_FILE"
flock -n 9 || fatal "another controlled preprocessing job is using Dataset${DATASET_ID}"
rm -f -- "$COMPLETE_MARKER"

echo "=== Dataset600-controlled exp17 preprocessing ==="
echo "Generator:   $GENERATOR_DESCRIPTION ($GENERATOR)"
echo "Dataset:     $DATASET_ID"
echo "Cases:       $EXPECTED_REAL_CASES real + $EXPECTED_SYNTHETIC_CASES synthetic"
echo "Source plan: Dataset600/$SOURCE_PLANS"
echo "Target plan: $TARGET_PLANS"

# --clean refreshes Dataset663's fingerprint and removes any stale cache while
# leaving Dataset600 and its immutable historical plan untouched.
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

python - \
    "$TARGET_PLAN_FILE" \
    "$PREPROCESSED_DIR" \
    "$RAW_DIR" \
    "$CONVERSION_COMPLETE" <<'PYEOF'
import json
import sys
from pathlib import Path

plan_path = Path(sys.argv[1])
preprocessed = Path(sys.argv[2])
raw = Path(sys.argv[3])
conversion_complete = Path(sys.argv[4])
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
if len(case_ids) != 315 or label_ids != case_ids:
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
            "Blosc2 controlled cache does not exactly match all 315 case IDs: "
            f"data={len(b2nd_data)}, seg={len(b2nd_seg)}, pkl={len(pkl_ids)}"
        )
elif npz_ids != case_ids or pkl_ids != case_ids:
    raise RuntimeError(
        "NPZ controlled cache does not exactly match all 315 case IDs: "
        f"npz={len(npz_ids)}, pkl={len(pkl_ids)}"
    )

gt_ids = {
    path.name[:-len(".nii.gz")]
    for path in (preprocessed / "gt_segmentations").glob("*.nii.gz")
}
if gt_ids != case_ids:
    raise RuntimeError(
        "gt_segmentations does not exactly match all 315 case IDs: "
        f"got {len(gt_ids)}"
    )

fingerprint = preprocessed / "dataset_fingerprint.json"
if not fingerprint.is_file():
    raise RuntimeError(f"dataset fingerprint is missing: {fingerprint}")
if fingerprint.stat().st_mtime_ns < conversion_complete.stat().st_mtime_ns:
    raise RuntimeError("dataset fingerprint predates completed exp17 conversion")
if cache.stat().st_mtime_ns < plan_path.stat().st_mtime_ns:
    raise RuntimeError("controlled cache predates its transferred plan")
print(f"Validated controlled cache {cache.name}: 315 training cases")
PYEOF

source_plan_sha256="$(sha256sum "$SOURCE_PLAN_FILE" | awk '{print $1}')"
target_plan_sha256="$(sha256sum "$TARGET_PLAN_FILE" | awk '{print $1}')"
synthetic_order_sha256="$(sha256sum "$RAW_ORDER" | awk '{print $1}')"
conversion_marker_sha256="$(sha256sum "$CONVERSION_COMPLETE" | awk '{print $1}')"
marker_tmp="${COMPLETE_MARKER}.tmp_${SLURM_JOB_ID:?SLURM_JOB_ID must be set}"
{
    echo "experiment_family=${EXPERIMENT_FAMILY}"
    echo "dataset_id=${DATASET_ID}"
    echo "generator=${GENERATOR}"
    echo "generator_description=${GENERATOR_DESCRIPTION}"
    echo "real_cases=${EXPECTED_REAL_CASES}"
    echo "synthetic_cases=${EXPECTED_SYNTHETIC_CASES}"
    echo "original_synthetic_cases=${EXPECTED_ORIGINAL_SYNTHETIC_CASES}"
    echo "extension_synthetic_cases=${EXPECTED_EXTENSION_SYNTHETIC_CASES}"
    echo "training_cases=${EXPECTED_TRAINING_CASES}"
    echo "test_cases=${EXPECTED_TEST_CASES}"
    echo "synthetic_candidate_order=${RAW_ORDER}"
    echo "synthetic_candidate_order_sha256=${synthetic_order_sha256}"
    echo "conversion_marker=${CONVERSION_COMPLETE}"
    echo "conversion_marker_sha256=${conversion_marker_sha256}"
    echo "source_dataset_id=${SOURCE_DATASET_ID}"
    echo "source_plans=${SOURCE_PLANS}"
    echo "source_plan_sha256=${source_plan_sha256}"
    echo "target_plans=${TARGET_PLANS}"
    echo "target_plan_sha256=${target_plan_sha256}"
    echo "patch_size=160,192,160"
    echo "batch_size=3"
    echo "normalization=ZScoreNormalization"
    echo "use_mask_for_norm=true"
    echo "brain_mask_mode=in_place_atomic"
    echo "brain_mask_threshold=0.05"
    echo "brain_mask_fill_holes=true"
    echo "brain_mask_dilate_pixels=2"
    echo "job_id=${SLURM_JOB_ID}"
} > "$marker_tmp"
mv -f -- "$marker_tmp" "$COMPLETE_MARKER"

echo "Dataset${DATASET_ID} controlled exp17 preprocessing complete."
