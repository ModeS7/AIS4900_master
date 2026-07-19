#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --array=0-13%2
#SBATCH --job-name="nn_pp_common105"
#SBATCH --output=IDUN/output/train/downstream/nnunet/preprocess_common105_%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/preprocess_common105_%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Plan and preprocess Dataset650--Dataset663 with the same default and
# ResEnc-L nnU-Net settings used by the earlier synthetic-only experiments.
# This is additive and refuses to overwrite any preprocessed dataset.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
(( ARRAY_INDEX >= 0 && ARRAY_INDEX < 14 )) || \
    fatal "array index is outside 0--13: $ARRAY_INDEX"

readonly DATASET_ID="$((650 + ARRAY_INDEX))"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

readonly RAW_DIR="${nnUNet_raw}/Dataset${DATASET_ID}_BrainMet"
readonly PREPROCESSED_DIR="${nnUNet_preprocessed}/Dataset${DATASET_ID}_BrainMet"

[[ -d "$RAW_DIR" ]] || fatal "raw Dataset${DATASET_ID} not found: $RAW_DIR"
[[ -s "$RAW_DIR/case_info.json" ]] || fatal "case_info.json not found: $RAW_DIR"
[[ -s "$RAW_DIR/dataset.json" ]] || fatal "dataset.json not found: $RAW_DIR"

# Dataset IDs are global in nnU-Net. Refuse any existing preprocessed
# directory with this ID, including a differently named stale dataset.
if compgen -G "$nnUNet_preprocessed/Dataset${DATASET_ID}_*" > /dev/null; then
    fatal "preprocessed Dataset${DATASET_ID} already exists under $nnUNet_preprocessed"
fi

cd "$REPO_ROOT"
module purge
module load Anaconda3/2024.02-1
conda activate nnunet

python - "$RAW_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
case_info = json.loads((raw / "case_info.json").read_text())
dataset_info = json.loads((raw / "dataset.json").read_text())

assert len(case_info["real_train_cases"]) == 105
assert len(case_info["synthetic_cases"]) == 105
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

assert actual_tr == expected_tr, (len(actual_tr), len(expected_tr))
assert actual_labels_tr == expected_labels_tr, (
    len(actual_labels_tr), len(expected_labels_tr)
)
assert actual_ts == expected_ts, (len(actual_ts), len(expected_ts))
assert actual_labels_ts == expected_labels_ts, (
    len(actual_labels_ts), len(expected_labels_ts)
)
print(f"Validated raw {raw.name}: 210 train and 51 test pairs")
PYEOF

mkdir -p "$nnUNet_preprocessed" "$nnUNet_results"

echo "=== Plan and preprocess Dataset${DATASET_ID} ==="
echo "Raw:          $RAW_DIR"
echo "Preprocessed: $PREPROCESSED_DIR"

nnUNetv2_plan_and_preprocess \
    -d "$DATASET_ID" \
    --verify_dataset_integrity \
    -c 3d_fullres \
    -np 4

nnUNetv2_plan_and_preprocess \
    -d "$DATASET_ID" \
    -pl nnUNetPlannerResEncL \
    -c 3d_fullres \
    -np 4

[[ -s "$PREPROCESSED_DIR/nnUNetPlans.json" ]] || \
    fatal "default plans were not created for Dataset${DATASET_ID}"
[[ -s "$PREPROCESSED_DIR/nnUNetResEncUNetLPlans.json" ]] || \
    fatal "ResEnc-L plans were not created for Dataset${DATASET_ID}"

python - "$PREPROCESSED_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

preprocessed = Path(sys.argv[1])
identifiers = set()
for plan_name in ("nnUNetPlans.json", "nnUNetResEncUNetLPlans.json"):
    plans = json.loads((preprocessed / plan_name).read_text())
    configuration = plans["configurations"]["3d_fullres"]
    identifiers.add(configuration["data_identifier"])

for identifier in sorted(identifiers):
    cache = preprocessed / identifier
    assert cache.is_dir(), cache
    b2nd_count = len(list(cache.glob("*.b2nd")))
    npz_count = len(list(cache.glob("*.npz")))
    if b2nd_count:
        # Current nnU-Net stores one data and one segmentation shard per case.
        assert b2nd_count == 420, (cache, b2nd_count)
    else:
        # Compatibility with older nnU-Net releases using one archive per case.
        assert npz_count == 210, (cache, npz_count)
    print(
        f"Validated {identifier}: "
        f"{b2nd_count} .b2nd / {npz_count} .npz files"
    )
PYEOF

echo "Dataset${DATASET_ID} planning and preprocessing complete."
