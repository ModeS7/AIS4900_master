#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=idun-06-02
#SBATCH --array=0-4%2
#SBATCH --job-name="nn_common105"
#SBATCH --output=IDUN/output/train/downstream/nnunet/convert_common105_%A_%a.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/convert_common105_%A_%a.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Convert the five matched common-105 generator datasets to nnU-Net raw format.
# This is additive: it creates Dataset650, 651, 652, 661, and 662 and never deletes or
# overwrites an existing raw or preprocessed dataset.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly LABELS=(
    exp1_1_1000
    exp1_1_1000plus
    exp32_2_1000
    exp1_to_exp48c_t025
    exp1_to_exp48d_t025
)
readonly DATASET_IDS=(650 651 652 661 662)

readonly ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?submit this file as its configured job array}"
(( ARRAY_INDEX >= 0 && ARRAY_INDEX < ${#LABELS[@]} )) || \
    fatal "array index is outside 0--4: $ARRAY_INDEX"

readonly LABEL="${LABELS[$ARRAY_INDEX]}"
readonly DATASET_ID="${DATASET_IDS[$ARRAY_INDEX]}"
readonly EVAL_ID="${EVAL_ID:-source_selection_synthmask525_seed42_euler100_pca2p323mm_try3}"
readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
: "${CLUSTER_BASE:=/cluster/work/${USER}}"

readonly EVAL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/${EVAL_ID}"
readonly MATCHED_ROOT="${EVAL_ROOT}/matched_common105"
readonly SYNTHETIC_DIR="${MATCHED_ROOT}/${LABEL}"
readonly COMMON_IDS_FILE="${MATCHED_ROOT}/common_candidate_ids.txt"
readonly REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
readonly NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

readonly TARGET_RAW="${nnUNet_raw}/Dataset${DATASET_ID}_BrainMet"

[[ -d "$REAL_DIR/train" && -d "$REAL_DIR/val" && -d "$REAL_DIR/test_new" ]] || \
    fatal "real train/val/test_new dataset is incomplete: $REAL_DIR"
[[ -d "$SYNTHETIC_DIR" ]] || fatal "matched synthetic dataset not found: $SYNTHETIC_DIR"
[[ -s "$COMMON_IDS_FILE" ]] || fatal "common candidate ID file not found: $COMMON_IDS_FILE"

mapfile -t COMMON_IDS < "$COMMON_IDS_FILE"
[[ ${#COMMON_IDS[@]} -eq 105 ]] || \
    fatal "common candidate ID file has ${#COMMON_IDS[@]} entries; expected 105"

unique_count="$(printf '%s\n' "${COMMON_IDS[@]}" | sort -u | wc -l)"
[[ "$unique_count" -eq 105 ]] || fatal "common candidate IDs are not unique"

case_count="$(find "$SYNTHETIC_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)"
[[ "$case_count" -eq 105 ]] || \
    fatal "$LABEL contains $case_count case directories; expected exactly 105"

for candidate_id in "${COMMON_IDS[@]}"; do
    [[ "$candidate_id" =~ ^[0-9]{5}$ ]] || fatal "invalid candidate ID: $candidate_id"
    [[ -s "$SYNTHETIC_DIR/$candidate_id/bravo.nii.gz" ]] || \
        fatal "$LABEL is missing BRAVO candidate $candidate_id"
    [[ -s "$SYNTHETIC_DIR/$candidate_id/seg.nii.gz" ]] || \
        fatal "$LABEL is missing segmentation candidate $candidate_id"
done

# A dataset ID is global in nnU-Net. Refuse both exact and differently named
# directories with the same ID so stale cases can never be mixed in.
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

echo "=== Convert matched common-105 dataset ==="
echo "Array task:  ${ARRAY_INDEX}/4"
echo "Generator:   ${LABEL}"
echo "Dataset ID:  ${DATASET_ID}"
echo "Synthetic:   ${SYNTHETIC_DIR}"
echo "Real:        ${REAL_DIR}"
echo "Output:      ${TARGET_RAW}"

python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "$REAL_DIR" \
    --synthetic-dir "$SYNTHETIC_DIR" \
    --nnunet-raw "$nnUNet_raw" \
    --dataset-id "$DATASET_ID" \
    --modality bravo

python - "$TARGET_RAW" "$COMMON_IDS_FILE" <<'PYEOF'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
common_ids = Path(sys.argv[2]).read_text().splitlines()
case_info = json.loads((target / "case_info.json").read_text())
dataset_info = json.loads((target / "dataset.json").read_text())

expected_synthetic = [f"BrainMetSyn_{candidate_id}" for candidate_id in common_ids]
assert case_info["synthetic_cases"] == expected_synthetic
assert len(case_info["real_train_cases"]) == 105
assert len(case_info["test_cases"]) == 51
assert case_info["modalities"] == ["bravo"]
assert dataset_info["numTraining"] == 210
assert dataset_info["channel_names"] == {"0": "BRAVO"}

for case_id in expected_synthetic:
    image = target / "imagesTr" / f"{case_id}_0000.nii.gz"
    label = target / "labelsTr" / f"{case_id}.nii.gz"
    assert image.is_file(), image
    assert label.is_file(), label

print(
    f"Validated {target.name}: 105 real train + 105 matched synthetic; "
    "51 held-out real cases"
)
PYEOF

cp -- "$COMMON_IDS_FILE" "$TARGET_RAW/common_candidate_ids.txt"
printf '%s\n' \
    "eval_id=${EVAL_ID}" \
    "generator=${LABEL}" \
    "dataset_id=${DATASET_ID}" \
    "synthetic_dir=${SYNTHETIC_DIR}" \
    > "$TARGET_RAW/source_selection.txt"

echo "Raw nnU-Net dataset ready: $TARGET_RAW"
echo "Planning/preprocessing is intentionally a separate later step."
