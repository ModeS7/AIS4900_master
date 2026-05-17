#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_d630_st25"
#SBATCH --output=IDUN/output/train/downstream/nnunet/convert_exp8_d630_size_targeted_25_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/convert_exp8_d630_size_targeted_25_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# exp8 family — size-targeted synthetic mixing.
# This SLURM builds Dataset 630 = 105 real + 25 size-targeted synth.
# Selection strategy: tiny-bucket volumes first; top up with small / medium /
# large if the tiny bucket runs out (priority order: tiny > small > medium > large).
#
# Hypothesis: tiny lesions are 54% of GT but only 36% detected by exp3.
# Pumping in synth that contains predominantly tiny lesions should boost
# tiny-lesion detection without overfitting to large-lesion patterns.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

set -e
cd ${SLURM_SUBMIT_DIR}

echo "=== exp8 d630 size-targeted N=25 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN_ROOT="${CLUSTER_BASE}/MedicalDataSets/generated"
POOL_DIR="${GEN_ROOT}/seg_candidates_525/exp48c_handoff_exp32"
MANIFEST="${GEN_ROOT}/exp8_size_manifest_exp48c_handoff_exp32.json"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_RESULTS}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# Step 1: build manifest (one-shot, reused by all exp8_* convert jobs)
if [ ! -f "${MANIFEST}" ]; then
    echo ""
    echo "=== Classifying synth pool by max lesion size ==="
    python -m medgen.scripts.classify_synth_by_lesion_size \
        --pool-dir "${POOL_DIR}" \
        --output "${MANIFEST}"
else
    echo "=== Manifest exists, reusing ${MANIFEST} ==="
fi

# Step 2: pick 25 vols (seed=1 → exp8_1 has its own subset, independent of other exp8_N)
SUBSET_DIR="${GEN_ROOT}/exp8_subsets/exp48c_handoff_exp32_st25_seed1"
echo ""
echo "=== Picking 25 size-targeted synth vols (seed=1) ==="
python -m medgen.scripts.pick_synth_subset_by_lesion_size \
    --manifest "${MANIFEST}" \
    --n 25 \
    --seed 1 \
    --output-dir "${SUBSET_DIR}" \
    --overwrite \
    --save-selection "${SUBSET_DIR}_selection.json"

# Step 3: convert to nnU-Net format (mirrors convert_seg_candidates.slurm pattern)
echo ""
echo "=== Cleaning previous Dataset 630 ==="
rm -rf "${nnUNet_raw}/Dataset630_BrainMet"
rm -rf "${nnUNet_preprocessed}/Dataset630_BrainMet"

echo ""
echo "=== Converting Dataset 630 (105 real + 25 size-targeted synth, bravo) ==="
python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "${REAL_DIR}" \
    --synthetic-dir "${SUBSET_DIR}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 630 \
    --modality bravo

# Step 4: plan and preprocess
echo ""
echo "=== Plan + preprocess Dataset 630 ==="
nnUNetv2_plan_and_preprocess -d 630 --verify_dataset_integrity -c 3d_fullres -np 4
nnUNetv2_plan_and_preprocess -d 630 -pl nnUNetPlannerResEncL -c 3d_fullres -np 4

echo ""
echo "=== Done. Submit exp8_1_size_targeted_25syn_d630.slurm next. ==="
