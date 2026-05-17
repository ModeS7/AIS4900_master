#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_d603"
#SBATCH --output=IDUN/output/train/downstream/nnunet/convert_d603_quad_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/convert_d603_quad_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Convert BrainMetShare-3 real data with all four modalities (t1_pre + t1_gd +
# flair + bravo) to nnU-Net format as Dataset 603, then plan/preprocess for
# both vanilla nnUNetPlans and ResEncL plans.
#
# Used by exp345_baseline_quad_d603.slurm. This mirrors Grøvik 2020 and
# Ottesen 2022's 4-channel input setup, so the resulting model is the right
# apples-to-apples comparison for the published multi-modal Dice numbers.

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

set -e

cd ${SLURM_SUBMIT_DIR}

echo "=== nnU-Net Dataset 603 (quad-modal) conversion ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_RESULTS}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# Clean any previous Dataset 603 so we don't ship stale plans into training.
echo ""
echo "=== Cleaning previous Dataset 603 ==="
rm -rf "${nnUNet_raw}/Dataset603_BrainMet"
rm -rf "${nnUNet_preprocessed}/Dataset603_BrainMet"

# Step 1: Convert (real data only; 4-channel quad preset = t1_pre, t1_gd, flair, bravo)
echo ""
echo "=== Converting Dataset 603 (quad: t1_pre + t1_gd + flair + bravo) ==="
python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "${REAL_DIR}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 603 \
    --modality quad

# Step 2: Plan + preprocess (both vanilla and ResEncL plans)
echo ""
echo "=== Plan + preprocess Dataset 603 ==="
nnUNetv2_plan_and_preprocess -d 603 --verify_dataset_integrity -c 3d_fullres -np 4
nnUNetv2_plan_and_preprocess -d 603 -pl nnUNetPlannerResEncL -c 3d_fullres -np 4

# Step 3: Sanity print the resulting plans (patch size, batch size, channel count)
echo ""
echo "=== Dataset 603 plan summary ==="
python -c "
import json, os
pp_base = os.environ['nnUNet_preprocessed']
pp_dir = next(
    os.path.join(pp_base, e) for e in os.listdir(pp_base)
    if e.startswith('Dataset603_')
)
for plans_name in ('nnUNetPlans.json', 'nnUNetResEncUNetLPlans.json'):
    plans_path = os.path.join(pp_dir, plans_name)
    if not os.path.exists(plans_path):
        continue
    with open(plans_path) as f:
        plans = json.load(f)
    print(f'  {plans_name}:')
    cfg = plans['configurations']['3d_fullres']
    print(f'    patch_size: {cfg[\"patch_size\"]}')
    print(f'    batch_size: {cfg[\"batch_size\"]}')
    print(f'    spacing:    {cfg[\"spacing\"]}')
    print(f'    norm:       {cfg[\"normalization_schemes\"]}')
"

echo ""
echo "=== Done. Submit exp345_baseline_quad_d603.slurm next. ==="
