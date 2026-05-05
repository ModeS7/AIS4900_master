#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nnunet_finish"
#SBATCH --output=IDUN/output/train/downstream/nnunet/finish_seg_candidates_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/finish_seg_candidates_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Resume + verify the seg-candidate nnU-Net preprocess that failed inside
# Dataset620 (blosc2 set_slice error). Re-runs preprocess for any dataset
# that is missing a planner JSON or has fewer .b2nd shards than expected.

set -e
cd ${SLURM_SUBMIT_DIR}

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"

DATASETS="600 601 602 610 611 612 613 614 615 620"

# Free disk check — abort if <50 GB left in work
FREE_GIB=$(df -BG --output=avail "${NNUNET_BASE}" | tail -1 | tr -dc '0-9')
echo "Free disk on ${NNUNET_BASE}: ${FREE_GIB} GiB"
[ "$FREE_GIB" -lt 50 ] && { echo "ERROR: only ${FREE_GIB} GiB free, need >=50."; exit 78; }

needs_default=()
needs_resenc=()

for DID in $DATASETS; do
    PP_DIR=$(ls -d "${nnUNet_preprocessed}"/Dataset${DID}_* 2>/dev/null | head -1)
    if [ -z "$PP_DIR" ]; then
        echo "Dataset${DID}: missing preprocessed dir → both planners needed"
        needs_default+=("$DID"); needs_resenc+=("$DID"); continue
    fi
    [ -f "${PP_DIR}/nnUNetPlans.json" ] || needs_default+=("$DID")
    [ -f "${PP_DIR}/nnUNetResEncUNetLPlans.json" ] || needs_resenc+=("$DID")
done

# Always rerun ResEnc preprocess for Dataset620 since that's where the crash
# occurred (the JSON may have been written but the .b2nd cache is partial).
case " ${needs_resenc[*]} " in
    *" 620 "*) ;;
    *) needs_resenc+=("620") ;;
esac

# Wipe partial Dataset620 ResEnc cache so the rerun has a clean slate
RES_DIR_620=$(ls -d "${nnUNet_preprocessed}"/Dataset620_*/nnUNetPlans_3d_fullres 2>/dev/null | head -1 | sed 's|nnUNetPlans_3d_fullres|nnUNetResEncUNetL_3d_fullres|')
if [ -n "$RES_DIR_620" ] && [ -d "$RES_DIR_620" ]; then
    echo "Removing partial ${RES_DIR_620}"
    rm -rf "$RES_DIR_620"
fi

echo ""
echo "Default planner needed for: ${needs_default[*]:-(none)}"
echo "ResEnc-L planner needed for: ${needs_resenc[*]:-(none)}"

for DID in "${needs_default[@]}"; do
    echo ""
    echo "--- default planner for Dataset${DID} ---"
    nnUNetv2_plan_and_preprocess -d ${DID} --verify_dataset_integrity -c 3d_fullres -np 4
done

for DID in "${needs_resenc[@]}"; do
    echo ""
    echo "--- ResEnc-L planner for Dataset${DID} ---"
    nnUNetv2_plan_and_preprocess -d ${DID} -pl nnUNetPlannerResEncL -c 3d_fullres -np 4
done

echo ""
echo "=== Final validation summary ==="
python - <<'PYEOF'
import os, json
base_pp = os.environ['nnUNet_preprocessed']
expected = [600, 601, 602, 610, 611, 612, 613, 614, 615, 620]

for did in expected:
    pp = next((os.path.join(base_pp, e) for e in os.listdir(base_pp)
               if e.startswith(f'Dataset{did}_')), None)
    if not pp:
        print(f'Dataset{did} — MISSING'); continue
    have = []
    for plans in ('nnUNetPlans.json', 'nnUNetResEncUNetLPlans.json'):
        if os.path.exists(os.path.join(pp, plans)):
            have.append(plans.replace('.json', ''))
    cfgs = []
    for cfg in sorted(os.listdir(pp)):
        cdir = os.path.join(pp, cfg)
        if not os.path.isdir(cdir): continue
        n_b2nd = sum(1 for f in os.listdir(cdir) if f.endswith('.b2nd'))
        if n_b2nd > 0:
            cfgs.append(f'{cfg}={n_b2nd}')
    print(f'Dataset{did}: plans=[{", ".join(have)}] caches=[{", ".join(cfgs)}]')
PYEOF

echo ""
echo "=== Done. ==="
