#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nn_add_1k"
#SBATCH --output=IDUN/output/train/downstream/nnunet/convert_exp1_1_1000_brain_masked_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/convert_exp1_1_1000_brain_masked_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# ADDITIVE convert + preprocess: adds Datasets 616 (RadImageNet-opt) and 617
# (ImageNet-opt) to the existing nnU-Net setup, both built from BRAIN-MASKED
# exp1_1_1000 generations. Also deletes the unused Group B Datasets 610, 620.
#
# Existing Datasets 600-615 are NOT touched. The 5 trained candidates and
# their fold checkpoints stay intact.
#
# Prereq: brain_mask_existing.slurm must have produced
#   ${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525_brain_masked/
#   ${CLUSTER_BASE}/MedicalDataSets/generated/exp1_1_bravo_imagenet_525_brain_masked/

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

echo "=== ADDITIVE: add Datasets 616 (RIN) + 617 (IN), drop 610 + 620 ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN_ROOT="${CLUSTER_BASE}/MedicalDataSets/generated"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"

# === Sanity: source brain-masked dirs must exist ============================
SRC_RIN="${GEN_ROOT}/exp1_1_bravo_radimagenet_525_brain_masked"
SRC_IN="${GEN_ROOT}/exp1_1_bravo_imagenet_525_brain_masked"
for D in "$SRC_RIN" "$SRC_IN"; do
    if [ ! -d "$D" ]; then
        echo "ERROR: brain-masked source missing: $D"
        echo "Run brain_mask_existing.slurm first."
        exit 2
    fi
    N=$(find "$D" -maxdepth 1 -type d -name '[0-9]*' | wc -l)
    echo "  $D : $N volumes"
    [ "$N" -lt 100 ] && { echo "ERROR: too few volumes in $D"; exit 3; }
done

# === Pre-state: confirm trained datasets are intact =========================
echo ""
echo "--- Pre-state: existing trained datasets ---"
for DID in 611 612 613 614 615; do
    if [ -d "${nnUNet_preprocessed}"/Dataset${DID}_* ]; then
        echo "  Dataset${DID}: present"
    else
        echo "  Dataset${DID}: MISSING — abort"
        exit 4
    fi
done

# === Drop unused Group B Datasets 610, 620 ==================================
echo ""
echo "--- Removing Group B datasets 610 + 620 ---"
for DID in 610 620; do
    for ROOT in "${nnUNet_raw}" "${nnUNet_preprocessed}"; do
        for D in "$ROOT"/Dataset${DID}_*; do
            if [ -d "$D" ]; then
                echo "  rm -rf $D"
                rm -rf "$D"
            fi
        done
    done
done

# === Convert Dataset 616 (real bravo + RIN-opt brain-masked) ================
echo ""
echo "--- Convert Dataset 616: real + RIN-opt brain-masked ---"
python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "${REAL_DIR}" \
    --synthetic-dir "${SRC_RIN}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 616 \
    --modality bravo

# === Convert Dataset 617 (real bravo + IN-opt brain-masked) =================
echo ""
echo "--- Convert Dataset 617: real + IN-opt brain-masked ---"
python -m medgen.downstream.nnunet.convert_dataset \
    --real-dir "${REAL_DIR}" \
    --synthetic-dir "${SRC_IN}" \
    --nnunet-raw "${nnUNet_raw}" \
    --dataset-id 617 \
    --modality bravo

# === Plan + preprocess (default + ResEnc-L) for 616 and 617 only ============
for DID in 616 617; do
    echo ""
    echo "--- plan + preprocess Dataset${DID} ---"
    nnUNetv2_plan_and_preprocess -d ${DID} --verify_dataset_integrity -c 3d_fullres -np 4
    nnUNetv2_plan_and_preprocess -d ${DID} -pl nnUNetPlannerResEncL -c 3d_fullres -np 4
done

# === Verify: 600-615 still intact, 610/620 gone, 616/617 added ==============
echo ""
echo "=== Final dataset summary ==="
python - <<'PYEOF'
import os, json
base_pp = os.environ['nnUNet_preprocessed']
expected_present = [600, 601, 602, 611, 612, 613, 614, 615, 616, 617]
expected_absent = [610, 620]

print("\nExpected present:")
for did in expected_present:
    pp = next((os.path.join(base_pp, e) for e in os.listdir(base_pp)
               if e.startswith(f'Dataset{did}_')), None)
    if not pp:
        print(f'  Dataset{did}: MISSING — abort')
        raise SystemExit(1)
    cfgs = []
    for cfg in sorted(os.listdir(pp)):
        cdir = os.path.join(pp, cfg)
        if not os.path.isdir(cdir): continue
        n = sum(1 for f in os.listdir(cdir) if f.endswith('.b2nd'))
        if n > 0: cfgs.append(f'{cfg}={n}')
    plans = [p for p in ('nnUNetPlans.json', 'nnUNetResEncUNetLPlans.json')
             if os.path.exists(os.path.join(pp, p))]
    print(f'  Dataset{did}: plans={plans} caches=[{", ".join(cfgs)}]')

print("\nExpected absent:")
for did in expected_absent:
    pp = next((os.path.join(base_pp, e) for e in os.listdir(base_pp)
               if e.startswith(f'Dataset{did}_')), None)
    if pp:
        print(f'  Dataset{did}: STILL PRESENT (should have been deleted) — abort')
        raise SystemExit(1)
    print(f'  Dataset{did}: removed ✓')

# Plan-architecture compatibility check: 616, 617 should match Group A signature
print("\nGroup A compatibility check (616/617 vs 600):")
for did in (600, 616, 617):
    pp = next(os.path.join(base_pp, e) for e in os.listdir(base_pp)
              if e.startswith(f'Dataset{did}_'))
    plans = json.load(open(os.path.join(pp, 'nnUNetResEncUNetLPlans.json')))
    fr = plans['configurations']['3d_fullres']
    print(f'  Dataset{did}: patch={fr["patch_size"]} use_mask_for_norm={fr.get("use_mask_for_norm")}')
PYEOF

echo ""
echo "=== Done. Datasets 616 + 617 added. Datasets 610 + 620 removed. ==="
