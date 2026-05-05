#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="nnunet_segcand"
#SBATCH --output=IDUN/output/train/downstream/nnunet/convert_seg_candidates_%j.out
#SBATCH --error=IDUN/output/train/downstream/nnunet/convert_seg_candidates_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Build the full set of nnU-Net datasets for the segmentation downstream
# evaluation. Wipes the existing nnU-Net raw + preprocessed dirs and rebuilds:
#
# Real-only baselines:
#   600  bravo  (single channel)
#   601  dual   (t1_pre + t1_gd)
#   602  triple (t1_pre + t1_gd + flair)
#
# Real + synthetic BRAVO (each pairs real bravo with one 525-vol synth set):
#   610  + exp1_1_1000_bravo_525                       (kept old baseline)
#   611  + seg_candidates_525/exp1_1_1000plus
#   612  + seg_candidates_525/exp32_2_1000
#   613  + seg_candidates_525/exp48c_standalone
#   614  + seg_candidates_525/exp48c_handoff_exp32
#   615  + seg_candidates_525/exp48d_handoff_exp1
#
# Real + synthetic DUAL:
#   620  + exp1v2_2_1000_dual_525

set -e
cd ${SLURM_SUBMIT_DIR}

echo "=== nnU-Net dataset rebuild for seg-candidate sweep ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
conda activate nnunet

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

NNUNET_BASE="${CLUSTER_BASE}/MedicalDataSets/nnunet"
NNUNET_RESULTS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
REAL_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN_ROOT="${CLUSTER_BASE}/MedicalDataSets/generated"

export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_RESULTS}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# === Wipe any existing datasets ============================================
echo ""
echo "=== Wiping existing nnU-Net raw + preprocessed dirs ==="
rm -rf "${nnUNet_raw}"/Dataset*
rm -rf "${nnUNet_preprocessed}"/Dataset*
echo "Done."

# === Helper ================================================================
# Args: <id> <modality> [synthetic_dir]
convert_one () {
    local DID="$1"
    local MOD="$2"
    local SYN="${3:-}"
    local SYN_ARG=""
    if [ -n "$SYN" ]; then
        local N=$(find "$SYN" -maxdepth 1 -type d -name "[0-9]*" 2>/dev/null | wc -l)
        echo ""
        echo "--- Dataset ${DID} | modality=${MOD} | synth=${SYN} (${N} vols) ---"
        if [ "$N" -lt 100 ]; then
            echo "  WARN: only ${N} synth volumes — proceeding anyway"
        fi
        SYN_ARG="--synthetic-dir ${SYN}"
    else
        echo ""
        echo "--- Dataset ${DID} | modality=${MOD} | real-only baseline ---"
    fi

    python -m medgen.downstream.nnunet.convert_dataset \
        --real-dir "${REAL_DIR}" \
        ${SYN_ARG} \
        --nnunet-raw "${nnUNet_raw}" \
        --dataset-id "${DID}" \
        --modality "${MOD}"
}

# === Step 1: convert all datasets =========================================
echo ""
echo "=========================================="
echo "  Step 1: Converting datasets"
echo "=========================================="

# Real-only baselines
convert_one 600 bravo
convert_one 601 dual
convert_one 602 triple

# Real + synth BRAVO
convert_one 610 bravo "${GEN_ROOT}/exp1_1_1000_bravo_525"
convert_one 611 bravo "${GEN_ROOT}/seg_candidates_525/exp1_1_1000plus"
convert_one 612 bravo "${GEN_ROOT}/seg_candidates_525/exp32_2_1000"
convert_one 613 bravo "${GEN_ROOT}/seg_candidates_525/exp48c_standalone"
convert_one 614 bravo "${GEN_ROOT}/seg_candidates_525/exp48c_handoff_exp32"
convert_one 615 bravo "${GEN_ROOT}/seg_candidates_525/exp48d_handoff_exp1"

# Real + synth DUAL
convert_one 620 dual "${GEN_ROOT}/exp1v2_2_1000_dual_525"

# === Step 2: plan and preprocess ==========================================
echo ""
echo "=========================================="
echo "  Step 2: nnU-Net plan and preprocess"
echo "=========================================="

DATASETS="600 601 602 610 611 612 613 614 615 620"

for DID in $DATASETS; do
    echo ""
    echo "--- plan + preprocess Dataset${DID} ---"
    nnUNetv2_plan_and_preprocess -d ${DID} --verify_dataset_integrity -c 3d_fullres -np 4
    nnUNetv2_plan_and_preprocess -d ${DID} -pl nnUNetPlannerResEncL -c 3d_fullres -np 4
done

# === Step 3: validation ====================================================
echo ""
echo "=========================================="
echo "  Step 3: Validation summary"
echo "=========================================="

python - <<'PYEOF'
import json, os
import numpy as np

base_pp = os.environ['nnUNet_preprocessed']
labels = {
    600: 'bravo  (real only)',
    601: 'dual   (real only)',
    602: 'triple (real only)',
    610: 'bravo + exp1_1_1000_bravo_525',
    611: 'bravo + seg_candidates_525/exp1_1_1000plus',
    612: 'bravo + seg_candidates_525/exp32_2_1000',
    613: 'bravo + seg_candidates_525/exp48c_standalone',
    614: 'bravo + seg_candidates_525/exp48c_handoff_exp32',
    615: 'bravo + seg_candidates_525/exp48d_handoff_exp1',
    620: 'dual   + exp1v2_2_1000_dual_525',
}

for did, label in labels.items():
    pp_dir = next(
        (os.path.join(base_pp, e) for e in os.listdir(base_pp)
         if e.startswith(f'Dataset{did}_')),
        None,
    )
    if not pp_dir:
        print(f'Dataset{did} ({label}) — MISSING')
        continue
    print(f'Dataset{did} ({label})')
    for plans_name in ('nnUNetPlans.json', 'nnUNetResEncUNetLPlans.json'):
        p = os.path.join(pp_dir, plans_name)
        if not os.path.exists(p):
            continue
        plans = json.load(open(p))
        for cfg_name, cfg in plans.get('configurations', {}).items():
            print(
                f'  {plans_name}: {cfg_name}: '
                f'patch={cfg.get("patch_size")} '
                f'batch={cfg.get("batch_size")} '
                f'spacing={cfg.get("spacing")}'
            )
    for cfg_dir in os.listdir(pp_dir):
        cfg_path = os.path.join(pp_dir, cfg_dir)
        if not os.path.isdir(cfg_path):
            continue
        npz_files = sorted(f for f in os.listdir(cfg_path) if f.endswith('.npz'))
        if not npz_files:
            continue
        empty = sum(
            1 for f in npz_files
            if (np.load(os.path.join(cfg_path, f)).get('seg', np.array([1])) > 0).sum() == 0
        )
        print(f'  {cfg_dir}: {len(npz_files)} samples, {empty} empty segs')
    print()
PYEOF

echo ""
echo "=== Done. ${#DATASETS[@]} datasets ready. ==="
