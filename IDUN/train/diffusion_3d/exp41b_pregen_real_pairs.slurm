#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="pregen"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/pregen_restoration_pairs_%j.out
#SBATCH --error=IDUN/output/eval/pregen_restoration_pairs_%j.err

# exp41b Phase 1: Pre-generate exp1_1_1000 outputs on the TRAIN split to
# build real (clean, generated) pairs for IR-SDE restoration training.
#
# For each train subject:
#   clean_bravo.nii.gz  ← copy of real/train/<subj>/bravo.nii.gz
#   seg.nii.gz          ← copy of real/train/<subj>/seg.nii.gz
#   degraded_000.nii.gz ← exp1_1_1000 generation seeded from this subj's seg
#   degraded_001.nii.gz ← different seed, same seg
#   degraded_002.nii.gz ← different seed, same seg
#
# Resumable: subjects with all 3 variants are skipped on re-run.
#
# Cost: ~105 train subjects × 3 variants × ~45s ≈ 2.5h on A100.

cd ${SLURM_SUBMIT_DIR}

echo "=== Pre-generate restoration pairs on train split ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"
nvidia-smi

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE_MIB" -lt 60000 ]; then
    echo "ERROR: Only ${FREE_MIB} MiB free on GPU. Aborting."
    exit 78
fi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

CHECKPOINT="/cluster/work/modestas/AIS4900_master/runs/diffusion_3d/bravo/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt"
DATA_ROOT="/cluster/work/modestas/MedicalDataSets/brainmetshare-3/train"
OUTPUT_ROOT="/cluster/work/modestas/MedicalDataSets/restoration_pairs_real/train"

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: train data dir not found: ${DATA_ROOT}"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

time python -m medgen.scripts.pregen_restoration_pairs \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --num-steps 32 \
    --num-variants 3 \
    --depth 160 \
    --seed-base 42

# Also run on val split (smaller — for validation during restoration training)
OUTPUT_ROOT_VAL="/cluster/work/modestas/MedicalDataSets/restoration_pairs_real/val"
DATA_ROOT_VAL="/cluster/work/modestas/MedicalDataSets/brainmetshare-3/val"
if [ -d "${DATA_ROOT_VAL}" ]; then
    mkdir -p "${OUTPUT_ROOT_VAL}"
    echo ""
    echo "=== Pre-generating val split too ==="
    time python -m medgen.scripts.pregen_restoration_pairs \
        --checkpoint "${CHECKPOINT}" \
        --data-root "${DATA_ROOT_VAL}" \
        --output-root "${OUTPUT_ROOT_VAL}" \
        --num-steps 32 \
        --num-variants 2 \
        --depth 160 \
        --seed-base 4200
fi

echo ""
echo "✓ Pre-generation complete."
echo "Train pairs at: ${OUTPUT_ROOT}"
echo "Val pairs at:   ${OUTPUT_ROOT_VAL}"
