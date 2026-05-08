#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="bm_exist"
#SBATCH --output=IDUN/output/eval/brain_mask_existing_%j.out
#SBATCH --error=IDUN/output/eval/brain_mask_existing_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Apply brain-mask zeroing to existing exp1_1_1000 generations (RIN-opt and IN-opt).
# Reads from MedicalDataSets/generated/exp1_1_bravo_{radimagenet,imagenet}_525/
# and writes to *_brain_masked/ side directories.
# Pure CPU post-processing — no diffusion sampling.

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

echo "=== Brain-mask existing exp1_1_1000 generations ==="
echo "Job: ${SLURM_JOB_ID} | Node: ${SLURM_JOB_NODELIST}"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

GEN_ROOT="${CLUSTER_BASE}/MedicalDataSets/generated"

run_one () {
    local NAME="$1"
    local SRC="${GEN_ROOT}/${NAME}"
    local DST="${GEN_ROOT}/${NAME}_brain_masked"
    if [ ! -d "$SRC" ]; then
        echo "ERROR: source missing: $SRC"
        return 1
    fi
    echo ""
    echo "--- ${NAME} ---"
    echo "  src: $SRC"
    echo "  dst: $DST"
    time python -m medgen.scripts.brain_mask_existing \
        --src "$SRC" \
        --dst "$DST" \
        --threshold 0.05 \
        --dilate-pixels 2
    echo "  done. count: $(find "$DST" -maxdepth 1 -type d -name '[0-9]*' | wc -l)"
}

run_one "exp1_1_bravo_radimagenet_525"
run_one "exp1_1_bravo_imagenet_525"

echo ""
echo "=== Done. ==="
