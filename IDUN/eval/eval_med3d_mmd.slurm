#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="eval_med3d_mmd"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/eval_med3d_mmd_%j.out
#SBATCH --error=IDUN/output/eval/eval_med3d_mmd_%j.err

# Med3D whole-volume Gaussian-MMD of each generated set vs the real reference
# (real_train), over the SAME sets as eval_diversity_msssim.slurm. LOWER = closer to
# the real distribution. Same primitives as training's Generation_3d/MED3D_MMD:
# Med3DFeatures (input_size=None, native res) + compute_cmmd (RBF, median bandwidth).
# Bandwidth is fixed once from the reference so datasets are directly comparable.
# real_test is scored too -> the real-vs-real MMD floor (should be ~0).
#
# NOTE: Med3D on native 256x256x150 through a 3D ResNet-50 is memory-heavy. If this
# OOMs, add e.g. --med3d-input-size 128 128 128 (Med3D adaptive-pools any size) and/or
# lower --batch-size. Feature extraction is the cost; the 100-draw bootstrap is cheap.

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
REAL="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN="${CLUSTER_BASE}/MedicalDataSets/generated"
SEGC="${GEN}/seg_candidates_525"

echo "=== Med3D-MMD | Job ${SLURM_JOB_ID} | Node ${SLURM_JOB_NODELIST} ==="
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.eval_med3d_mmd \
    --reference "real_train:${REAL}/train" \
    --dataset "real_test:${REAL}/test1" \
    --dataset "exp1_1_imagenet:${GEN}/exp1_1_bravo_imagenet_525_brain_masked" \
    --dataset "exp1_1_radimagenet:${GEN}/exp1_1_bravo_radimagenet_525_brain_masked" \
    --dataset "seg_cand_exp1_1_1000plus:${SEGC}/exp1_1_1000plus" \
    --dataset "seg_cand_exp32_2_1000:${SEGC}/exp32_2_1000" \
    --dataset "seg_cand_exp48c_standalone:${SEGC}/exp48c_standalone" \
    --dataset "seg_cand_exp48c_handoff_exp32:${SEGC}/exp48c_handoff_exp32" \
    --dataset "seg_cand_exp48d_handoff_exp1:${SEGC}/exp48d_handoff_exp1" \
    --pattern bravo.nii.gz \
    --target-shape 150 256 256 \
    --pool-cap 256 \
    --subsample-n 51 \
    --bootstrap 100 \
    --batch-size 4 \
    --seed 42 \
    --output IDUN/output/eval/med3d_mmd_${SLURM_JOB_ID}.json
