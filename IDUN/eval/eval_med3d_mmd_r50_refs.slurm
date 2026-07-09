#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="eval_med3d_mmd_r50_refs"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/eval_med3d_mmd_r50_refs_%j.out
#SBATCH --error=IDUN/output/eval/eval_med3d_mmd_r50_refs_%j.err

# FINAL Med3D-MMD run: ResNet-50 ONLY, each synthetic set vs THREE real references
# (test-51 / train-105 / all-156 via --combine-all). LOWER = closer to real. Resolves the
# reference-confound (step counts tuned vs test, downstream on test, panel naively vs train)
# by scoring all three at once; dataset features are extracted ONCE and reused across refs.
#
# NATIVE 256x256x150 (no --med3d-input-size) — matches the training-time
# Generation_3d/MED3D_MMD metric and the committed eval_med3d_mmd.slurm, so these numbers
# are directly comparable. r50 fits at native 256 on gpu80g/gpu40g (the 128^3 downsize was
# only needed for the deep-backbone sweep). All 525 synthetic volumes used (--pool-cap 0).
#
# r50 weights are already HF-cached, so NO login-node pre-cache is required here (unlike the
# 7-backbone sweep). real_test is scored too -> the real-vs-real MMD floor (~0 vs test ref).

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
REAL="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN="${CLUSTER_BASE}/MedicalDataSets/generated"
SEGC="${GEN}/seg_candidates_525"

echo "=== Med3D-MMD r50 vs test/train/all | Job ${SLURM_JOB_ID} | Node ${SLURM_JOB_NODELIST} ==="
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.eval_med3d_mmd_networks \
    --reference "test:${REAL}/test1" \
    --reference "train:${REAL}/train" \
    --combine-all \
    --dataset "real_test:${REAL}/test1" \
    --dataset "exp1_1_imagenet:${GEN}/exp1_1_bravo_imagenet_525_brain_masked" \
    --dataset "exp1_1_radimagenet:${GEN}/exp1_1_bravo_radimagenet_525_brain_masked" \
    --dataset "seg_cand_exp1_1_1000plus:${SEGC}/exp1_1_1000plus" \
    --dataset "seg_cand_exp32_2_1000:${SEGC}/exp32_2_1000" \
    --dataset "seg_cand_exp48c_standalone:${SEGC}/exp48c_standalone" \
    --dataset "seg_cand_exp48c_handoff_exp32:${SEGC}/exp48c_handoff_exp32" \
    --dataset "seg_cand_exp48d_handoff_exp1:${SEGC}/exp48d_handoff_exp1" \
    --networks 50 \
    --pattern bravo.nii.gz \
    --target-shape 150 256 256 \
    --pool-cap 0 \
    --subsample-n 51 \
    --bootstrap 100 \
    --batch-size 2 \
    --seed 42 \
    --output runs/eval/med3d_mmd_r50_refs_${SLURM_JOB_ID}.json
