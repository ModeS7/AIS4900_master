#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="eval_all_metrics"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/eval_all_metrics_%j.out
#SBATCH --error=IDUN/output/eval/eval_all_metrics_%j.err

# Full metric panel for the synthetic sets vs THREE references: test(51), train(105),
# and all(156)=train+test (--combine-all). Per (dataset, reference): FID/KID (ImageNet +
# RadImageNet), CMMD, Med3D-MMD. Reference-independent (once): PCA pass%+error, MS-SSIM
# diversity. Generated features extracted ONCE and reused across references.
#
# WHY 3 refs: step counts were tuned to FID-vs-TEST, downstream is scored on TEST, but the
# first panel scored vs TRAIN — FID is reference-sensitive, so the metric<->downstream story
# must be checked against the matching reference. NOTE: 'all' contains test, so real_test vs
# 'all' is partly self (floor row, read with that caveat).
#
# ALL 525 synthetic volumes used (--pool-cap 0). Diversity is O(N^2) → --diversity-cap 256
# (N-stable, verified). Med3D native 256x256x150 → gpu80g + --med3d-batch 2.

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
REAL="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN="${CLUSTER_BASE}/MedicalDataSets/generated"
SEGC="${GEN}/seg_candidates_525"

echo "=== All-metrics panel (3 refs, all 525) | Job ${SLURM_JOB_ID} | Node ${SLURM_JOB_NODELIST} ==="
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.eval_all_metrics \
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
    --pattern bravo.nii.gz \
    --target-shape 150 256 256 \
    --pool-cap 0 \
    --diversity-cap 256 \
    --trim-slices 0 \
    --med3d-batch 2 \
    --pca-model data/brain_pca_256x256x160.npz \
    --pca-threshold 0.05 \
    --seed 42 \
    --output runs/eval/all_metrics_3ref_${SLURM_JOB_ID}.json
