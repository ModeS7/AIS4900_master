#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="eval_all_metrics"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/eval_all_metrics_%j.out
#SBATCH --error=IDUN/output/eval/eval_all_metrics_%j.err

# Full generation-metric panel vs real_train, over the SAME 9 sets as the diversity /
# Med3D-MMD runs. Per dataset: FID/KID (ImageNet + RadImageNet), CMMD, Med3D-MMD, PCA
# brain-shape pass-rate + error, MS-SSIM diversity. Point estimates on the loaded pool
# (N shown); for equal-N bootstrap CIs use eval_diversity.slurm / eval_med3d_mmd.slurm.
#
# gpu80g: Med3D runs on native 256x256x150 whole volumes (heavy). If it OOMs, lower
# --med3d-batch (already 2). PCA threshold in the npz is the known-buggy 0.0196 (156-vs-181
# sample bug) — compare the mean reconstruction error (threshold-free), not just pass%.

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
REAL="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN="${CLUSTER_BASE}/MedicalDataSets/generated"
SEGC="${GEN}/seg_candidates_525"

echo "=== All-metrics panel | Job ${SLURM_JOB_ID} | Node ${SLURM_JOB_NODELIST} ==="
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.eval_all_metrics \
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
    --trim-slices 0 \
    --med3d-batch 2 \
    --pca-model data/brain_pca_256x256x160.npz \
    --pca-threshold 0.05 \
    --seed 42 \
    --output runs/eval/all_metrics_${SLURM_JOB_ID}.json
