#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="eval_diversity_msssim"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/eval_diversity_msssim_%j.out
#SBATCH --error=IDUN/output/eval/eval_diversity_msssim_%j.err

# MS-SSIM-3D sample-to-sample diversity over multiple volume sets.
#   Higher = more diverse (less mode collapse). Same-slice pairwise (1 - MS-SSIM),
#   averaged over depth and all pairs. Sets are compared at a COMMON subsample N
#   (=51, the official test size) with bootstrap CIs, so the unequal set sizes
#   (105 / 51 / 525) do not confound the ranking. Raw full-pool numbers are also
#   reported (labelled, N-dependent).
#
# Datasets (all loaded as bravo.nii.gz images):
#   real_train          105  brainmetshare-3/train  (native 256x256x150)
#   real_test            51  brainmetshare-3/test1  (official Grovik hold-out)
#   exp1_1_imagenet     525  generated, brain-masked
#   exp1_1_radimagenet  525  generated, brain-masked
#   seg_cand_* (x5)     525  seg_candidates_525/{exp1_1_1000plus, exp32_2_1000,
#                            exp48c_standalone, exp48c_handoff_exp32, exp48d_handoff_exp1}
#   NB: despite the "seg_candidates" name, these dirs hold generated BRAVO volumes
#       (convert_seg_candidates.slurm loads them with modality=bravo).

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
REAL="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN="${CLUSTER_BASE}/MedicalDataSets/generated"
SEGC="${GEN}/seg_candidates_525"

echo "=== MS-SSIM-3D diversity | Job ${SLURM_JOB_ID} | Node ${SLURM_JOB_NODELIST} ==="
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

export PYTORCH_ALLOC_CONF=expandable_segments:True

time python -m medgen.scripts.eval_diversity \
    --dataset "real_train:${REAL}/train" \
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
    --bootstrap 30 \
    --depth-stride 1 \
    --pair-chunk 512 \
    --metric msssim \
    --seed 42 \
    --output IDUN/output/eval/diversity_msssim_${SLURM_JOB_ID}.json
