#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-10:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="eval_diversity_lpips"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/eval_diversity_lpips_%j.out
#SBATCH --error=IDUN/output/eval/eval_diversity_lpips_%j.err

# Sample-to-sample diversity via TRUE LPIPS (AlexNet backbone) over the SAME 9 sets
# as eval_diversity_msssim.slurm. Diversity = mean pairwise LPIPS DISTANCE (higher =
# more diverse; NOT 1-x). Compared at common N=51 with bootstrap CIs.
#
# network_type=alex => LPIPS with AlexNet, the LPIPS paper's DEFAULT metric backbone
# (Zhang et al. 2018) and ~23x faster than VGG here. vgg => LPIPS-VGG (common in
# generative-eval reporting, and the authors' choice for LPIPS-as-loss).
# radimagenet_resnet50 => RadImageNet perceptual distance (NOT LPIPS — see quality.py).
#
# COST (benchmarked, A100, 128-pair batches, all 150 slices x 30 draws x 9 sets):
#   alex ~7 h   |   vgg ~165 h (7 days!)
# => DO NOT just flip network_type to vgg at these settings. For a VGG run, also coarsen:
#    --network-type vgg --depth-stride 6 --bootstrap 10   (~7 h, ~25 of 150 slices).
#
# The full-pool number is skipped for perceptual (network all-pairs is intractable);
# only the equal-N bootstrap is reported. If alex times out, raise --depth-stride to 2.
#
# NOTE: FlowLet (Danese et al., MIA 2026) reports diversity with intra-set MS-SSIM,
# NOT LPIPS — so eval_diversity_msssim.slurm is the FlowLet-comparable run; this LPIPS
# run is a complementary in-house second lens.

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
REAL="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN="${CLUSTER_BASE}/MedicalDataSets/generated"
SEGC="${GEN}/seg_candidates_525"

echo "=== LPIPS-3D (alex) diversity | Job ${SLURM_JOB_ID} | Node ${SLURM_JOB_NODELIST} ==="
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
    --pair-chunk 128 \
    --metric perceptual \
    --network-type alex \
    --seed 42 \
    --output IDUN/output/eval/diversity_lpips_${SLURM_JOB_ID}.json
