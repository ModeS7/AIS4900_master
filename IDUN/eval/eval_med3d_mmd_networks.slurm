#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-16:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g"
#SBATCH --job-name="eval_med3d_mmd_networks"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --output=IDUN/output/eval/eval_med3d_mmd_networks_%j.out
#SBATCH --error=IDUN/output/eval/eval_med3d_mmd_networks_%j.err

# Med3D-MMD swept across ALL MedicalNet ResNet backbones (10/18/34/50/101/152/200),
# for the SAME sets as eval_med3d_mmd.slurm / eval_diversity_msssim.slurm. Answers:
# does the Med3D-MMD ranking of datasets hold across backbones, or is it an artifact of
# the ResNet-50 feature space? Multi-reference (test-51 / train-105 / all-156) because the
# reference distribution materially changes the MMD (reference-confound). LOWER = closer.
#
# >>> OFFLINE-WEIGHTS PREREQUISITE (run ONCE on a LOGIN node with internet) <<<
# Compute nodes are offline (HF_HUB_OFFLINE); only Resnet50 is cached by default. Pre-cache
# the other six depths' weights BEFORE sbatch, or the fail-fast precheck aborts the job:
#
#   module load Anaconda3/2024.02-1 && conda activate AIS4900
#   python -c "from monai.networks.nets.resnet import get_pretrained_resnet_medicalnet as g; \
#              [g(d) for d in (10,18,34,50,101,152,200)]"
#
# VRAM note: deep backbones (resnet200) on native 256^3 OOM. --med3d-input-size 128 128 128
# bounds VRAM across ALL depths (Med3D adaptive-pools any size); consistency across depths
# matters more than matching the native-res single-network number for a ranking study.
# If it still OOMs, drop --batch-size to 1 or --med3d-input-size to 96 96 96.

set -e
cd ${SLURM_SUBMIT_DIR}

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE
REAL="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
GEN="${CLUSTER_BASE}/MedicalDataSets/generated"
SEGC="${GEN}/seg_candidates_525"

echo "=== Med3D-MMD networks sweep | Job ${SLURM_JOB_ID} | Node ${SLURM_JOB_NODELIST} ==="
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
    --networks 10 18 34 50 101 152 200 \
    --pattern bravo.nii.gz \
    --target-shape 150 256 256 \
    --med3d-input-size 128 128 128 \
    --pool-cap 256 \
    --subsample-n 51 \
    --bootstrap 30 \
    --batch-size 2 \
    --seed 42 \
    --output runs/eval/med3d_mmd_networks_${SLURM_JOB_ID}.json
