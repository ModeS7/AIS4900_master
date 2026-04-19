#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu40g"
#SBATCH --job-name="eval_bin_exp2e"
#SBATCH --output=IDUN/output/eval/eval_bin_adherence_exp2e_%j.out
#SBATCH --error=IDUN/output/eval/eval_bin_adherence_exp2e_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

# Bin adherence evaluation for exp2e (multi-level auxiliary bin prediction)
# Model: seg_conditioned FiLM, 128x128x160, multi-level aux, regional weighting
# Compare against exp2c_1 baseline (same FiLM approach, no multi-level)

cd ${SLURM_SUBMIT_DIR}

echo "=== Bin Adherence Evaluation: exp2e (Multi-Level Aux) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
# CLUSTER_BASE default (override via env before sbatch)
: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# FILL IN CHECKPOINT PATH BEFORE SUBMISSION
# Run dir: ${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp2e_pixel_seg_multilevel_aux_20260226-130811
# ============================================================================
DATA_DIR="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3"
CKPT_EXP2E="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/seg/exp2e_pixel_seg_multilevel_aux_20260226-130811/checkpoint_latest.pt"

echo ""
echo "=== Evaluating exp2e (Multi-Level Aux FiLM) ==="
echo "Checkpoint: ${CKPT_EXP2E}"
echo ""

time python misc/eval_bin_adherence.py \
    --checkpoint "${CKPT_EXP2E}" \
    --model_type film \
    --data_dir "${DATA_DIR}" \
    --output_dir results/bin_adherence/exp2e_multilevel_aux \
    --cfg_scales "1.0,2.0,3.0,5.0" \
    --max_conditions 10 \
    --batch_size 4 \
    --num_steps 40 \
    --image_size 128 \
    --depth 160 \
    --seed 42

echo ""
echo "=== Evaluation Complete ==="
