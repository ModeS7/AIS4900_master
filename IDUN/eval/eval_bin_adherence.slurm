#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="eval_bin_adherence"
#SBATCH --output=IDUN/output/eval/eval_bin_adherence_%j.out
#SBATCH --error=IDUN/output/eval/eval_bin_adherence_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH --exclude=idun-06-02

cd ${SLURM_SUBMIT_DIR}

echo "=== Bin Adherence Evaluation: exp2_1 (FiLM) + exp2b_1 (Input) + exp2c_1 (Improved FiLM) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Memory optimization
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ============================================================================
# FILL IN CHECKPOINT PATHS BEFORE SUBMISSION
# ============================================================================
DATA_DIR="/cluster/work/modestas/MedicalDataSets/brainmetshare-3"
CKPT_FILM="runs/diffusion_3d/seg/exp2_1_pixel_seg_sizebin_rflow_256x160_20260201-015003/checkpoint_latest.pt"
CKPT_INPUT="runs/diffusion_3d/seg/exp2b_1_pixel_seg_input_cond_rflow_256x160_20260201-133413/checkpoint_latest.pt"
CKPT_EXP2C="runs/diffusion_3d/seg/exp2c_1_pixel_seg_improved_20260218-183155/checkpoint_latest.pt"

# ============================================================================
# Evaluation: exp2_1 FiLM conditioning
# ============================================================================
echo ""
echo "=== Evaluating exp2_1 (FiLM) ==="
echo ""
time python misc/eval_bin_adherence.py \
    --checkpoint "${CKPT_FILM}" \
    --model_type film \
    --data_dir "${DATA_DIR}" \
    --output_dir results/bin_adherence/exp2_1_film \
    --cfg_scales "1.0,2.0,3.0,5.0" \
    --max_conditions 10 \
    --batch_size 4 \
    --num_steps 40 \
    --image_size 256 \
    --depth 160 \
    --seed 42

# ============================================================================
# Evaluation: exp2b_1 Input channel conditioning
# ============================================================================
echo ""
echo "=== Evaluating exp2b_1 (Input) ==="
echo ""
time python misc/eval_bin_adherence.py \
    --checkpoint "${CKPT_INPUT}" \
    --model_type input \
    --data_dir "${DATA_DIR}" \
    --output_dir results/bin_adherence/exp2b_1_input \
    --cfg_scales "1.0,2.0,3.0,5.0" \
    --max_conditions 10 \
    --batch_size 4 \
    --num_steps 40 \
    --image_size 256 \
    --depth 160 \
    --seed 42

# ============================================================================
# Evaluation: exp2c_1 Improved FiLM conditioning
# (embedding_dim=64, projection_hidden_dim=512, projection_num_layers=4,
#  cfg_dropout_prob=0.25, regional_weighting)
# ============================================================================
echo ""
echo "=== Evaluating exp2c_1 (Improved FiLM) ==="
echo ""
time python misc/eval_bin_adherence.py \
    --checkpoint "${CKPT_EXP2C}" \
    --model_type film \
    --data_dir "${DATA_DIR}" \
    --output_dir results/bin_adherence/exp2c_1_film \
    --cfg_scales "1.0,2.0,3.0,5.0" \
    --max_conditions 10 \
    --batch_size 4 \
    --num_steps 40 \
    --image_size 256 \
    --depth 160 \
    --seed 42

echo ""
echo "=== Evaluation Complete ==="

conda deactivate
