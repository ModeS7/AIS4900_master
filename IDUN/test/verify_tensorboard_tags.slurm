#!/bin/bash
#SBATCH --job-name=verify_tb_tags
#SBATCH --output=IDUN/output/test/verify_tb_tags_%j.out
#SBATCH --error=IDUN/output/test/verify_tb_tags_%j.err
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --account=share-ie-idi

# =============================================================================
# TensorBoard Tag Verification - Complete Test Suite (All Trainers)
# =============================================================================
# Submit: sbatch IDUN/test/verify_tensorboard_tags.slurm
# All outputs saved to: /cluster/work/modestas/AIS4900_master/runs_tb_verify/
# =============================================================================

set -e

cd ${SLURM_SUBMIT_DIR}

echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Working dir: $(pwd)"
echo "Started: $(date)"
echo "============================================================"

# Load environment
module purge
module load Anaconda3/2024.02-1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate AIS4900

# Verify GPU
nvidia-smi
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# =============================================================================
# Configuration - ALL outputs go to single directory
# =============================================================================
TEST_BASE="/cluster/work/modestas/AIS4900_master/runs_tb_verify"
TEST_DIR="$TEST_BASE/run_$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}"
mkdir -p "$TEST_DIR"

echo ""
echo "All outputs saved to: $TEST_DIR"
echo ""

# Common training args for 2D
COMMON_2D="paths=cluster paths.model_dir=$TEST_DIR training=fast_debug training.epochs=2 training.warmup_epochs=0 training.limit_train_batches=10 training.batch_size=8"

# Common training args for 3D (smaller batch, more memory)
COMMON_3D="paths=cluster paths.model_dir=$TEST_DIR training=fast_debug training.epochs=2 training.warmup_epochs=0 training.limit_train_batches=5 training.batch_size=1"

# Track results
PASSED=0
FAILED=0
TOTAL=0

# =============================================================================
# Helper: Check TensorBoard tags
# =============================================================================
check_tags() {
    local run_dir="$1"
    local test_name="$2"
    shift 2
    local patterns=("$@")

    echo ""
    echo "============================================================"
    echo "CHECKING: $test_name"
    echo "Run dir: $run_dir"
    echo "============================================================"

    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        echo "ERROR: Run directory not found!"
        return 1
    fi

    local tb_dir="$run_dir/tensorboard"
    if [ ! -d "$tb_dir" ]; then
        echo "ERROR: No tensorboard directory found in $run_dir"
        ls -la "$run_dir"
        return 1
    fi

    local tags=$(python -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator('$tb_dir')
ea.Reload()
tags = ea.Tags().get('scalars', [])
for t in sorted(tags):
    print(t)
" 2>/dev/null)

    if [ -z "$tags" ]; then
        echo "ERROR: No tags found!"
        return 1
    fi

    echo ""
    echo "Found tags:"
    echo "$tags" | head -50
    echo ""

    local all_found=true
    echo "Expected patterns:"
    for pattern in "${patterns[@]}"; do
        if echo "$tags" | grep -q "$pattern"; then
            echo "  [FOUND] $pattern"
        else
            echo "  [MISSING] $pattern"
            all_found=false
        fi
    done

    if $all_found; then
        echo ""
        echo "Result: PASS"
        return 0
    else
        echo ""
        echo "Result: FAIL"
        return 1
    fi
}

find_run() {
    local subdir="$1"
    local prefix="$2"
    ls -td ${TEST_DIR}/${subdir}/${prefix}*/ 2>/dev/null | head -1
}

# =============================================================================
# 2D TRAINERS
# =============================================================================

# -----------------------------------------------------------------------------
# TEST 1: VAE bravo
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 1: VAE 2D bravo"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_vae \
    mode=bravo \
    $COMMON_2D \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=test1_vae_bravo_

RUN=$(find_run "vae_2d/bravo" "test1_vae_bravo_")
if check_tags "$RUN" "VAE 2D bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "Validation/LPIPS_bravo" \
    "Validation/MS-SSIM-3D_bravo" \
    "regional_bravo/tumor_loss"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 2: VQVAE bravo
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 2: VQVAE 2D bravo"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_vqvae \
    mode=bravo \
    $COMMON_2D \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=test2_vqvae_bravo_

RUN=$(find_run "vqvae_2d/bravo" "test2_vqvae_bravo_")
if check_tags "$RUN" "VQVAE 2D bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "Validation/LPIPS_bravo" \
    "regional_bravo/tumor_loss"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 3: VQVAE seg
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 3: VQVAE 2D seg"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_vqvae \
    mode=seg \
    vqvae.seg_mode=true \
    $COMMON_2D \
    training.logging.regional_losses=false \
    training.logging.msssim=false \
    training.logging.psnr=false \
    training.logging.lpips=false \
    training.name=test3_vqvae_seg_

RUN=$(find_run "vqvae_2d/seg" "test3_vqvae_seg_")
if check_tags "$RUN" "VQVAE 2D seg" \
    "Validation/Dice_seg" \
    "Validation/IoU_seg" \
    "Loss/BCE_val"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 4: DCAE bravo
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 4: DCAE 2D bravo"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_dcae \
    mode=bravo \
    $COMMON_2D \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=test4_dcae_bravo_

RUN=$(find_run "compression_2d/bravo" "test4_dcae_bravo_")
if check_tags "$RUN" "DCAE 2D bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "Validation/LPIPS_bravo" \
    "regional_bravo/tumor_loss"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 5: DCAE seg
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 5: DCAE 2D seg"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_dcae \
    mode=seg \
    dcae.seg_mode=true \
    $COMMON_2D \
    training.logging.regional_losses=false \
    training.logging.msssim=false \
    training.logging.psnr=false \
    training.logging.lpips=false \
    training.name=test5_dcae_seg_

RUN=$(find_run "compression_2d/seg" "test5_dcae_seg_")
if check_tags "$RUN" "DCAE 2D seg" \
    "Validation/Dice_seg" \
    "Validation/IoU_seg" \
    "Loss/BCE_val"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 6: Diffusion bravo
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 6: Diffusion 2D bravo"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train \
    mode=bravo \
    strategy=rflow \
    $COMMON_2D \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=false \
    training.name=test6_diff_bravo_

RUN=$(find_run "diffusion/bravo" "test6_diff_bravo_")
if check_tags "$RUN" "Diffusion 2D bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "regional_bravo/tumor_loss" \
    "test_best/PSNR_bravo" \
    "test_best/MS-SSIM_bravo"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 7: VAE multi_modality
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 7: VAE 2D multi_modality"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_vae \
    mode=multi_modality \
    $COMMON_2D \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=test7_vae_multi_

RUN=$(find_run "vae_2d/multi_modality" "test7_vae_multi_")
if check_tags "$RUN" "VAE 2D multi_modality" \
    "Validation/PSNR_t1_pre" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_t1_pre" \
    "Validation/MS-SSIM-3D_t1_pre" \
    "regional_t1_pre/tumor_loss" \
    "regional_bravo/tumor_loss"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# =============================================================================
# 3D TRAINERS
# =============================================================================

# -----------------------------------------------------------------------------
# TEST 8: VAE 3D bravo
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 8: VAE 3D bravo"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_vae_3d \
    mode=bravo \
    $COMMON_3D \
    vae_3d.disable_gan=true \
    training.gradient_checkpointing=true \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=false \
    training.name=test8_vae3d_bravo_

RUN=$(find_run "vae_3d/bravo" "test8_vae3d_bravo_")
if check_tags "$RUN" "VAE 3D bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "regional_bravo/tumor_loss"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 9: VQVAE 3D bravo
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 9: VQVAE 3D bravo"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_vqvae_3d \
    mode=bravo \
    $COMMON_3D \
    vqvae_3d.disable_gan=true \
    training.gradient_checkpointing=true \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=false \
    training.name=test9_vqvae3d_bravo_

RUN=$(find_run "vqvae_3d/bravo" "test9_vqvae3d_bravo_")
if check_tags "$RUN" "VQVAE 3D bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "regional_bravo/tumor_loss"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 10: VQVAE 3D seg
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 10: VQVAE 3D seg"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_vqvae_3d \
    mode=seg \
    vqvae_3d.seg_mode=true \
    vqvae_3d.disable_gan=true \
    $COMMON_3D \
    training.gradient_checkpointing=true \
    training.logging.regional_losses=false \
    training.logging.msssim=false \
    training.logging.psnr=false \
    training.logging.lpips=false \
    training.name=test10_vqvae3d_seg_

RUN=$(find_run "vqvae_3d/seg" "test10_vqvae3d_seg_")
if check_tags "$RUN" "VQVAE 3D seg" \
    "Validation/Dice_seg" \
    "Validation/IoU_seg" \
    "Loss/BCE_val"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 11: DCAE 3D bravo
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 11: DCAE 3D bravo"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_dcae_3d \
    mode=bravo \
    $COMMON_3D \
    dcae_3d.disable_gan=true \
    training.gradient_checkpointing=true \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=false \
    training.name=test11_dcae3d_bravo_

RUN=$(find_run "compression_3d/bravo" "test11_dcae3d_bravo_")
if check_tags "$RUN" "DCAE 3D bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "regional_bravo/tumor_loss"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# -----------------------------------------------------------------------------
# TEST 12: DCAE 3D seg
# -----------------------------------------------------------------------------
echo ""
echo "############################################################"
echo "# TEST 12: DCAE 3D seg"
echo "############################################################"
TOTAL=$((TOTAL + 1))

python -m medgen.scripts.train_dcae_3d \
    mode=seg \
    dcae_3d.seg_mode=true \
    dcae_3d.disable_gan=true \
    $COMMON_3D \
    training.gradient_checkpointing=true \
    training.logging.regional_losses=false \
    training.logging.msssim=false \
    training.logging.psnr=false \
    training.logging.lpips=false \
    training.name=test12_dcae3d_seg_

RUN=$(find_run "compression_3d/seg" "test12_dcae3d_seg_")
if check_tags "$RUN" "DCAE 3D seg" \
    "Validation/Dice_seg" \
    "Validation/IoU_seg" \
    "Loss/BCE_val"; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "############################################################"
echo "# SUMMARY"
echo "############################################################"
echo ""
echo "Passed: $PASSED / $TOTAL"
echo "Failed: $FAILED / $TOTAL"
echo ""
echo "All outputs in: $TEST_DIR"
echo ""
echo "Finished: $(date)"
echo ""

# Save summary
cat > "$TEST_DIR/summary.txt" << EOF
TensorBoard Tag Verification Summary
====================================
Date: $(date)
Job ID: $SLURM_JOB_ID
Node: $SLURM_JOB_NODELIST

Results: $PASSED / $TOTAL passed

2D Trainers:
1. VAE bravo
2. VQVAE bravo
3. VQVAE seg
4. DCAE bravo
5. DCAE seg
6. Diffusion bravo
7. VAE multi_modality

3D Trainers:
8. VAE 3D bravo
9. VQVAE 3D bravo
10. VQVAE 3D seg
11. DCAE 3D bravo
12. DCAE 3D seg

Directory: $TEST_DIR
EOF

echo "Summary saved to: $TEST_DIR/summary.txt"

echo ""
echo "Created directories:"
find "$TEST_DIR" -maxdepth 4 -type d | head -40

conda deactivate

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "SOME TESTS FAILED!"
    exit 1
else
    echo ""
    echo "ALL TESTS PASSED!"
    exit 0
fi
