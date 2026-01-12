#!/bin/sh
#SBATCH --job-name=verify_tb_tags
#SBATCH --output=IDUN/output/test/verify_tb_tags_%j.out
#SBATCH --error=IDUN/output/test/verify_tb_tags_%j.err
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100"
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=share-ie-idi

# =============================================================================
# TensorBoard Tag Verification Script
# =============================================================================
# This script runs all trainers with minimal epochs and verifies that
# TensorBoard tags are logged correctly with modality suffixes.
# =============================================================================

cd ${SLURM_SUBMIT_DIR}

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_JOB_NODELIST"
nvidia-smi

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# Test output directory
TEST_DIR="runs/tag_verification/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TEST_DIR"

echo "============================================================"
echo "TensorBoard Tag Verification"
echo "Output directory: $TEST_DIR"
echo "Started at: $(date)"
echo "============================================================"

# Python script to check TensorBoard tags
cat > "$TEST_DIR/check_tags.py" << 'PYTHONSCRIPT'
#!/usr/bin/env python3
"""Check TensorBoard tags for a run directory."""
import sys
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_tags(run_dir):
    """Get all scalar tags from a run directory."""
    tb_dir = os.path.join(run_dir, 'tensorboard')
    if not os.path.exists(tb_dir):
        # Try finding event file directly
        for f in os.listdir(run_dir):
            if f.startswith('events.'):
                tb_dir = run_dir
                break

    if not os.path.exists(tb_dir):
        return []

    ea = EventAccumulator(tb_dir)
    ea.Reload()
    return sorted(ea.Tags().get('scalars', []))

def check_tags(run_dir, expected_patterns, test_name):
    """Check if expected tag patterns are present."""
    tags = get_tags(run_dir)

    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print(f"Run dir: {run_dir}")
    print(f"{'='*60}")

    if not tags:
        print("ERROR: No tags found!")
        return False

    print(f"\nAll tags ({len(tags)}):")
    for tag in tags:
        print(f"  {tag}")

    print(f"\nExpected patterns:")
    all_found = True
    for pattern in expected_patterns:
        found = any(pattern in tag for tag in tags)
        status = "FOUND" if found else "MISSING"
        print(f"  [{status}] {pattern}")
        if not found:
            all_found = False

    print(f"\nResult: {'PASS' if all_found else 'FAIL'}")
    return all_found

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: check_tags.py <run_dir> <test_name> <pattern1> [pattern2] ...")
        sys.exit(1)

    run_dir = sys.argv[1]
    test_name = sys.argv[2]
    patterns = sys.argv[3:]

    success = check_tags(run_dir, patterns, test_name)
    sys.exit(0 if success else 1)
PYTHONSCRIPT

chmod +x "$TEST_DIR/check_tags.py"

# Track overall results
PASSED=0
FAILED=0
RESULTS=""

run_test() {
    local test_name="$1"
    local train_cmd="$2"
    shift 2
    local expected_patterns=("$@")

    echo ""
    echo "============================================================"
    echo "Running: $test_name"
    echo "============================================================"

    # Run training
    if eval "$train_cmd"; then
        echo "Training completed successfully"
    else
        echo "ERROR: Training failed!"
        FAILED=$((FAILED + 1))
        RESULTS="$RESULTS\n[FAIL] $test_name - Training failed"
        return 1
    fi

    # Find the latest run directory
    local run_pattern="$3"
    local latest_run=$(ls -td $run_pattern 2>/dev/null | head -1)

    if [ -z "$latest_run" ]; then
        echo "ERROR: Could not find run directory matching: $run_pattern"
        FAILED=$((FAILED + 1))
        RESULTS="$RESULTS\n[FAIL] $test_name - Run directory not found"
        return 1
    fi

    # Check tags
    if python "$TEST_DIR/check_tags.py" "$latest_run" "$test_name" "${expected_patterns[@]}"; then
        PASSED=$((PASSED + 1))
        RESULTS="$RESULTS\n[PASS] $test_name"
    else
        FAILED=$((FAILED + 1))
        RESULTS="$RESULTS\n[FAIL] $test_name - Missing expected tags"
    fi
}

# =============================================================================
# Test 1: VAE bravo mode - validation tags with modality suffix
# =============================================================================
echo ""
echo ">>> TEST 1: VAE bravo mode"
python -m medgen.scripts.train_vae \
    mode=bravo \
    paths=cluster \
    training=fast_debug \
    training.epochs=2 \
    training.warmup_epochs=0 \
    training.limit_train_batches=10 \
    training.batch_size=8 \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=tag_test_

VAE_BRAVO_RUN=$(ls -td runs/vae_2d/bravo/tag_test_*/ 2>/dev/null | head -1)
python "$TEST_DIR/check_tags.py" "$VAE_BRAVO_RUN" "VAE bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "Validation/LPIPS_bravo" \
    "Validation/MS-SSIM-3D_bravo" \
    "regional_bravo/tumor_loss" \
    "regional_bravo/background_loss" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))

# =============================================================================
# Test 2: VQVAE bravo mode - validation tags with modality suffix
# =============================================================================
echo ""
echo ">>> TEST 2: VQVAE bravo mode"
python -m medgen.scripts.train_vqvae \
    mode=bravo \
    paths=cluster \
    training=fast_debug \
    training.epochs=2 \
    training.warmup_epochs=0 \
    training.limit_train_batches=10 \
    training.batch_size=8 \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=tag_test_

VQVAE_BRAVO_RUN=$(ls -td runs/vqvae_2d/bravo/tag_test_*/ 2>/dev/null | head -1)
python "$TEST_DIR/check_tags.py" "$VQVAE_BRAVO_RUN" "VQVAE bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "Validation/LPIPS_bravo" \
    "regional_bravo/tumor_loss" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))

# =============================================================================
# Test 3: DCAE bravo mode - validation tags with modality suffix
# =============================================================================
echo ""
echo ">>> TEST 3: DCAE bravo mode"
python -m medgen.scripts.train_dcae \
    mode=bravo \
    paths=cluster \
    training=fast_debug \
    training.epochs=2 \
    training.warmup_epochs=0 \
    training.limit_train_batches=10 \
    training.batch_size=8 \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=tag_test_

DCAE_BRAVO_RUN=$(ls -td runs/compression_2d/bravo/tag_test_*/ 2>/dev/null | head -1)
python "$TEST_DIR/check_tags.py" "$DCAE_BRAVO_RUN" "DCAE bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "Validation/LPIPS_bravo" \
    "regional_bravo/tumor_loss" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))

# =============================================================================
# Test 4: DCAE seg mode - Dice/IoU validation tags
# =============================================================================
echo ""
echo ">>> TEST 4: DCAE seg mode"
python -m medgen.scripts.train_dcae \
    mode=seg \
    dcae.seg_mode=true \
    paths=cluster \
    training=fast_debug \
    training.epochs=2 \
    training.warmup_epochs=0 \
    training.limit_train_batches=10 \
    training.batch_size=8 \
    training.logging.regional_losses=true \
    training.logging.msssim=false \
    training.logging.psnr=false \
    training.logging.lpips=false \
    training.name=tag_test_

DCAE_SEG_RUN=$(ls -td runs/compression_2d/seg/tag_test_*/ 2>/dev/null | head -1)
python "$TEST_DIR/check_tags.py" "$DCAE_SEG_RUN" "DCAE seg mode" \
    "Validation/Dice_seg" \
    "Validation/IoU_seg" \
    "Loss/BCE_val" \
    "Loss/Boundary_val" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))

# =============================================================================
# Test 5: Diffusion bravo mode - validation and test tags
# =============================================================================
echo ""
echo ">>> TEST 5: Diffusion bravo mode"
python -m medgen.scripts.train \
    mode=bravo \
    strategy=rflow \
    paths=cluster \
    training=fast_debug \
    training.epochs=2 \
    training.warmup_epochs=0 \
    training.limit_train_batches=10 \
    training.batch_size=8 \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=tag_test_

DIFF_BRAVO_RUN=$(ls -td runs/diffusion/bravo/tag_test_*/ 2>/dev/null | head -1)
python "$TEST_DIR/check_tags.py" "$DIFF_BRAVO_RUN" "Diffusion bravo" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_bravo" \
    "regional_bravo/tumor_loss" \
    "test_best/PSNR_bravo" \
    "test_best/MS-SSIM_bravo" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))

# =============================================================================
# Test 6: VAE multi_modality - per-modality tags
# =============================================================================
echo ""
echo ">>> TEST 6: VAE multi_modality mode"
python -m medgen.scripts.train_vae \
    mode=multi_modality \
    paths=cluster \
    training=fast_debug \
    training.epochs=2 \
    training.warmup_epochs=0 \
    training.limit_train_batches=10 \
    training.batch_size=8 \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=tag_test_

VAE_MULTI_RUN=$(ls -td runs/vae_2d/multi_modality/tag_test_*/ 2>/dev/null | head -1)
python "$TEST_DIR/check_tags.py" "$VAE_MULTI_RUN" "VAE multi_modality" \
    "Validation/PSNR_t1_pre" \
    "Validation/PSNR_bravo" \
    "Validation/MS-SSIM_t1_pre" \
    "Validation/MS-SSIM-3D_t1_pre" \
    "regional_t1_pre/tumor_loss" \
    "regional_bravo/tumor_loss" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "VERIFICATION SUMMARY"
echo "============================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""
echo "Results:"
echo -e "$RESULTS"
echo ""
echo "Completed at: $(date)"
echo "============================================================"

# Save summary to file
cat > "$TEST_DIR/summary.txt" << EOF
TensorBoard Tag Verification Summary
=====================================
Date: $(date)
Passed: $PASSED
Failed: $FAILED

Results:
$(echo -e "$RESULTS")

Run directories tested:
- VAE bravo: $VAE_BRAVO_RUN
- VQVAE bravo: $VQVAE_BRAVO_RUN
- DCAE bravo: $DCAE_BRAVO_RUN
- DCAE seg: $DCAE_SEG_RUN
- Diffusion bravo: $DIFF_BRAVO_RUN
- VAE multi_modality: $VAE_MULTI_RUN
EOF

echo "Summary saved to: $TEST_DIR/summary.txt"

# Cleanup
conda deactivate

# Exit with error if any tests failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi
