#!/bin/bash
# =============================================================================
# Local TensorBoard Tag Verification Script
# =============================================================================
# Run with: ./scripts/verify_tensorboard_tags.sh
# =============================================================================

set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

TEST_DIR="runs/tag_verification/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TEST_DIR"

echo "============================================================"
echo "TensorBoard Tag Verification (Local)"
echo "Output directory: $TEST_DIR"
echo "============================================================"

# Create tag checker script
cat > "$TEST_DIR/check_tags.py" << 'PYTHONSCRIPT'
#!/usr/bin/env python3
import sys
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_tags(run_dir):
    tb_dir = os.path.join(run_dir, 'tensorboard')
    if not os.path.exists(tb_dir):
        return []
    ea = EventAccumulator(tb_dir)
    ea.Reload()
    return sorted(ea.Tags().get('scalars', []))

def check_tags(run_dir, expected_patterns, test_name):
    tags = get_tags(run_dir)
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
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
    run_dir = sys.argv[1]
    test_name = sys.argv[2]
    patterns = sys.argv[3:]
    success = check_tags(run_dir, patterns, test_name)
    sys.exit(0 if success else 1)
PYTHONSCRIPT

PASSED=0
FAILED=0

# Minimal training config for fast testing
COMMON_ARGS="training.epochs=2 training.warmup_epochs=0 training.limit_train_batches=5 training.batch_size=4"

# =============================================================================
# Test 1: VAE bravo
# =============================================================================
echo -e "\n>>> TEST 1: VAE bravo mode"
timeout 300 python -m medgen.scripts.train_vae \
    mode=bravo \
    $COMMON_ARGS \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=true \
    training.name=tag_test_ || true

VAE_RUN=$(ls -td runs/vae_2d/bravo/tag_test_*/ 2>/dev/null | head -1)
if [ -n "$VAE_RUN" ]; then
    python "$TEST_DIR/check_tags.py" "$VAE_RUN" "VAE bravo" \
        "Validation/PSNR_bravo" \
        "Validation/MS-SSIM_bravo" \
        "Validation/MS-SSIM-3D_bravo" \
        "regional_bravo/tumor_loss" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))
else
    echo "SKIP: VAE bravo run not found"
    FAILED=$((FAILED + 1))
fi

# =============================================================================
# Test 2: DCAE seg mode
# =============================================================================
echo -e "\n>>> TEST 2: DCAE seg mode"
timeout 300 python -m medgen.scripts.train_dcae \
    mode=seg \
    dcae.seg_mode=true \
    $COMMON_ARGS \
    training.logging.regional_losses=false \
    training.logging.msssim=false \
    training.logging.psnr=false \
    training.logging.lpips=false \
    training.name=tag_test_ || true

SEG_RUN=$(ls -td runs/compression_2d/seg/tag_test_*/ 2>/dev/null | head -1)
if [ -n "$SEG_RUN" ]; then
    python "$TEST_DIR/check_tags.py" "$SEG_RUN" "DCAE seg" \
        "Validation/Dice_seg" \
        "Validation/IoU_seg" \
        "Loss/BCE_val" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))
else
    echo "SKIP: DCAE seg run not found"
    FAILED=$((FAILED + 1))
fi

# =============================================================================
# Test 3: Diffusion bravo (with test eval)
# =============================================================================
echo -e "\n>>> TEST 3: Diffusion bravo mode"
timeout 300 python -m medgen.scripts.train \
    mode=bravo \
    strategy=rflow \
    $COMMON_ARGS \
    training.logging.regional_losses=true \
    training.logging.msssim=true \
    training.logging.psnr=true \
    training.logging.lpips=false \
    training.name=tag_test_ || true

DIFF_RUN=$(ls -td runs/diffusion/bravo/tag_test_*/ 2>/dev/null | head -1)
if [ -n "$DIFF_RUN" ]; then
    python "$TEST_DIR/check_tags.py" "$DIFF_RUN" "Diffusion bravo" \
        "Validation/PSNR_bravo" \
        "Validation/MS-SSIM_bravo" \
        "regional_bravo/tumor_loss" \
        "test_best/PSNR_bravo" && PASSED=$((PASSED + 1)) || FAILED=$((FAILED + 1))
else
    echo "SKIP: Diffusion bravo run not found"
    FAILED=$((FAILED + 1))
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "VERIFICATION SUMMARY"
echo "============================================================"
echo "Passed: $PASSED / $((PASSED + FAILED))"
echo "Failed: $FAILED"
echo "============================================================"

# Save summary
cat > "$TEST_DIR/summary.txt" << EOF
TensorBoard Tag Verification Summary
Date: $(date)
Passed: $PASSED
Failed: $FAILED

Runs tested:
- VAE bravo: $VAE_RUN
- DCAE seg: $SEG_RUN
- Diffusion bravo: $DIFF_RUN
EOF

echo "Summary saved to: $TEST_DIR/summary.txt"

exit $FAILED
