#!/bin/bash
# Create golden checkpoints for E2E testing
#
# Usage: ./tests/fixtures/create_golden_checkpoint.sh
#
# This creates minimal checkpoints (1 epoch, few batches) for testing.
# These are NOT meant for real inference, just for verifying the pipeline works.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${SCRIPT_DIR}/golden_checkpoint"

echo "Creating golden checkpoints in: ${CHECKPOINT_DIR}"

# Common settings for minimal training (diffusion)
# Uses smoke_test model config: tiny UNet (~174K params)
COMMON_ARGS=(
    "training.epochs=1"
    "training.batch_size=2"
    "training.limit_train_batches=10"
    "training.generation_metrics.enabled=false"
    "training.use_compile=false"
    "training.warmup_epochs=0"
    "model=smoke_test"
)

# Create bravo checkpoint
echo ""
echo "=== Creating bravo checkpoint ==="
python -m medgen.scripts.train \
    mode=bravo \
    strategy=rflow \
    "${COMMON_ARGS[@]}" \
    "hydra.run.dir=${CHECKPOINT_DIR}/bravo"

# Create seg checkpoint
echo ""
echo "=== Creating seg checkpoint ==="
python -m medgen.scripts.train \
    mode=seg \
    strategy=rflow \
    "${COMMON_ARGS[@]}" \
    "hydra.run.dir=${CHECKPOINT_DIR}/seg"

# Create VAE checkpoint
echo ""
echo "=== Creating VAE checkpoint ==="
python -m medgen.scripts.train_compression \
    --config-name=vae \
    mode=multi_modality \
    training.epochs=1 \
    training.batch_size=2 \
    training.limit_train_batches=10 \
    training.use_compile=false \
    training.warmup_epochs=0 \
    "vae.channels=[32,64]" \
    "vae.attention_levels=[false,false]" \
    "vae.num_res_blocks=1" \
    "vae.disc_num_channels=8" \
    "model=smoke_test" \
    "hydra.run.dir=${CHECKPOINT_DIR}/vae"

echo ""
echo "=== Golden checkpoints created successfully ==="
echo ""
echo "Structure:"
find "${CHECKPOINT_DIR}" -name "*.pt" -o -name "events.out.tfevents.*" | head -20
echo ""
echo "You can now run E2E tests: pytest tests/e2e -v"
