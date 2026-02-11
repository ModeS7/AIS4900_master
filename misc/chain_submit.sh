#!/bin/bash
# Chain-submit SLURM training jobs for seamless multi-segment training.
#
# Splits long training runs into shorter SLURM jobs that queue faster.
# All segments share the same output directory for continuous TensorBoard.
#
# Requirements:
#   The SLURM script must include the chain support block:
#
#     # === Job chaining support (set by misc/chain_submit.sh) ===
#     CHAIN_ARGS=""
#     if [ -n "${CHAIN_RUN_DIR:-}" ]; then
#         CHAIN_ARGS="hydra.run.dir=${CHAIN_RUN_DIR}"
#         echo "Chain mode: writing to ${CHAIN_RUN_DIR}"
#     fi
#     if [ -n "${CHAIN_RESUME:-}" ]; then
#         if [ -f "${CHAIN_RESUME}" ]; then
#             if python -c "import zipfile; zipfile.ZipFile('${CHAIN_RESUME}').close()"; then
#                 CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CHAIN_RESUME}"
#                 echo "Chain mode: resuming from ${CHAIN_RESUME}"
#             else
#                 echo "WARNING: checkpoint_latest.pt failed validation, trying checkpoint_best.pt"
#                 BEST_CKPT="$(dirname "${CHAIN_RESUME}")/checkpoint_best.pt"
#                 if [ -f "$BEST_CKPT" ] && python -c "import zipfile; zipfile.ZipFile('${BEST_CKPT}').close()"; then
#                     CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${BEST_CKPT}"
#                     echo "Chain mode: resuming from ${BEST_CKPT} (fallback)"
#                 else
#                     echo "WARNING: no valid checkpoint found, starting fresh."
#                 fi
#             fi
#         else
#             echo "Chain mode: no checkpoint at ${CHAIN_RESUME}, starting fresh."
#         fi
#     fi
#
#   And append ${CHAIN_ARGS} at the end of the python command.
#
# Usage:
#   # 3 segments using time from SLURM script
#   ./misc/chain_submit.sh IDUN/train/diffusion_3d/exp9_1_ldm_4x_bravo.slurm 3
#
#   # 5 segments, 3 hours each (overrides script's --time)
#   ./misc/chain_submit.sh IDUN/train/diffusion_3d/exp9_1_ldm_4x_bravo.slurm 5 0-03:00:00
#
# How it works:
#   - Segment 1: fresh start, creates run directory
#   - Segments 2+: resume from checkpoint_latest.pt
#   - All segments write to the same directory (seamless TensorBoard)
#   - Uses --dependency=afterany so chains continue even if time-killed
#   - Training exits cleanly when all epochs are done (empty range)

set -euo pipefail

SCRIPT="${1:?Usage: $0 <slurm_script> [num_segments] [time_per_segment]}"
NUM_SEGMENTS="${2:-3}"
TIME_OVERRIDE="${3:-}"

if [ ! -f "$SCRIPT" ]; then
    echo "Error: SLURM script not found: $SCRIPT"
    exit 1
fi

# Verify script has chain support
if ! grep -q 'CHAIN_RUN_DIR' "$SCRIPT"; then
    echo "Error: SLURM script missing chain support block."
    echo "See misc/chain_submit.sh header for the required block."
    exit 1
fi

# Build run directory path
# Extract the config-name to determine the runs subdirectory
SCRIPT_NAME=$(basename "$SCRIPT" .slurm)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Determine runs subdirectory by parsing --config-name from the python command.
# This is the source of truth — it determines the Hydra output directory structure.
#
# Mapping:
#   config-name         -> runs subdir          -> has mode subdir?
#   diffusion           -> diffusion_2d/{mode}  -> yes (mode.subdir or mode.name)
#   diffusion_3d        -> diffusion_3d/{mode}  -> yes
#   vae/vqvae/dcae      -> compression_2d/{mode}-> yes (mode.name)
#   vae_3d/vqvae_3d/... -> compression_3d/{mode}-> yes
#   segmentation        -> segmentation_{N}d    -> no (uses spatial_dims)
# Detect script type: config-name for diffusion/compression, script name for segmentation
CONFIG_NAME=$(grep -oP '\-\-config-name=\K\S+' "$SCRIPT" | head -1)
TRAIN_SCRIPT=$(grep -oP 'python -m \K\S+' "$SCRIPT" | head -1)

# Segmentation uses train_segmentation without --config-name
if [ "$TRAIN_SCRIPT" = "medgen.scripts.train_segmentation" ]; then
    CONFIG_NAME="segmentation"
fi

case "$CONFIG_NAME" in
    diffusion)      RUNS_SUBDIR="diffusion_2d" ;;
    diffusion_3d)   RUNS_SUBDIR="diffusion_3d" ;;
    vae|vqvae|dcae) RUNS_SUBDIR="compression_2d" ;;
    vae_3d|vqvae_3d|dcae_3d) RUNS_SUBDIR="compression_3d" ;;
    segmentation)
        SPATIAL_DIMS=$(grep -oP 'spatial_dims=\K[23]' "$SCRIPT" | head -1)
        RUNS_SUBDIR="segmentation_${SPATIAL_DIMS:-2}d" ;;
    *)
        echo "Warning: unknown config-name '${CONFIG_NAME}', defaulting to diffusion_2d"
        RUNS_SUBDIR="diffusion_2d" ;;
esac

# Extract mode subdirectory to match Hydra's directory structure.
# Diffusion + compression configs use a mode subdir; segmentation does not.
MODE=$(grep -oP '^\s*mode=\K\S+' "$SCRIPT" | head -1 | tr -d ' \\')
MODE_SUBDIR=""
if [ -n "$MODE" ]; then
    MODE_SUBDIR="$MODE"
    # Check for subdir override in mode config (e.g., bravo_seg_cond -> bravo_latent)
    SCRIPT_DIR=$(dirname "$0")
    MODE_CONFIG="${SCRIPT_DIR}/../configs/mode/${MODE}.yaml"
    if [ -f "$MODE_CONFIG" ]; then
        SUBDIR_OVERRIDE=$(grep -oP 'subdir:\s*\K\S+' "$MODE_CONFIG" 2>/dev/null || true)
        if [ -n "$SUBDIR_OVERRIDE" ]; then
            MODE_SUBDIR="$SUBDIR_OVERRIDE"
        fi
    fi
fi

# Build run directory: runs/{type}/{mode?}/{name}_chain_{timestamp}
if [ -n "$MODE_SUBDIR" ]; then
    RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/${RUNS_SUBDIR}/${MODE_SUBDIR}/${SCRIPT_NAME}_chain_${TIMESTAMP}"
else
    RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/${RUNS_SUBDIR}/${SCRIPT_NAME}_chain_${TIMESTAMP}"
fi

# Optional time override
SBATCH_EXTRA=""
if [ -n "$TIME_OVERRIDE" ]; then
    SBATCH_EXTRA="--time=$TIME_OVERRIDE"
fi

# Send SIGTERM 5 minutes before SLURM's time limit.
# This gives the trainer enough time to finish the current batch,
# save a checkpoint (even 50GB+ models), and exit cleanly.
# The B: prefix sends to the entire process group (so Python gets it).
SBATCH_EXTRA="$SBATCH_EXTRA --signal=B:TERM@300"

# Create SLURM output directory if script uses a subdirectory
OUTPUT_DIR=$(grep -m1 '#SBATCH --output=' "$SCRIPT" | sed 's/.*--output=//' | xargs dirname)
if [ -n "$OUTPUT_DIR" ] && [ "$OUTPUT_DIR" != "." ]; then
    mkdir -p "$OUTPUT_DIR"
fi

echo "=== Chain Submit ==="
echo "Script:       $SCRIPT"
echo "Segments:     $NUM_SEGMENTS"
echo "Run directory: $RUN_DIR"
if [ -n "$TIME_OVERRIDE" ]; then
    echo "Time/segment: $TIME_OVERRIDE"
fi
echo ""

# Submit first job (fresh start)
JOB1=$(sbatch --parsable $SBATCH_EXTRA \
    --export=ALL,CHAIN_RUN_DIR="${RUN_DIR}" \
    "$SCRIPT")
echo "Segment 1/${NUM_SEGMENTS}: job ${JOB1} (fresh start)"

# Submit chained jobs
PREV=$JOB1
ALL_JOBS="$JOB1"
for i in $(seq 2 $NUM_SEGMENTS); do
    NEXT=$(sbatch --parsable $SBATCH_EXTRA \
        --dependency=afterany:${PREV} \
        --export=ALL,CHAIN_RUN_DIR="${RUN_DIR}",CHAIN_RESUME="${RUN_DIR}/checkpoint_latest.pt" \
        "$SCRIPT")
    echo "Segment ${i}/${NUM_SEGMENTS}: job ${NEXT} (after ${PREV})"
    ALL_JOBS="${ALL_JOBS},${NEXT}"
    PREV=$NEXT
done

echo ""
echo "All jobs: ${ALL_JOBS}"
echo "Monitor:  squeue -u \$(whoami)"
echo "Cancel:   scancel ${ALL_JOBS}"
