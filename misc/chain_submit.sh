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
#     fi
#     if [ -n "${CHAIN_RESUME:-}" ]; then
#         if [ -f "${CHAIN_RESUME}" ]; then
#             CHAIN_ARGS="${CHAIN_ARGS} training.resume_from=${CHAIN_RESUME}"
#         else
#             echo "No checkpoint found at ${CHAIN_RESUME}, starting fresh."
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

# Detect 2D vs 3D from script path
case "$SCRIPT" in
    *diffusion_3d*)   RUNS_SUBDIR="diffusion_3d" ;;
    *compression_3d*) RUNS_SUBDIR="compression_3d" ;;
    *compression*)    RUNS_SUBDIR="compression_2d" ;;
    *)                RUNS_SUBDIR="diffusion_2d" ;;
esac

RUN_DIR="/cluster/work/modestas/AIS4900_master/runs/${RUNS_SUBDIR}/${SCRIPT_NAME}_chain_${TIMESTAMP}"

# Optional time override
SBATCH_EXTRA=""
if [ -n "$TIME_OVERRIDE" ]; then
    SBATCH_EXTRA="--time=$TIME_OVERRIDE"
fi

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
