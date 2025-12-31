#!/bin/bash
# Submit SLURM job preferring H100, fallback to original constraint after timeout
#
# Usage:
#   ./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm
#   ./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm 1800  # 30min timeout
#   ./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm --bg  # Run in background
#
# Supports: gpu80g, a100, h100, a100|h100, h100|a100, etc.
# Default: Wait 10 minutes for H100, then fallback to original constraint

set -e

# Check for --bg flag
if [[ "$*" == *"--bg"* ]]; then
    # Remove --bg from args and relaunch in background
    ARGS="${@/--bg/}"
    nohup "$0" $ARGS > /tmp/submit_h100_$$.log 2>&1 &
    echo "Running in background (PID: $!)"
    echo "Log: /tmp/submit_h100_$$.log"
    echo "Check with: tail -f /tmp/submit_h100_$$.log"
    exit 0
fi

JOB_SCRIPT=$1
WAIT_TIME=${2:-600}  # Default: 10 minutes (600 seconds)

if [ -z "$JOB_SCRIPT" ]; then
    echo "Usage: $0 <slurm_script> [wait_seconds]"
    echo "Example: $0 IDUN/train/vae/exp1_progressive_baseline.slurm 600"
    exit 1
fi

if [ ! -f "$JOB_SCRIPT" ]; then
    echo "Error: SLURM script not found: $JOB_SCRIPT"
    exit 1
fi

# Extract original constraint from the file
ORIGINAL_CONSTRAINT=$(grep -oP '(?<=--constraint=")[^"]+' "$JOB_SCRIPT" | head -1)

if [ -z "$ORIGINAL_CONSTRAINT" ]; then
    echo "Error: No --constraint found in $JOB_SCRIPT"
    echo "Expected format: #SBATCH --constraint=\"gpu80g\" or similar"
    exit 1
fi

echo "============================================================"
echo "Original constraint: $ORIGINAL_CONSTRAINT"
echo "Strategy: Try H100 first, fallback to original after ${WAIT_TIME}s"
echo "Script: $JOB_SCRIPT"
echo "============================================================"

# Check if already H100-only (no need to modify)
if [ "$ORIGINAL_CONSTRAINT" = "h100" ]; then
    echo "Already H100-only constraint, submitting directly..."
    JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
    echo "Job submitted: $JOB_ID"
    echo "Monitor with: squeue -j $JOB_ID"
    exit 0
fi

# Create temp file for H100-only version
TMP_H100=$(mktemp /tmp/job_h100_XXXXXX.slurm)

# Create H100-only version by replacing any constraint with h100
sed "s/--constraint=\"[^\"]*\"/--constraint=\"h100\"/" "$JOB_SCRIPT" > "$TMP_H100"

# Submit H100 job
JOB_ID=$(sbatch --parsable "$TMP_H100")
echo "H100 job submitted: $JOB_ID"
echo "Waiting up to ${WAIT_TIME}s for H100..."

# Check status periodically
ELAPSED=0
CHECK_INTERVAL=30

while [ $ELAPSED -lt $WAIT_TIME ]; do
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))

    STATE=$(squeue -j "$JOB_ID" -h -o %T 2>/dev/null || echo "COMPLETED")

    if [ "$STATE" = "RUNNING" ]; then
        echo "H100 job is now RUNNING!"
        rm -f "$TMP_H100"
        exit 0
    elif [ "$STATE" = "COMPLETED" ] || [ -z "$STATE" ]; then
        echo "H100 job completed or not found"
        rm -f "$TMP_H100"
        exit 0
    fi

    echo "  [$ELAPSED/${WAIT_TIME}s] Status: $STATE"
done

# Timeout - fallback to original constraint
echo ""
echo "H100 timeout reached. Cancelling and falling back to original constraint..."
echo "Fallback constraint: $ORIGINAL_CONSTRAINT"
scancel "$JOB_ID" 2>/dev/null || true

sleep 2

# Submit with original file (original constraint)
JOB_ID_FALLBACK=$(sbatch --parsable "$JOB_SCRIPT")
echo "Fallback job submitted: $JOB_ID_FALLBACK (constraint: $ORIGINAL_CONSTRAINT)"

# Cleanup
rm -f "$TMP_H100"

echo "============================================================"
echo "Fallback complete. Monitor with: squeue -j $JOB_ID_FALLBACK"
echo "============================================================"
