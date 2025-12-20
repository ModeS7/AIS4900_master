#!/bin/bash
# Submit SLURM job preferring H100, fallback to A100 after timeout
#
# Usage:
#   ./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm
#   ./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm 1800  # 30min timeout
#   ./IDUN/submit_prefer_h100.sh IDUN/train/vae/exp1_progressive_baseline.slurm --bg  # Run in background
#
# Default: Wait 10 minutes for H100, then fallback to A100

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

# Create temp files for H100 and A100 versions
TMP_H100=$(mktemp /tmp/job_h100_XXXXXX.slurm)
TMP_A100=$(mktemp /tmp/job_a100_XXXXXX.slurm)

# Create H100-only version
sed 's/--constraint="a100|h100"/--constraint="h100"/' "$JOB_SCRIPT" > "$TMP_H100"
sed -i 's/--constraint="h100|a100"/--constraint="h100"/' "$TMP_H100"

# Create fallback version (H100|A100 - accepts either)
sed 's/--constraint="h100"/--constraint="a100|h100"/' "$TMP_H100" > "$TMP_A100"

echo "============================================================"
echo "Submitting with H100 preference (fallback to H100|A100 after ${WAIT_TIME}s)"
echo "Script: $JOB_SCRIPT"
echo "============================================================"

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
        rm -f "$TMP_H100" "$TMP_A100"
        exit 0
    elif [ "$STATE" = "COMPLETED" ] || [ -z "$STATE" ]; then
        echo "H100 job completed or not found"
        rm -f "$TMP_H100" "$TMP_A100"
        exit 0
    fi

    echo "  [$ELAPSED/${WAIT_TIME}s] Status: $STATE"
done

# Timeout - fallback to H100|A100
echo ""
echo "H100 timeout reached. Cancelling and falling back to H100|A100..."
scancel "$JOB_ID" 2>/dev/null || true

sleep 2

JOB_ID_FALLBACK=$(sbatch --parsable "$TMP_A100")
echo "Fallback job submitted: $JOB_ID_FALLBACK (H100|A100)"

# Cleanup
rm -f "$TMP_H100" "$TMP_A100"

echo "============================================================"
echo "Fallback complete. Monitor with: squeue -j $JOB_ID_FALLBACK"
echo "============================================================"
