#!/bin/bash
# Usage: ./scripts/validate_before_submit.sh IDUN/train/diffusion/exp9.slurm

set -e
SLURM_FILE=$1

if [ -z "$SLURM_FILE" ] || [ ! -f "$SLURM_FILE" ]; then
    echo "Usage: $0 <slurm_file>"
    echo "Example: $0 IDUN/train/diffusion/exp1_rflow.slurm"
    exit 1
fi

echo "=== Validating: $SLURM_FILE ==="
echo ""

# Extract python command (handles multiline commands with backslash continuation)
# Handle CRLF line endings (common in files created on Windows or synced via git)
# Then join continuation lines and extract the python command
PYTHON_CMD=$(tr -d '\r' < "$SLURM_FILE" | perl -0777 -pe 's/\\\n\s*/ /g' | grep -E "^\s*(time\s+)?python -m medgen\.scripts\." | head -1 | sed 's/^\s*time\s*//' | sed 's/  */ /g')

if [ -z "$PYTHON_CMD" ]; then
    echo "[ERROR] Could not find python command in SLURM file"
    exit 1
fi

# --- Sanitize the extracted command for SAFE local validation -----------------
# Chained SLURM scripts end the train command with a trailing ' &' so the
# SIGUSR1 timeout trap can resubmit. If left in, the `eval`s below would launch
# a REAL, detached training run on the local GPU (an orphaned process that keeps
# consuming GPU/RAM). Strip it. Also drop cluster-only shell placeholders
# (${CHAIN_ARGS}, ${EXP_NAME}, ${CHAIN_RUN_DIR}, ...) that are only set by the
# SLURM harness at submit time and would otherwise mangle the command here.
PYTHON_CMD=$(echo "$PYTHON_CMD" | sed -E 's/[[:space:]]*&[[:space:]]*$//')
PYTHON_CMD=$(echo "$PYTHON_CMD" | sed -E 's/\$\{CHAIN_ARGS\}//g')
PYTHON_CMD=$(echo "$PYTHON_CMD" | sed -E 's/training\.name=\$\{EXP_NAME\}_?/training.name=validate_tmp_/g')
PYTHON_CMD=$(echo "$PYTHON_CMD" | sed -E 's/\$\{[A-Za-z_][A-Za-z0-9_]*\}//g')
PYTHON_CMD=$(echo "$PYTHON_CMD" | sed -E 's/[[:space:]]+/ /g; s/[[:space:]]+$//')

echo "Command: $PYTHON_CMD"
echo ""

# Step 1: Syntax check
echo "[1/4] Syntax check..."
python3 -m py_compile src/medgen/**/*.py
echo "       PASSED"

# Step 2: Import test
echo "[2/4] Import test..."
python -c "from medgen.scripts.train import main; from medgen.scripts.train_compression import main"
echo "       PASSED"

# Step 3: Config resolution (--cfg job)
echo "[3/4] Config resolution..."
# Remove 'paths=cluster' and add 'paths=local' for local validation
LOCAL_CMD=$(echo "$PYTHON_CMD" | sed 's/paths=cluster/paths=local/g')
eval "$LOCAL_CMD --cfg job" > /dev/null 2>&1 || {
    echo "       [WARNING] Config resolution failed, trying with training.epochs=0..."
    eval "$LOCAL_CMD training.epochs=0 --cfg job" > /dev/null
}
echo "       PASSED"

# Step 4: 1-batch dry run
echo "[4/4] Dry run (1 epoch, 2 batches)..."
# Write the dry run into a throwaway scratch dir so it never pollutes runs/.
# bs=1 + no compile keep it fast and within memory for 3D experiments.
DRY_RUNDIR="${TMPDIR:-/tmp}/medgen_validate_dry_$$"
DRY_CMD="$LOCAL_CMD training=fast_debug training.epochs=1 training.limit_train_batches=2 training.warmup_epochs=0 training.batch_size=1 training.use_compile=false hydra.run.dir=${DRY_RUNDIR}"
# Remove paths=cluster if still present
DRY_CMD=$(echo "$DRY_CMD" | sed 's/paths=cluster/paths=local/g')
echo "       Running: ${DRY_CMD:0:100}..."
# Run and capture output, show last 30 lines, check exit code
set +e  # Don't exit on error
OUTPUT=$(eval "$DRY_CMD" 2>&1)
EXIT_CODE=$?
set -e
rm -rf "${DRY_RUNDIR}"
echo "$OUTPUT" | tail -30
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "       FAILED (exit code: $EXIT_CODE)"
    exit 1
fi
echo "       PASSED"

echo ""
echo "=== VALIDATION PASSED ==="
echo "Safe to submit: sbatch $SLURM_FILE"
