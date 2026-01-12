#!/bin/bash
# Download TensorBoard event files from cluster, preserving directory structure
#
# Usage:
#   ./scripts/download_tensorboard.sh [remote_runs_path] [local_destination]
#
# Examples:
#   ./scripts/download_tensorboard.sh  # Uses defaults
#   ./scripts/download_tensorboard.sh /cluster/work/modestas/AIS4900_master/runs ./runs_cluster
#   ./scripts/download_tensorboard.sh runs/vqvae_3d ./runs_cluster  # Relative path on cluster

set -e

# Configuration - edit these defaults for your setup
REMOTE_HOST="idun-login1.hpc.ntnu.no"
REMOTE_USER="${CLUSTER_USER:-modestas}"
DEFAULT_REMOTE_PATH="/cluster/work/modestas/AIS4900_master/runs"
DEFAULT_LOCAL_PATH="./runs_cluster"

# Parse arguments
REMOTE_PATH="${1:-$DEFAULT_REMOTE_PATH}"
LOCAL_PATH="${2:-$DEFAULT_LOCAL_PATH}"

# Handle relative paths on remote
if [[ ! "$REMOTE_PATH" = /* ]]; then
    REMOTE_PATH="/cluster/work/modestas/AIS4900_master/$REMOTE_PATH"
fi

echo "Downloading TensorBoard files..."
echo "  From: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo "  To:   ${LOCAL_PATH}"
echo ""

# Create local directory
mkdir -p "$LOCAL_PATH"

# rsync with filters:
#   --include='*/' : include all directories (needed for structure)
#   --include='events.out.tfevents*' : include TensorBoard files
#   --exclude='*' : exclude everything else
#   --prune-empty-dirs : don't create empty directories
rsync -avz --progress \
    --include='*/' \
    --include='events.out.tfevents*' \
    --exclude='*' \
    --prune-empty-dirs \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/" \
    "${LOCAL_PATH}/"

echo ""
echo "Done! View with: tensorboard --logdir ${LOCAL_PATH}"
