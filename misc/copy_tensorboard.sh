#!/bin/bash
# Copy TensorBoard event files to a separate directory for easy downloading
# Run this on the cluster, then download the resulting folder
#
# Usage:
#   ./misc/copy_tensorboard.sh [source_runs] [destination]
#
# Examples:
#   ./misc/copy_tensorboard.sh                    # runs/ -> runs_tb/
#   ./misc/copy_tensorboard.sh runs runs_tb       # Same as above
#   ./misc/copy_tensorboard.sh runs/vqvae_3d tb   # Specific subfolder

set -e

SOURCE="${1:-runs}"
DEST="${2:-runs_tb}"

echo "Copying TensorBoard files..."
echo "  From: $SOURCE"
echo "  To:   $DEST"
echo ""

# Remove old destination if exists
rm -rf "$DEST"

# Copy only tfevents files, preserving structure
rsync -av \
    --include='*/' \
    --include='events.out.tfevents*' \
    --exclude='*' \
    --prune-empty-dirs \
    "$SOURCE/" "$DEST/"

# Show size comparison
echo ""
echo "Size comparison:"
du -sh "$SOURCE" 2>/dev/null || echo "  Source: (unable to calculate)"
du -sh "$DEST"
echo ""
echo "Done! Now download with: scp -r user@idun:path/$DEST ./local_path/"
