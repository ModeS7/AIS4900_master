#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name="extract_data"
#SBATCH --output=IDUN/output/extract_data_%j.out
#SBATCH --error=IDUN/output/extract_data_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=ALL

# Change to submission directory
cd ${SLURM_SUBMIT_DIR}

# Print job information
echo "============================================================"
echo "Data Extraction Job"
echo "============================================================"
echo "Job ID:            $SLURM_JOB_ID"
echo "Node:              $SLURM_JOB_NODELIST"
echo "CPUs:              $SLURM_CPUS_ON_NODE"
echo "============================================================"
echo ""

# Set paths (cluster paths)
ARCHIVE_FILE="/cluster/work/modestas/MedicalDataSets/data.tar.gz"
OUTPUT_DIR="/cluster/work/modestas/MedicalDataSets"

# Check if archive exists
if [ ! -f "$ARCHIVE_FILE" ]; then
    echo "ERROR: Archive not found: $ARCHIVE_FILE"
    exit 1
fi

# Show archive info
echo "Archive: $ARCHIVE_FILE"
echo "Size: $(du -h $ARCHIVE_FILE | cut -f1)"
echo "Output: $OUTPUT_DIR"
echo ""

# Extract with progress using parallel gzip (pigz) if available
echo "============================================================"
if command -v pigz &> /dev/null; then
    echo "Extracting datasets with pigz (parallel, using $SLURM_CPUS_ON_NODE cores)..."
    echo "============================================================"
    time tar -I pigz -xvf "$ARCHIVE_FILE" -C "$OUTPUT_DIR"
else
    echo "pigz not found, falling back to standard tar (single-threaded)..."
    echo "============================================================"
    time tar -xvzf "$ARCHIVE_FILE" -C "$OUTPUT_DIR"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "SUCCESS: Extraction complete"
    echo "============================================================"
    echo "Extracted to: $OUTPUT_DIR"
    ls -lh "$OUTPUT_DIR"
else
    echo ""
    echo "============================================================"
    echo "ERROR: Extraction failed"
    echo "============================================================"
    exit 1
fi
