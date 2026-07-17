#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=idun-06-02
#SBATCH --job-name="recover_exp48a"
#SBATCH --output=IDUN/output/generate/recover_exp48a_%j.out
#SBATCH --error=IDUN/output/generate/recover_exp48a_%j.err
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL

# Validate and publish the 105 volumes completed by job 24795269 before its
# post-generation repository-commit guard failed. No generation is rerun.

set -Eeuo pipefail

: "${CLUSTER_BASE:=/cluster/work/${USER}}"
export CLUSTER_BASE

readonly REPO_ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR must be set}"
readonly PANEL_ROOT="${CLUSTER_BASE}/MedicalDataSets/evalModels/source_selection_train105_seed42_euler100"
readonly TRAIN_ROOT="${CLUSTER_BASE}/MedicalDataSets/brainmetshare-3/train"
readonly BRAVO_ROOT="${CLUSTER_BASE}/AIS4900_master/runs/diffusion_3d/bravo"

cd "$REPO_ROOT"

module purge
module load Anaconda3/2024.02-1
conda activate AIS4900

python IDUN/generate/panel_manifest.py recover-handoff \
    --label exp1_to_exp48a_t025 \
    --slurm-job-id 24795269 \
    --failure-log "$REPO_ROOT/IDUN/output/generate/gen_exp1_to_exp48a_24795269.err" \
    --output-log "$REPO_ROOT/IDUN/output/generate/gen_exp1_to_exp48a_24795269.out" \
    --observed-manifest-write-commit b442ea392b8f8a41d10049872b82bd9cdfcd3f7f \
    --input-root "$TRAIN_ROOT" \
    --dataset-root "$PANEL_ROOT/.staging/exp1_to_exp48a_t025.job-24795269" \
    --final-dataset-root "$PANEL_ROOT/exp1_to_exp48a_t025" \
    --low-checkpoint "$BRAVO_ROOT/exp48a_lowt_only_lpips_strong_20260425-160342/checkpoint_latest.pt" \
    --high-checkpoint "$BRAVO_ROOT/exp1_1_1000_pixel_bravo_20260402-121556/checkpoint_latest.pt" \
    --handoff-t 0.25

echo "Recovery complete: $PANEL_ROOT/exp1_to_exp48a_t025"
