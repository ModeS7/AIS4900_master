#!/usr/bin/env bash
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="h100"
#SBATCH --job-name="gen_sm_48a_tail300"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/gen_sm_exp48a_tail300_%j.out
#SBATCH --error=IDUN/output/generate/gen_sm_exp48a_tail300_%j.err

set -Eeuo pipefail

export SYNTHMASK_START_CANDIDATE=300
export SYNTHMASK_OUTPUT_LABEL=exp1_to_exp48a_t025__tail_from_00300

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec bash "$REPO_ROOT/IDUN/generate/run_eval_generator_synthmask.sh" \
    exp1_to_exp48a_t025 \
    exp48a_lowt_only_lpips_strong_20260425-160342 \
    exp1_1_1000_pixel_bravo_20260402-121556 \
    0.25
