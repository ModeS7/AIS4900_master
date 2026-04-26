#!/bin/bash
# Submit all Tier 1 find_optimal_steps SLURM jobs.
# Usage:
#   ./IDUN/eval/tier1/submit_all_tier1.sh           # submit everything
#   ./IDUN/eval/tier1/submit_all_tier1.sh --dry-run # just show what would run
#   ./IDUN/eval/tier1/submit_all_tier1.sh exp17_2 exp34_0_1000  # submit only these

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0

# Group order: cheapest first (Mamba re-search has the strongest case),
# then UViT/UNet at 8h, then HDiT-XL at 14h.
ALL_JOBS=(
    # 1d Mamba — cheapest re-search (strongest case for changing rankings)
    "exp34_0_1000"
    "exp34_1_1000"
    # 1c misc
    "exp15"
    "exp18"
    "exp29"
    # 1b UNet model-scaling sweep
    "exp20_4"
    "exp20_6"
    "exp20_7"
    "exp20_2"
    "exp20_3"
    "exp20_5"
    # 1a HDiT (most expensive)
    "exp17_2"
    "exp17_3"
)

# Parse args
TARGETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            grep -E '^# ' "$0" | head -10
            exit 0
            ;;
        *) TARGETS+=("$1") ;;
    esac
    shift
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("${ALL_JOBS[@]}")
fi

echo "Tier 1 find_optimal_steps batch — ${#TARGETS[@]} jobs"
echo "================================================="

submitted_ids=()
for exp in "${TARGETS[@]}"; do
    slurm_file="${SCRIPT_DIR}/find_optimal_steps_${exp}.slurm"
    if [[ ! -f "$slurm_file" ]]; then
        echo "  SKIP $exp (no SLURM file at $slurm_file)"
        continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  WOULD SUBMIT $slurm_file"
        continue
    fi
    output=$(sbatch "$slurm_file")
    echo "  $output  ← $exp"
    jobid=$(echo "$output" | awk '{print $NF}')
    submitted_ids+=("$jobid:$exp")
done

if [[ $DRY_RUN -eq 0 ]]; then
    echo ""
    echo "Submitted ${#submitted_ids[@]} jobs:"
    for s in "${submitted_ids[@]}"; do
        echo "  $s"
    done
    echo ""
    echo "Watch progress:"
    echo "  squeue -u \$USER --format='%.10i %.30j %.8T %.10M %R'"
    echo "  tail -f IDUN/output/eval/find_optimal_steps_<exp>_<jobid>.out"
    echo ""
    echo "Aggregate results once done:"
    echo "  python scripts/extract_idun_logs.py --only find_optimal_steps_exp --out /tmp/tier1_results.json"
fi
