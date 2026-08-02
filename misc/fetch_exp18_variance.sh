#!/bin/bash
# Fetch the exp18 training-variance replicates from the cluster.
# Run this LOCALLY (it pulls from the cluster), not on the cluster.
#
# Usage:
#   CLUSTER=user@idun-login1.hpc.ntnu.no ./misc/fetch_exp18_variance.sh
#   ./misc/fetch_exp18_variance.sh runs/eval/exp18_variance   # explicit destination
#
# Env:
#   CLUSTER       ssh target or alias                (default: cluster)
#   CLUSTER_BASE  remote work root, single-quoted so
#                 $USER expands on the REMOTE side   (default: /cluster/work/$USER)
#
# WHAT exp18 IS
#   IDUN/train/downstream/nnunet/exp18_{1,2}_*.slurm train fold 0 four times per arm with
#   identical data and identical fold membership. The spread across replicates is
#   run-to-run training variability with nothing else mixed in. It exists to answer
#   whether a downstream difference of a few thousandths of Dice is resolvable at all by
#   a single-run design.
#
# TWO READOUTS COME BACK, AND THEY ARE NOT INTERCHANGEABLE
#   fold_0/validation/summary.json
#       nnU-Net's own fold-0 validation: 21 real held-out patients, Dice ~0.62.
#   eval_<experiment>.json
#       Official-test evaluation: the 51 test patients, Dice ~0.32, carrying per_case
#       per-patient Dice. This exists because the slurm eval step calls
#       `eval_nnunet --experiment baseline`, and that mode predicts imagesTs only
#       (src/medgen/scripts/eval_nnunet.py, Mode 1). It is not an accident of this fetch.
#
#   Both are UPPER bounds on the noise of a five-fold ensemble, because one fold-0 model
#   is noisier than an ensemble of five. Prefer whichever matches the scale of the
#   difference being judged, and say which one was used.

set -Eeuo pipefail

fatal() {
    echo "FATAL: $*" >&2
    exit 1
}

readonly CLUSTER="${CLUSTER:-cluster}"
readonly CLUSTER_BASE="${CLUSTER_BASE:-/cluster/work/\$USER}"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DEST="${1:-${REPO_ROOT}/runs/eval/exp18_variance}"

readonly REMOTE_RUNS="${CLUSTER_BASE}/AIS4900_master/runs/downstream/nnunet"
readonly REMOTE_OUT="${CLUSTER_BASE}/AIS4900_master/IDUN/output/train/downstream"

# arm : dataset : plans  -- must track the readonly block at the top of each slurm file
readonly ARMS=("exp18_1_var_real_fold0:600:nnUNetResEncUNetLPlans"
               "exp18_2_var_hyb210_fold0:663:nnUNetResEncUNetLPlansD600")
readonly REPLICATES=(1 2 3 4)

command -v rsync >/dev/null || fatal "rsync is not installed"
mkdir -p "$DEST"

LIST="$(mktemp)"
trap 'rm -f "$LIST"' EXIT

for spec in "${ARMS[@]}"; do
    IFS=: read -r arm dataset plans <<<"$spec"
    model="nnUNetTrainerBrainMets__${plans}__3d_fullres"
    for rep in "${REPLICATES[@]}"; do
        exp="${arm}_rep${rep}"
        base="${exp}/Dataset${dataset}_BrainMet/${model}"
        echo "${base}/fold_0/validation/summary.json"   # 21 validation patients
        echo "${base}/fold_0/debug.json"                # epochs reached, resolved plan
        echo "${base}/dataset.json"                     # case counts for this arm
        echo "eval_${exp}.json"                         # 51 test patients, per_case
    done
done >"$LIST"

echo "=== fetching $(wc -l <"$LIST") files ==="
echo "  from ${CLUSTER}:${REMOTE_RUNS}"
echo "  to   ${DEST}"
echo ""
# --ignore-missing-args: report gaps below rather than aborting the whole transfer, so a
# single failed replicate does not cost the other seven.
rsync -avh --ignore-missing-args --files-from="$LIST" \
      "${CLUSTER}:${REMOTE_RUNS}/" "$DEST/" || true

# Job stdout carries the preflight result: the line proving a replicate trained on exactly
# the cases the ORIGINAL run's fold 0 used. Without it the replicates measure training
# noise PLUS a data difference, and the screen means nothing.
echo ""
echo "=== fetching slurm logs ==="
rsync -avh --ignore-missing-args \
      "${CLUSTER}:${REMOTE_OUT}/*exp18_*var*" "$DEST/slurm/" 2>/dev/null || \
    echo "  none matched -- fetch by hand if the preflight needs checking"

echo ""
echo "=== what arrived ==="
missing=0
while read -r f; do
    if [[ -s "$DEST/$f" ]]; then
        printf '  ok       %s\n' "$f"
    else
        printf '  MISSING  %s\n' "$f"
        missing=$((missing + 1))
    fi
done <"$LIST"

echo ""
total="$(wc -l <"$LIST")"
if (( missing == total )); then
    # Everything absent almost never means every replicate failed. It means the transfer
    # itself did not happen, and rsync's error was swallowed by `|| true` above.
    echo "  ALL ${total} files missing -- the transfer failed, not the jobs."
    echo "    check the ssh target: CLUSTER=${CLUSTER}"
    echo "    check the remote root: ${REMOTE_RUNS}"
elif (( missing )); then
    echo "  ${missing} of ${total} file(s) missing."
    echo "    no summary.json  -> that replicate's final validation did not run"
    echo "    no eval_*.json   -> the eval step failed; the slurm only warns on that,"
    echo "                        it does not fail the job, so check the .out file"
else
    echo "  all ${#ARMS[@]} arms x ${#REPLICATES[@]} replicates present."
fi
echo ""
echo "  next: point the training-variance analysis at --results ${DEST}"
