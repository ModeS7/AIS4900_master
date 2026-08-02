#!/bin/bash
# Gather the exp18 training-variance replicates into one directory.
#
# Runs either place, and detects which:
#   ON the cluster   the 32 files are already on this filesystem, so it collects them
#                    locally and prints the one command that pulls the result down.
#   OFF the cluster  it rsyncs them over ssh. Set CLUSTER to your ssh target.
#
# Usage:
#   ./misc/fetch_exp18_variance.sh                            # on the cluster
#   CLUSTER=user@idun-login1.hpc.ntnu.no ./misc/fetch_exp18_variance.sh   # from home
#   ./misc/fetch_exp18_variance.sh runs/eval/exp18_variance   # explicit destination
#
# Env:
#   CLUSTER       ssh target or alias, used only when not already on the cluster
#   CLUSTER_BASE  work root. The default is written with a literal $USER so it resolves
#                 correctly whether it is expanded here or by the remote shell.
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
# Matches #SBATCH --output in both slurm files, which is relative to the submit dir:
#   IDUN/output/train/downstream/nnunet/<arm>/%A_%a.out
readonly REMOTE_OUT="${CLUSTER_BASE}/AIS4900_master/IDUN/output/train/downstream/nnunet"

# Same paths with $USER resolved against the shell we are actually running in. If they
# exist we are already on the cluster and ssh would be both unnecessary and wrong: the
# first run of this script was executed on idun-login2 and tried to resolve a host named
# "cluster" from inside the cluster.
readonly HERE_RUNS="$(eval echo "$REMOTE_RUNS")"
readonly HERE_OUT="$(eval echo "$REMOTE_OUT")"
if [[ -d "$HERE_RUNS" ]]; then
    readonly ON_CLUSTER=1
    readonly SRC_RUNS="${HERE_RUNS}"
    readonly SRC_OUT="${HERE_OUT}"
else
    readonly ON_CLUSTER=0
    readonly SRC_RUNS="${CLUSTER}:${REMOTE_RUNS}"
    readonly SRC_OUT="${CLUSTER}:${REMOTE_OUT}"
fi

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
        # 51 test patients with per_case. NOT at <results>/eval_<exp>.json: _setup_env
        # (eval_nnunet.py:39) reassigns nnUNet_results to <results>/<experiment> before
        # the output path is built from it, so the file lands one level deeper.
        echo "${exp}/eval_${exp}.json"
    done
done >"$LIST"

echo "=== collecting $(wc -l <"$LIST") files ==="
(( ON_CLUSTER )) && echo "  mode: on the cluster, no ssh" \
                 || echo "  mode: over ssh as ${CLUSTER}"
echo "  from ${SRC_RUNS}"
echo "  to   ${DEST}"
echo ""
# --ignore-missing-args: report gaps below rather than aborting the whole transfer, so a
# single failed replicate does not cost the other seven.
rsync -avh --ignore-missing-args --files-from="$LIST" \
      "${SRC_RUNS}/" "$DEST/" || true

# Job stdout carries the preflight result: the line proving a replicate trained on exactly
# the cases the ORIGINAL run's fold 0 used. Without it the replicates measure training
# noise PLUS a data difference, and the screen means nothing.
echo ""
echo "=== collecting slurm logs ==="
mkdir -p "$DEST/slurm"
for spec in "${ARMS[@]}"; do
    IFS=: read -r arm _ _ <<<"$spec"
    if rsync -avh --ignore-missing-args "${SRC_OUT}/${arm}/" "$DEST/slurm/${arm}/" \
            >/dev/null 2>&1 && compgen -G "$DEST/slurm/${arm}/*" >/dev/null; then
        echo "  ${arm}: $(find "$DEST/slurm/${arm}" -type f | wc -l) file(s)"
    else
        echo "  ${arm}: none found in ${SRC_OUT}/${arm}/"
    fi
done

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
    if (( ON_CLUSTER )); then
        echo "    ${SRC_RUNS} exists but holds none of these experiment names."
        echo "    check that the arm names in this script match the slurm files."
    else
        echo "    check the ssh target: CLUSTER=${CLUSTER}"
        echo "    check the remote root: ${REMOTE_RUNS}"
    fi
elif (( missing )); then
    echo "  ${missing} of ${total} file(s) missing."
    echo "    no summary.json  -> that replicate's final validation did not run"
    echo "    no eval_*.json   -> the eval step failed; the slurm only warns on that,"
    echo "                        it does not fail the job, so check the .out file"
else
    echo "  all ${#ARMS[@]} arms x ${#REPLICATES[@]} replicates present."
fi

echo ""
if (( ON_CLUSTER )); then
    echo "  everything is now under one directory. Pull it down with:"
    echo ""
    echo "    rsync -avh ${USER}@$(hostname -f 2>/dev/null || hostname):${DEST}/ \\"
    echo "        ~/NTNU/AIS4900_master/runs/eval/exp18_variance/"
else
    echo "  next: point the training-variance analysis at --results ${DEST}"
fi
