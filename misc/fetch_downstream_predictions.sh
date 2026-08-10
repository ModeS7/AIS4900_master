#!/bin/bash
# Pull saved nnU-Net official-test predictions for a handful of patients, so a
# qualitative segmentation panel can be built off the cluster.
#
# Runs either place, and detects which:
#   ON the cluster   the masks are already on this filesystem, so it collects them
#                    locally and prints the one command that pulls the result down.
#   OFF the cluster  it rsyncs them over ssh. Set CLUSTER to your ssh target.
#
# Usage:
#   ./misc/fetch_downstream_predictions.sh                          # on the cluster
#   CLUSTER=user@idun-login1.hpc.ntnu.no ./misc/fetch_downstream_predictions.sh
#   CONDITIONS="real_reference weighted_huber_handoff" ./misc/fetch_downstream_predictions.sh
#   CASES="BrainMet_038 BrainMet_257" ./misc/fetch_downstream_predictions.sh
#
# Env:
#   CLUSTER       ssh target or alias, used only when not already on the cluster
#   CLUSTER_BASE  work root, default written with a literal $USER so it resolves
#                 whether expanded here or by the remote shell
#   CONDITIONS    space-separated condition labels, default = the three-column panel
#   CASES         space-separated case ids, default = the five documented below
#   DEST          output directory, default runs/eval/downstream_predictions
#
# WHY THIS EXISTS
#   The MIA submission questionnaire asks authors to confirm that findings derived from
#   3D MRI are shown from at least two orthogonal planes. The generation findings already
#   are: the qualitative figure carries axial, coronal and sagittal. The SEGMENTATION
#   findings are reported only as tables, because building a panel needs the predicted
#   masks, and those live only here -- the local dataset drive has the real volumes and
#   the ground truth but no model output.
#
#   Everything else the panel needs is already off-cluster:
#     real BRAVO + ground truth   <dataset root>/brainmetshare-3/test/<case>/{bravo,seg}.nii.gz
#   so this fetches the one missing ingredient and nothing more.
#
# WHICH CASES, AND WHY THESE
#   The 51 official test patients sorted by real-reference volumetric Dice, then five
#   evenly spaced ranks. This is the same selection rule the generation gallery uses, so
#   the panels stay comparable, and it spans the range instead of flattering it:
#
#     rank  0/50  BrainMet_063  reference 0.000   handoff (c) 0.000   <- detection floor
#     rank 12/50  BrainMet_081  reference 0.056   handoff (c) 0.048
#     rank 25/50  BrainMet_257  reference 0.275   handoff (c) 0.268   <- near cohort mean
#     rank 38/50  BrainMet_241  reference 0.579   handoff (c) 0.454
#     rank 50/50  BrainMet_038  reference 0.917   handoff (c) 0.907
#
#   Derived from results_archive/SEG7_job_24929214/per_case.csv. Rank 0 is one of the
#   nine patients every condition scores zero on. Keep it. A panel that shows only the
#   cases a model handles is not a finding, it is a highlight reel.
#
# WHICH CONDITIONS
#   Default is the three that carry the paper's argument: the real-data reference, the
#   validation-selected synthetic source, and the smallest hybrid level. Override with
#   CONDITIONS to fetch any of the twelve listed in PRED_DIR below.
#
# PROVENANCE
#   Every path below is copied from the `documented_prediction_dir` field of
#   results_archive/{SEG7_job_24929214,HYB4_job_24943071}/manifest.json, which is what
#   recompute_nnunet_metrics.py recorded when it produced the published numbers. If a
#   path 404s, re-read the manifest rather than guessing the directory name.

set -Eeuo pipefail

CLUSTER_BASE="${CLUSTER_BASE:-/cluster/work/\$USER}"
DEST="${DEST:-runs/eval/downstream_predictions}"
CONDITIONS="${CONDITIONS:-real_reference weighted_huber_handoff hybrid_25}"
CASES="${CASES:-BrainMet_063 BrainMet_081 BrainMet_257 BrainMet_241 BrainMet_038}"

# condition -> experiment directory. The prediction pool is always
#   <base>/AIS4900_master/runs/downstream/nnunet/<exp>/eval_<exp>/predictions
exp_for() {
  case "$1" in
    real_reference)                  echo "exp3_baseline_v2_d600" ;;
    original_mse)                    echo "exp16_2_synthetic_105_common105_exp1_1_1000_d650" ;;
    extended_mse)                    echo "exp16_2_synthetic_105_common105_exp1_1_1000plus_d651" ;;
    perceptual_continuation)         echo "exp16_2_synthetic_105_common105_exp32_2_1000_d652" ;;
    strong_perceptual_continuation)  echo "exp16_2_synthetic_105_common105_exp47a_d654" ;;
    weighted_huber_transition)       echo "exp16_2_synthetic_105_common105_exp47c_d656" ;;
    weighted_huber_handoff)          echo "exp16_2_synthetic_105_common105_exp1_to_exp48c_t025_d661" ;;
    pseudo_huber_perceptual_handoff) echo "exp16_2_synthetic_105_common105_exp1_to_exp48d_t025_d662" ;;
    hybrid_25)                       echo "exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663" ;;
    hybrid_50)                       echo "exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663" ;;
    hybrid_105)                      echo "exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663" ;;
    hybrid_210)                      echo "exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663" ;;
    *) echo "" ;;
  esac
}

on_cluster() { [[ -d /cluster/work ]]; }

mkdir -p "$DEST"
want=0; got=0; missing=()

for cond in $CONDITIONS; do
  exp="$(exp_for "$cond")"
  if [[ -z "$exp" ]]; then
    echo "unknown condition: $cond" >&2
    echo "known: real_reference original_mse extended_mse perceptual_continuation" >&2
    echo "       strong_perceptual_continuation weighted_huber_transition" >&2
    echo "       weighted_huber_handoff pseudo_huber_perceptual_handoff" >&2
    echo "       hybrid_25 hybrid_50 hybrid_105 hybrid_210" >&2
    exit 2
  fi
  src="AIS4900_master/runs/downstream/nnunet/${exp}/eval_${exp}/predictions"
  mkdir -p "$DEST/$cond"
  echo "== $cond"
  for case_id in $CASES; do
    want=$((want + 1))
    if on_cluster; then
      base="$(eval echo "$CLUSTER_BASE")"
      if cp "$base/$src/${case_id}.nii.gz" "$DEST/$cond/" 2>/dev/null; then
        got=$((got + 1))
      else
        missing+=("$cond/$case_id")
      fi
    else
      : "${CLUSTER:?set CLUSTER to your ssh target, or run this on the cluster}"
      if rsync -q "${CLUSTER}:${CLUSTER_BASE}/$src/${case_id}.nii.gz" "$DEST/$cond/" 2>/dev/null; then
        got=$((got + 1))
      else
        missing+=("$cond/$case_id")
      fi
    fi
  done
  printf '   %s file(s) in %s\n' "$(find "$DEST/$cond" -name '*.nii.gz' | wc -l)" "$DEST/$cond"
done

echo
echo "fetched $got of $want"
if ((${#missing[@]})); then
  echo "MISSING:"
  printf '   %s\n' "${missing[@]}"
  echo "   check the directory in results_archive/*/manifest.json documented_prediction_dir"
fi

if on_cluster; then
  echo
  echo "now pull it down with:"
  echo "  rsync -avz <cluster>:$(pwd)/$DEST/ ./$DEST/"
fi

echo
echo "the panel also needs, already off-cluster:"
echo "  <dataset root>/brainmetshare-3/test/<case>/bravo.nii.gz   real volume"
echo "  <dataset root>/brainmetshare-3/test/<case>/seg.nii.gz     ground truth"
