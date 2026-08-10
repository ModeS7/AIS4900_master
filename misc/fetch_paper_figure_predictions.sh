#!/bin/bash
# One command, one file out. Fetches every prediction the paper's segmentation figures
# need and leaves a single tarball to carry off the cluster.
#
#   ./misc/fetch_paper_figure_predictions.sh
#
# Then move the one file it names. Nothing else to copy, no env vars to paste.
#
# WHAT IT COLLECTS
#   15 cases x 12 conditions = 180 masks, about 2.5 MB packed.
#
#   FLOOR CASES (9) -- every training source scores exactly zero volumetric Dice on
#   these. They are the subject of their own appendix gallery, because the failure is not
#   uniform: six carry a non-empty prediction that misses the reference entirely, two
#   predict nothing at all, and one flips between the two depending on condition.
#
#   DETECTED CASES (6) -- six evenly spaced ranks over the 42 patients that some model
#   does detect, sorted by real-reference Dice. Drawn from the detected set rather than
#   all 51 on purpose: sampling the whole cohort would land on floor cases by chance, and
#   a gallery meant to show how quality varies between training sources would then carry
#   rows of blank contours that belong in the other gallery.
#
#   All twelve conditions, so the appendix can show reference against all seven synthetic
#   sources and reference against all four hybrid levels without a second trip.
#
# Case selection is deterministic: results_archive/SEG7_job_24929214/per_case.csv sorted
# by real-reference Dice, evenly spaced ranks. Same rule as the generation gallery.

set -Eeuo pipefail

cd "$(dirname "$0")/.."

CASES="BrainMet_038 BrainMet_047 BrainMet_063 BrainMet_087 BrainMet_112 BrainMet_143 \
BrainMet_183 BrainMet_192 BrainMet_232 BrainMet_257 BrainMet_277 BrainMet_306 \
BrainMet_312 BrainMet_316 BrainMet_321"

CONDITIONS="real_reference original_mse extended_mse perceptual_continuation \
strong_perceptual_continuation weighted_huber_transition weighted_huber_handoff \
pseudo_huber_perceptual_handoff hybrid_25 hybrid_50 hybrid_105 hybrid_210"

DEST="${DEST:-runs/eval/downstream_predictions}"
TARBALL="${TARBALL:-$HOME/downstream_predictions.tgz}"

CASES="$CASES" CONDITIONS="$CONDITIONS" DEST="$DEST" \
  ./misc/fetch_downstream_predictions.sh

n=$(find "$DEST" -name '*.nii.gz' | wc -l)
tar czf "$TARBALL" -C "$(dirname "$DEST")" "$(basename "$DEST")"

echo
echo "=========================================================================="
echo " $n masks packed into ONE file:"
echo
echo "   $TARBALL   ($(du -h "$TARBALL" | cut -f1))"
echo
echo " Copy that file off the cluster however you like, then unpack it next to"
echo " the MedGen checkout on your workstation:"
echo
echo "   tar xzf downstream_predictions.tgz -C <local-AIS4900_master>/runs/eval/"
echo "=========================================================================="
