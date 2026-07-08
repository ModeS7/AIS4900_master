"""Full generation-metric panel for each synthetic set vs MULTIPLE real references.

Per dataset, computed against every --reference (e.g. test-51 / train-105 / all-156):
FID/KID (ImageNet + RadImageNet, 2D axial slices), CMMD (BiomedCLIP), Med3D-MMD (true-3D
whole-volume). Plus reference-INDEPENDENT metrics computed once: PCA brain-shape pass-rate
+ mean error, and intra-set MS-SSIM diversity.

WHY multiple references: the step counts that produced these datasets were chosen by
find_optimal_steps against `--ref-split test` (51), the downstream nnU-Net is evaluated on
test (51), but a naive panel scored against train (105) — three different references. FID
is very reference-sensitive (distribution AND sample size), so the metric↔downstream story
must be checked against the SAME reference. This runs all three and extracts the (expensive)
generated features ONCE, reusing them across references.

Generated features use ALL volumes by default (--pool-cap 0). Reference and generated
volumes go through the SAME extraction functions, so FID/KID/CMMD/MMD are matched. The
Med3D RBF bandwidth is fixed per reference (median heuristic from that reference). Diversity
is O(N^2) in pairs, so it is computed on a --diversity-cap subsample (N-stable — verified).

Example:
    python -m medgen.scripts.eval_all_metrics \
        --reference test:$HOME/MedicalDataSets/brainmetshare-3/test1 \
        --reference train:$HOME/MedicalDataSets/brainmetshare-3/train \
        --combine-all \
        --dataset exp1_1_imagenet:$HOME/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \
        --pool-cap 0 --output all_metrics.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from medgen.metrics.brain_mask import BrainPCAModel, create_brain_mask
from medgen.metrics.feature_extractors import (
    BiomedCLIPFeatures,
    Med3DFeatures,
    ResNet50Features,
)
from medgen.metrics.generation import compute_cmmd, compute_fid, compute_kid
from medgen.metrics.quality import compute_msssim
from medgen.scripts.eval_diversity import (
    find_volumes,
    load_volume,
    pairwise_diversity_3d,
    parse_dataset_arg,
)
from medgen.scripts.eval_med3d_mmd import median_bandwidth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_all_metrics")

FEAT_KEYS = ("in", "rin", "clip", "med3d")


# =============================================================================
# Loading + extraction
# =============================================================================

def load_stacked(paths, target_shape):
    """Load volumes into a [B, 1, D, H, W] CPU tensor (all [0,1]-normalized)."""
    return torch.stack([load_volume(p, target_shape) for p in tqdm(paths, desc="load", leave=False)])


def to_numpy_list(vols):
    """[B, 1, D, H, W] tensor -> list of [D, H, W] numpy views (no copy)."""
    return [vols[i, 0].numpy() for i in range(vols.shape[0])]


@torch.no_grad()
def med3d_from_tensor(vols, extractor, device, batch_size):
    """Extract [N, 2048] Med3D features from a stacked tensor, in batches."""
    feats = []
    for s in range(0, vols.shape[0], batch_size):
        feats.append(extractor.extract_features(vols[s:s + batch_size].to(device)).cpu())
    return torch.cat(feats, dim=0)


def extract_all_features(vols, extractors, trim_slices, device, med3d_batch):
    """Extract IN/RIN/CLIP slice features + Med3D whole-volume features for one set."""
    from medgen.scripts.eval_ode_solvers import extract_generated_features
    np_list = to_numpy_list(vols)
    return {
        "in": extract_generated_features(np_list, extractors["in"], trim_slices),
        "rin": extract_generated_features(np_list, extractors["rin"], trim_slices),
        "clip": extract_generated_features(np_list, extractors["clip"], trim_slices),
        "med3d": med3d_from_tensor(vols, extractors["med3d"], device, med3d_batch),
    }


# =============================================================================
# Metrics
# =============================================================================

def pca_stats(vols, pca_model, threshold):
    """Brain-shape PCA pass-rate + mean reconstruction error over a set's volumes."""
    errors, passes = [], 0
    for i in range(vols.shape[0]):
        vol_np = vols[i, 0].numpy()
        mask = create_brain_mask(vol_np, threshold=threshold, fill_holes=True, dilate_pixels=0)
        ok, err = pca_model.is_valid(mask.astype(np.float32))
        errors.append(err)
        passes += int(ok)
    return passes / vols.shape[0], float(np.mean(errors))


def two_sample_metrics(gen_feats, ref_feats, device, bandwidth):
    """FID/KID (IN+RIN), CMMD, Med3D-MMD of one dataset vs ONE reference."""
    kid_sub = min(100, gen_feats["in"].shape[0], ref_feats["in"].shape[0])
    kid_in_m, kid_in_s = compute_kid(ref_feats["in"], gen_feats["in"], subset_size=kid_sub)
    kid_rin_m, kid_rin_s = compute_kid(ref_feats["rin"], gen_feats["rin"], subset_size=kid_sub)
    return {
        "fid_imagenet": compute_fid(ref_feats["in"], gen_feats["in"]),
        "kid_imagenet": kid_in_m, "kid_imagenet_std": kid_in_s,
        "fid_radimagenet": compute_fid(ref_feats["rin"], gen_feats["rin"]),
        "kid_radimagenet": kid_rin_m, "kid_radimagenet_std": kid_rin_s,
        "cmmd": compute_cmmd(ref_feats["clip"], gen_feats["clip"]),
        "med3d_mmd": compute_cmmd(
            ref_feats["med3d"].to(device), gen_feats["med3d"].to(device), kernel_bandwidth=bandwidth
        ),
    }


def reference_independent(vols, device, pca_model, pca_threshold, diversity_cap, rng):
    """Intra-set MS-SSIM diversity (on a cap; N-stable) + PCA over all volumes."""
    B = vols.shape[0]
    if diversity_cap and diversity_cap < B:
        idx = sorted(rng.choice(B, size=diversity_cap, replace=False))
        div_vols = vols[idx]
    else:
        div_vols = vols
    div = pairwise_diversity_3d(
        div_vols, device,
        lambda g, r: compute_msssim(g, r, data_range=1.0, spatial_dims=2),
        is_similarity=True,
    )
    res = {"msssim_diversity": div, "diversity_n": int(div_vols.shape[0])}
    if pca_model is not None:
        pass_rate, mean_err = pca_stats(vols, pca_model, pca_threshold)
        res["pca_pass_rate"] = pass_rate
        res["pca_mean_error"] = mean_err
    return res


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", action="append", required=True, metavar="LABEL:PATH", help="Real reference (repeatable, e.g. test/train).")
    ap.add_argument("--combine-all", action="store_true", help="Add a synthetic 'all' reference = concatenation of all --reference sets (e.g. train+test=156).")
    ap.add_argument("--dataset", action="append", required=True, metavar="LABEL:PATH", help="Synthetic dataset to score (repeatable).")
    ap.add_argument("--pattern", default="bravo.nii.gz")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[150, 256, 256], metavar=("D", "H", "W"))
    ap.add_argument("--pool-cap", type=int, default=0, help="Max volumes per synthetic set (0 = ALL, e.g. all 525).")
    ap.add_argument("--ref-cap", type=int, default=0, help="Cap reference volumes (0 = all).")
    ap.add_argument("--diversity-cap", type=int, default=256, help="Volumes used for the O(N^2) diversity (N-stable; 0 = all).")
    ap.add_argument("--trim-slices", type=int, default=0)
    ap.add_argument("--med3d-batch", type=int, default=2)
    ap.add_argument("--pca-model", default="data/brain_pca_256x256x160.npz", help="Brain-shape PCA npz ('none' to skip).")
    ap.add_argument("--pca-threshold", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="all_metrics_report.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    target_shape = tuple(args.target_shape)
    references = [parse_dataset_arg(s) for s in args.reference]
    datasets = [parse_dataset_arg(s) for s in args.dataset]

    pca_model = None
    if args.pca_model and args.pca_model.lower() != "none":
        pca_model = BrainPCAModel(args.pca_model)
        logger.info("Brain PCA loaded (threshold=%.6f)", pca_model.error_threshold)

    logger.info("Device=%s  target_shape=%s  loading extractors...", device, target_shape)
    extractors = {
        "in": ResNet50Features(device, network_type="imagenet", compile_model=False),
        "rin": ResNet50Features(device, network_type="radimagenet", compile_model=False),
        "clip": BiomedCLIPFeatures(device, compile_model=False),
        "med3d": Med3DFeatures(device, input_size=None, compile_model=False),
    }

    def _capped_paths(path, cap):
        paths = find_volumes(path, args.pattern)
        total = len(paths)
        if total < 2:
            raise ValueError(f"Need >=2 volumes, found {total} under {path}")
        if cap and total > cap:
            keep = sorted(rng.choice(total, size=cap, replace=False))
            paths = [paths[i] for i in keep]
            logger.warning("%s: %d found, using a random %d (cap).", path, total, cap)
        return paths, total

    # Reference features (each extracted once).
    ref_data = {}
    for rlabel, rpath in references:
        logger.info("Reference %-10s <- %s", rlabel, rpath)
        rpaths, rfound = _capped_paths(rpath, args.ref_cap)
        rvols = load_stacked(rpaths, target_shape)
        rfeats = extract_all_features(rvols, extractors, args.trim_slices, device, args.med3d_batch)
        ref_data[rlabel] = {"feats": rfeats, "bandwidth": median_bandwidth(rfeats["med3d"]), "n": rfound}
        del rvols
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.combine_all and len(references) > 1:
        combined = {k: torch.cat([ref_data[lbl]["feats"][k] for lbl, _ in references], dim=0) for k in FEAT_KEYS}
        ref_data["all"] = {
            "feats": combined,
            "bandwidth": median_bandwidth(combined["med3d"]),
            "n": sum(ref_data[lbl]["n"] for lbl, _ in references),
        }
    ref_names = list(ref_data.keys())
    logger.info("References: %s", {n: ref_data[n]["n"] for n in ref_names})

    report = {
        "config": {
            "references": {n: ref_data[n]["n"] for n in ref_names},
            "med3d_bandwidth": {n: ref_data[n]["bandwidth"] for n in ref_names},
            "target_shape": list(target_shape), "pool_cap": args.pool_cap,
            "diversity_cap": args.diversity_cap, "trim_slices": args.trim_slices,
            "pca_model": args.pca_model, "pattern": args.pattern, "seed": args.seed,
        },
        "datasets": {},
    }

    for label, path in datasets:
        logger.info("Scoring %-24s <- %s", label, path)
        paths, total_found = _capped_paths(path, args.pool_cap)
        vols = load_stacked(paths, target_shape)
        gen_feats = extract_all_features(vols, extractors, args.trim_slices, device, args.med3d_batch)
        per_ref = {
            rname: two_sample_metrics(gen_feats, ref_data[rname]["feats"], device, ref_data[rname]["bandwidth"])
            for rname in ref_names
        }
        ri = reference_independent(vols, device, pca_model, args.pca_threshold, args.diversity_cap, rng)
        report["datasets"][label] = {"per_reference": per_ref, "total_found": total_found, "pool_size": int(vols.shape[0]), **ri}
        del vols, gen_feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for e in extractors.values():
        if hasattr(e, "unload"):
            e.unload()

    # Console: one table per reference (FID/CMMD/Med3D) + one reference-independent table.
    for rname in ref_names:
        print(f"\n=== vs {rname} (N_ref={ref_data[rname]['n']}) — FID/CMMD/Med3D lower=better ===")
        print(f"{'dataset':28s} {'N':>4s} {'FID_IN':>8s} {'FID_RIN':>8s} {'CMMD':>8s} {'MED3D':>8s}")
        for label, r in report["datasets"].items():
            m = r["per_reference"][rname]
            print(f"{label:28s} {r['pool_size']:>4d} {m['fid_imagenet']:>8.2f} {m['fid_radimagenet']:>8.2f} "
                  f"{m['cmmd']:>8.4f} {m['med3d_mmd']:>8.4f}")

    print("\n=== reference-independent (diversity higher=better; PCA_err lower=better) ===")
    print(f"{'dataset':28s} {'N':>4s} {'divN':>5s} {'MSSSIM_div':>11s} {'PCA%':>6s} {'PCA_err':>9s}")
    for label, r in report["datasets"].items():
        pca = f"{r['pca_pass_rate']*100:.0f}" if "pca_pass_rate" in r else "n/a"
        pcae = f"{r['pca_mean_error']:.5f}" if "pca_mean_error" in r else "n/a"
        print(f"{label:28s} {r['pool_size']:>4d} {r.get('diversity_n', 0):>5d} {r['msssim_diversity']:>11.4f} {pca:>6s} {pcae:>9s}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
