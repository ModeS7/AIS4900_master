"""Med3D-MMD of each set vs real reference(s), swept across ALL MedicalNet backbones.

Single-metric companion to eval_all_metrics.py: instead of many metrics against one
backbone, this runs the ONE metric (Med3D whole-volume Gaussian-MMD) against MANY
backbones — every MedicalNet 3D ResNet depth (10/18/34/50/101/152/200). eval_med3d_mmd.py
uses only ResNet-50; this answers "does the Med3D-MMD ranking hold across backbones, or
is it an artifact of the ResNet-50 feature space?".

For each depth: MMD(reference, dataset) in that backbone's feature space (BasicBlock
depths 10/18/34 -> 512-d, Bottleneck depths 50/101/152/200 -> 2048-d). Lower = closer to
real. All MMD math is reused verbatim from eval_med3d_mmd (``median_bandwidth``,
``mmd_with_ci``, path-streaming ``load_features``), so per-depth numbers match a
single-network run of that script. Multi-reference (repeatable --reference + --combine-all)
mirrors eval_all_metrics, because the reference distribution (test-51 / train-105 / all-156)
materially changes the MMD.

The RBF bandwidth is fixed once per (depth, reference) from the reference (median
heuristic) and reused for every dataset, so datasets are directly comparable within a
backbone. Bandwidths differ ACROSS backbones (different feature spaces) — compare rankings
across depths, not absolute MMD values.

OFFLINE WEIGHTS: only ResNet-50 is typically cached. The other six depths' weights
(``TencentMedicalNet/MedicalNet-Resnet{10,18,34,101,152,200}``) must be pre-downloaded on a
node with internet before an offline (HF_HUB_OFFLINE) compute-node run — see the SLURM
header. A startup precheck loads+unloads every requested depth and fails fast if a weight
set is missing.

Example:
    python -m medgen.scripts.eval_med3d_mmd_networks \
        --reference test:$HOME/MedicalDataSets/brainmetshare-3/test1 \
        --reference train:$HOME/MedicalDataSets/brainmetshare-3/train \
        --combine-all \
        --dataset exp1_1_imagenet:$HOME/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \
        --output med3d_mmd_networks.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch

from medgen.metrics.feature_extractors import Med3DFeatures
from medgen.scripts.eval_diversity import find_volumes, parse_dataset_arg
from medgen.scripts.eval_med3d_mmd import load_features, median_bandwidth, mmd_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_med3d_mmd_networks")


def network_label(depth: int) -> str:
    return f"resnet{depth}"


def precheck_weights(depths: list[int], device: torch.device, input_size) -> None:
    """Load+unload each backbone so missing offline weights fail fast (before the long run)."""
    logger.info("Prechecking weights for depths %s ...", depths)
    for depth in depths:
        ext = Med3DFeatures(device, input_size=input_size, compile_model=False, model_depth=depth)
        try:
            ext.load()
        except Exception as e:  # re-raise with an actionable, offline-aware message
            raise SystemExit(
                f"MedicalNet-Resnet{depth} weights not available ({e}). On an offline node they "
                f"must be pre-cached on a login node first:\n"
                f"  python -c \"from monai.networks.nets.resnet import get_pretrained_resnet_medicalnet as g; "
                f"g({depth})\""
            ) from e
        ext.unload()
    logger.info("Precheck OK — all %d backbones available.", len(depths))


def build_references(
    references, pattern, target_shape, extractor, device, batch_size, ref_cap, rng, combine_all,
):
    """Extract reference features + median-heuristic bandwidth for one backbone.

    Returns an ordered dict {label: {"feats", "bandwidth", "n"}}, adding a synthetic
    "all" reference (concatenation) when --combine-all and >1 reference are given.
    """
    ref_data: dict[str, dict] = {}
    for rlabel, rpath in references:
        feats, found = load_features(rpath, pattern, target_shape, extractor, device, batch_size, ref_cap, rng)
        ref_data[rlabel] = {"feats": feats, "bandwidth": median_bandwidth(feats), "n": found}
    if combine_all and len(references) > 1:
        combined = torch.cat([ref_data[lbl]["feats"] for lbl, _ in references], dim=0)
        ref_data["all"] = {
            "feats": combined,
            "bandwidth": median_bandwidth(combined),
            "n": sum(ref_data[lbl]["n"] for lbl, _ in references),
        }
    return ref_data


def score_network(
    depth, references, datasets, args, device, rng, subsample_n,
):
    """Run the full multi-reference MMD panel for ONE backbone depth."""
    input_size = tuple(args.med3d_input_size) if args.med3d_input_size else None
    extractor = Med3DFeatures(device, input_size=input_size, compile_model=False, model_depth=depth)
    target_shape = tuple(args.target_shape)

    ref_data = build_references(
        references, args.pattern, target_shape, extractor, device,
        args.batch_size, args.ref_cap, rng, args.combine_all,
    )
    ref_names = list(ref_data.keys())
    feature_dim = int(next(iter(ref_data.values()))["feats"].shape[1])
    logger.info("[%s] feature_dim=%d  references=%s", network_label(depth), feature_dim,
                {n: ref_data[n]["n"] for n in ref_names})

    net_result = {
        "feature_dim": feature_dim,
        "references": {n: ref_data[n]["n"] for n in ref_names},
        "bandwidth": {n: ref_data[n]["bandwidth"] for n in ref_names},
        "datasets": {},
    }

    for label, path in datasets:
        feats, total_found = load_features(
            path, args.pattern, target_shape, extractor, device, args.batch_size, args.pool_cap, rng,
        )
        per_ref = {}
        for rname in ref_names:
            res = mmd_with_ci(
                ref_data[rname]["feats"], feats, device, subsample_n,
                args.bootstrap, ref_data[rname]["bandwidth"], rng,
            )
            per_ref[rname] = res
        net_result["datasets"][label] = {
            "total_found": total_found,
            "pool_size": int(feats.shape[0]),
            "per_reference": per_ref,
        }
        del feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extractor.unload()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return net_result, ref_names


def print_tables(report, datasets, primary_ref):
    """Console: per-network detail + a cross-network full-MMD matrix vs the primary ref."""
    net_names = list(report["networks"].keys())
    ds_labels = [lbl for lbl, _ in datasets]

    for net in net_names:
        nr = report["networks"][net]
        ref_names = list(nr["references"].keys())
        print(f"\n=== {net} (dim={nr['feature_dim']}) — full MMD, LOWER = closer to real ===")
        header = f"{'dataset':28s} {'N':>4s} " + " ".join(f"{r[:9]:>10s}" for r in ref_names)
        print(header)
        for label in ds_labels:
            d = nr["datasets"].get(label)
            if d is None:
                continue
            cells = " ".join(
                f"{d['per_reference'][r]['full_mmd']:>10.5f}"
                if d['per_reference'][r]['full_mmd'] is not None else f"{'n/a':>10s}"
                for r in ref_names
            )
            print(f"{label:28s} {d['pool_size']:>4d} {cells}")

    # Cross-network matrix vs the primary reference — the rank-stability view.
    print(f"\n=== cross-network full MMD vs '{primary_ref}' (rows=datasets, cols=backbones) ===")
    print(f"{'dataset':28s} " + " ".join(f"{n.replace('resnet','r'):>9s}" for n in net_names))
    for label in ds_labels:
        cells = []
        for net in net_names:
            d = report["networks"][net]["datasets"].get(label)
            v = d["per_reference"].get(primary_ref, {}).get("full_mmd") if d else None
            cells.append(f"{v:>9.5f}" if v is not None else f"{'n/a':>9s}")
        print(f"{label:28s} " + " ".join(cells))

    # Per-network best (lowest-MMD) dataset vs primary ref — quick rank-agreement check.
    print(f"\n=== best (lowest-MMD) dataset per backbone vs '{primary_ref}' ===")
    for net in net_names:
        dsets = report["networks"][net]["datasets"]
        ranked = sorted(
            ((lbl, d["per_reference"].get(primary_ref, {}).get("full_mmd")) for lbl, d in dsets.items()),
            key=lambda kv: (kv[1] is None, kv[1] if kv[1] is not None else 0.0),
        )
        order = " > ".join(lbl for lbl, _ in ranked)
        print(f"  {net:12s}: {order}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", action="append", required=True, metavar="LABEL:PATH", help="Real reference (repeatable, e.g. test/train).")
    ap.add_argument("--combine-all", action="store_true", help="Add a synthetic 'all' reference = concatenation of all --reference sets (e.g. train+test=156).")
    ap.add_argument("--dataset", action="append", required=True, metavar="LABEL:PATH", help="Dataset to score as LABEL:PATH (repeatable).")
    ap.add_argument("--networks", nargs="+", type=int, default=list(Med3DFeatures.SUPPORTED_DEPTHS), metavar="DEPTH", help="MedicalNet ResNet depths to sweep (default: all 7).")
    ap.add_argument("--pattern", default="bravo.nii.gz", help="Filename glob searched recursively per set.")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[150, 256, 256], metavar=("D", "H", "W"), help="Common (D H W) all volumes are resized to before Med3D.")
    ap.add_argument("--med3d-input-size", nargs=3, type=int, default=None, metavar=("D", "H", "W"), help="Optional Med3D internal resize grid (default None = target-shape).")
    ap.add_argument("--pool-cap", type=int, default=256, help="Max volumes per dataset (random subset if more; 0 = all).")
    ap.add_argument("--ref-cap", type=int, default=0, help="Cap reference volumes (0 = use all).")
    ap.add_argument("--subsample-n", type=int, default=51, help="Common N for the comparable estimate. 0 = min dataset size.")
    ap.add_argument("--bootstrap", type=int, default=30, help="Bootstrap resamples for the CI (0 = full MMD only, faster).")
    ap.add_argument("--batch-size", type=int, default=2, help="Med3D extraction batch size.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-precheck", action="store_true", help="Skip the fail-fast weight precheck.")
    ap.add_argument("--output", default="med3d_mmd_networks_report.json")
    args = ap.parse_args()

    depths = list(dict.fromkeys(args.networks))  # dedupe, preserve order
    bad = [d for d in depths if d not in Med3DFeatures.SUPPORTED_DEPTHS]
    if bad:
        raise SystemExit(f"Unsupported depths {bad}; choose from {Med3DFeatures.SUPPORTED_DEPTHS}.")

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    input_size = tuple(args.med3d_input_size) if args.med3d_input_size else None
    references = [parse_dataset_arg(s) for s in args.reference]
    datasets = [parse_dataset_arg(s) for s in args.dataset]
    primary_ref = references[0][0]

    logger.info("Device=%s  depths=%s  target_shape=%s", device, depths, tuple(args.target_shape))

    if not args.skip_precheck:
        precheck_weights(depths, device, input_size)

    # Common N defaults to the smallest capped dataset (matches eval_med3d_mmd).
    counts = {lbl: min(len(find_volumes(p, args.pattern)), args.pool_cap or 10**9) for lbl, p in datasets}
    subsample_n = args.subsample_n or min(counts.values())
    logger.info("Effective pools: %s | common subsample N = %d", counts, subsample_n)

    report = {
        "config": {
            "references": [f"{lbl}:{p}" for lbl, p in references],
            "combine_all": args.combine_all,
            "primary_reference": primary_ref,
            "networks": depths,
            "target_shape": list(args.target_shape),
            "med3d_input_size": list(input_size) if input_size else None,
            "pool_cap": args.pool_cap, "ref_cap": args.ref_cap,
            "subsample_n": subsample_n, "bootstrap": args.bootstrap,
            "pattern": args.pattern, "seed": args.seed,
        },
        "networks": {},
    }

    for depth in depths:
        logger.info("==== Backbone resnet%d ====", depth)
        net_result, _ = score_network(depth, references, datasets, args, device, rng, subsample_n)
        report["networks"][network_label(depth)] = net_result

    print_tables(report, datasets, primary_ref)
    print(f"\n(full MMD headline; mean@N / 95% CI at common N={subsample_n}, {args.bootstrap} bootstrap draws)")
    print("NOTE: bandwidths differ across backbones — compare RANKINGS across depths, not absolute MMD.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
