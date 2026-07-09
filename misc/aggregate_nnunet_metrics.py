"""Aggregate downstream nnU-Net metrics across many experiments into one table.

Walks per-experiment `eval_*.json` files (written by
src/medgen/downstream/nnunet/evaluate.py) and collects EVERY metric into one CSV +
markdown table + JSON, so all exp3/6/7/8 numbers live in one place:

  - Volumetric Dice (primary), Grøvik Dice, lesion-wise Dice, slice-wise Dice (Ottesen)
  - Overall + size-stratified Dice/IoU (tiny/small/medium/large) with n_tumors per bin
  - Detection: rate AND detected/total COUNTS (overall + per size). Native counts
    (n_detected/n_total) are used when present; otherwise derived as round(rate*n_tumors).
  - False positives (overall + per size), precision/recall/HD95

Pure JSON parsing — no medgen import, so it runs in .venv_nnunet or any env.

When both `eval_X.json` and `eval_X_fixed.json` exist, the *_fixed* one is used
(it carries the Ottesen sagittal-axis slice-Dice fix + detection). Experiments whose
detection_metrics is empty are flagged (they need a re-eval to populate detection).

Usage:
    python misc/aggregate_nnunet_metrics.py                 # auto-walk runs/
    python misc/aggregate_nnunet_metrics.py PATH_OR_GLOB ... # explicit files/globs
    python misc/aggregate_nnunet_metrics.py --runs-dir /cluster/.../runs/downstream/nnunet
"""

import argparse
import csv
import glob
import json
import os

SIZES = ("tiny", "small", "medium", "large")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def experiment_name(path: str) -> str:
    """Derive an experiment label from an eval-json path."""
    base = os.path.basename(path)
    for pre in ("eval_",):
        if base.startswith(pre):
            base = base[len(pre):]
    for suf in ("_fixed.json", ".json"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return base


def discover(inputs: list[str], runs_dir: str | None) -> list[str]:
    """Resolve inputs (files/globs/dirs) to a deduped list of eval-json files.

    Prefers `*_fixed.json` over its plain sibling. Skips threshold/size-comparison JSONs.
    """
    candidates: list[str] = []
    patterns = list(inputs)
    if runs_dir:
        patterns += [os.path.join(runs_dir, "**", "eval_*.json")]
    if not patterns:
        patterns = [
            os.path.join(REPO, "runs", "**", "eval_*.json"),
            os.path.join(REPO, "runs", "downstream", "nnunet", "**", "eval_*.json"),
        ]
    for pat in patterns:
        if os.path.isfile(pat):
            candidates.append(pat)
        elif os.path.isdir(pat):
            candidates += glob.glob(os.path.join(pat, "**", "eval_*.json"), recursive=True)
        else:
            candidates += glob.glob(pat, recursive=True)

    # Drop non-eval JSONs and dedupe, preferring *_fixed.json per experiment.
    bad = ("threshold", "size_comparison", "per_fold", "predict_from_raw", "dataset", "plans", "fingerprint")
    by_exp: dict[str, str] = {}
    for path in sorted(set(candidates)):
        b = os.path.basename(path)
        if not b.startswith("eval_") or any(x in b for x in bad):
            continue
        exp = experiment_name(path)
        # Prefer a _fixed variant if we see one for the same experiment.
        if exp in by_exp and by_exp[exp].endswith("_fixed.json"):
            continue
        by_exp[exp] = path
    return [by_exp[e] for e in sorted(by_exp)]


def _detected_total(rate, native_det, native_tot, n_tumors):
    """Return (n_detected, n_total): native counts if present, else derived from rate."""
    if native_tot is not None:
        return native_det, native_tot
    if rate is None or n_tumors is None:
        return None, None
    return round(rate * n_tumors), n_tumors


def extract(path: str) -> dict:
    """Pull every metric of interest out of one eval-json into a flat row."""
    with open(path) as f:
        d = json.load(f)
    dv = d.get("dice_variants", {})
    rg = d.get("regional_metrics", {})
    det = d.get("detection_metrics", {})
    gl = d.get("global_metrics", {})

    row: dict = {
        "experiment": experiment_name(path),
        "eval_json": os.path.relpath(path, REPO),
        "num_cases": d.get("num_cases"),
        # --- Dice variants ---
        "vol_dice_mean": dv.get("dice_mean"),
        "vol_dice_std": dv.get("dice_std"),
        "grovik_dice_mean": dv.get("dice_grovik_mean"),
        "lesion_dice_bratsmets": dv.get("dice_lesionwise_bratsmets_mean"),
        "lesion_dice_regional": rg.get("dice"),
        "slice_dice_yi2023_mean": dv.get("dice_yi2023_slicewise_mean"),
        "slice_dice_yi2023_std": dv.get("dice_yi2023_slicewise_std"),
        # --- global ---
        "precision": gl.get("precision"),
        "recall": gl.get("recall"),
        "hd95": gl.get("hd95"),
        # --- overall lesions/detection ---
        "n_tumors_total": rg.get("n_tumors"),
        "detection_rate": det.get("detection_rate"),
        "false_positives": det.get("false_positives"),
        "detection_empty": len(det) == 0,
    }

    # Overall detected/total (native or derived).
    nd, nt = _detected_total(det.get("detection_rate"), det.get("n_detected"), det.get("n_total"), rg.get("n_tumors"))
    row["n_detected_total"] = nd
    row["n_total_total"] = nt

    # Per-size Dice/IoU/detection.
    for s in SIZES:
        row[f"dice_{s}"] = rg.get(f"dice_{s}")
        row[f"iou_{s}"] = rg.get(f"iou_{s}")
        n_tum = rg.get(f"n_tumors_{s}")
        row[f"n_tumors_{s}"] = n_tum
        row[f"detection_rate_{s}"] = det.get(f"detection_rate_{s}")
        row[f"fp_{s}"] = det.get(f"fp_{s}")
        nd_s, nt_s = _detected_total(
            det.get(f"detection_rate_{s}"), det.get(f"n_detected_{s}"), det.get(f"n_total_{s}"), n_tum
        )
        row[f"n_detected_{s}"] = nd_s
        row[f"n_total_{s}"] = nt_s
    return row


def _fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_markdown(rows: list[dict], out_md: str) -> None:
    """Compact headline table: Dice variants + detection detected/total per size."""
    cols = [
        ("experiment", "experiment"), ("vol_dice_mean", "VolDice"), ("lesion_dice_regional", "LesDice"),
        ("slice_dice_yi2023_mean", "SliceDice"), ("detection_rate", "Det"),
        ("n_detected_total", "det"), ("n_total_total", "tot"),
        ("n_detected_tiny", "tinyDet"), ("n_total_tiny", "tinyTot"),
        ("false_positives", "FP"),
    ]
    with open(out_md, "w") as f:
        f.write("| " + " | ".join(h for _, h in cols) + " |\n")
        f.write("|" + "|".join("---" for _ in cols) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(_fmt(r.get(k)) for k, _ in cols) + " |\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="eval-json files, globs, or dirs (default: walk runs/).")
    ap.add_argument("--runs-dir", default=None, help="Directory to recursively search for eval_*.json.")
    ap.add_argument("--out-csv", default=os.path.join(REPO, "runs/eval/downstream_metrics_all.csv"))
    ap.add_argument("--out-md", default=os.path.join(REPO, "runs/eval/downstream_metrics_all.md"))
    ap.add_argument("--out-json", default=os.path.join(REPO, "runs/eval/downstream_metrics_all.json"))
    args = ap.parse_args()

    files = discover(args.inputs, args.runs_dir)
    if not files:
        raise SystemExit("No eval_*.json files found. Pass paths/globs or --runs-dir.")
    print(f"Found {len(files)} experiment eval files:")
    for p in files:
        print(f"  {os.path.relpath(p, REPO)}")

    rows = [extract(p) for p in files]
    rows.sort(key=lambda r: r["experiment"])

    flagged = [r["experiment"] for r in rows if r.get("detection_empty")]
    if flagged:
        print(f"\nWARNING: detection_metrics EMPTY (need re-eval to populate): {flagged}")

    for out in (args.out_csv, args.out_md, args.out_json):
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    for r in rows:  # union of keys in case schemas differ
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    write_markdown(rows, args.out_md)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2)

    # Console preview.
    print(f"\n{'experiment':34s} {'VolDice':>8s} {'SliceDice':>9s} {'Det':>6s} {'det/tot':>10s} {'tinyDet/tot':>12s} {'FP':>5s}")
    for r in rows:
        det = f"{r['n_detected_total']}/{r['n_total_total']}" if r.get("n_total_total") is not None else "n/a"
        tiny = f"{r['n_detected_tiny']}/{r['n_total_tiny']}" if r.get("n_total_tiny") is not None else "n/a"
        print(f"{r['experiment']:34s} {_fmt(r['vol_dice_mean']):>8s} {_fmt(r['slice_dice_yi2023_mean']):>9s} "
              f"{_fmt(r['detection_rate']):>6s} {det:>10s} {tiny:>12s} {_fmt(r['false_positives'],0):>5s}")
    print(f"\nWrote {args.out_csv}\n      {args.out_md}\n      {args.out_json}")


if __name__ == "__main__":
    main()
