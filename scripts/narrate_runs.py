"""Helpers to render per-experiment narratives from runs_tb_extracted.json.

Usage (interactive):
    from scripts.narrate_runs import load_records, runs_in, render_run
    all_recs = load_records("runs_tb_extracted.json")
    for r in runs_in(all_recs, category="diffusion_3d", mode="bravo"):
        print(render_run(r))

The narrative emitter (render_run) reads whatever scalar tags exist in a
record and emits markdown. Diffusion-specific, compression-specific, and
downstream-specific sections are selected automatically based on tag
presence — the script does not hard-code which run is which type.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Tag groups (first tag found wins within a group for headline reporting)
LOSS_TAGS = [
    "Loss/MSE_train", "Loss/MSE_val",
    "Loss/total_train", "Loss/total_val",
    "Loss/L1_train", "Loss/L1_val",
    "Loss/huber_train", "Loss/huber_val",
    "Loss/perceptual_train", "Loss/perceptual_val",
    "Loss/lpips_train", "Loss/lpips_val",
    "Loss/ffl_train", "Loss/ffl_val",
    "Loss/focal_frequency_train", "Loss/focal_frequency_val",
    "Loss/recon", "Loss/kl", "Loss/codebook", "Loss/commit",
    "Loss/discriminator", "Loss/adversarial",
    "Loss/grad_accum",
    "training/AuxBin",
]

GEN_TAGS = [
    "Generation/KID_mean_val", "Generation/KID_mean_train",
    "Generation/KID_std_val", "Generation/KID_std_train",
    "Generation/CMMD_val", "Generation/CMMD_train",
    "Generation/FID_val", "Generation/FID_train",
    "Generation/extended_KID_mean_val", "Generation/extended_KID_mean_train",
    "Generation/extended_CMMD_val", "Generation/extended_CMMD_train",
    "Generation/extended_FID_val", "Generation/extended_FID_train",
]

DIVERSITY_TAGS = [
    "Generation_Diversity/extended_LPIPS",
    "Generation_Diversity/extended_MSSSIM",
    "Generation_Diversity/LPIPS",
    "Generation_Diversity/MSSSIM",
]

VAL_QUALITY_TAGS_PREFIXES = ["Validation/"]
REGIONAL_PREFIXES = ["regional_", "regional/"]
TIMESTEP_PREFIX = "Timestep/"
PCA_PREFIXES = ["PCA/"]
MORPH_PREFIXES = ["Morph/"]
LR_PREFIX = "LR/"
VRAM_PREFIX = "VRAM/"
FLOPS_PREFIX = "FLOPs/"
TRAINING_PREFIX = "training/"

DICE_PREFIXES = ["Dice/", "HD95/", "NSD/", "val/Dice", "val/HD95"]

# Downstream seg metric tag heuristics — matches nnU-Net / SegResNet format:
#   val/loss, val/mean_fg_dice, test/dice, test/hd95, test/precision, test/recall
DOWNSTREAM_METRIC_SUFFIXES = (
    "/dice", "/dice_tiny", "/dice_small", "/dice_medium", "/dice_large",
    "/hd95", "/iou", "/precision", "/recall", "/loss", "/mean_fg_dice",
    "/ema_fg_dice", "/dice_class_0", "/learning_rate",
)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def runs_in(records: list[dict], category: str | None = None, mode: str | None = None) -> list[dict]:
    out = records
    if category is not None:
        out = [r for r in out if r.get("category") == category]
    if mode is not None:
        out = [r for r in out if r.get("mode") == mode]
    return sorted(out, key=lambda r: (r.get("run_stem", ""), r.get("timestamp", "")))


def _sv(record: dict, tag: str, field: str = "last") -> tuple[int, float] | None:
    """Return (step, value) for a scalar tag if present, else None."""
    s = record.get("scalars", {}).get(tag)
    if not isinstance(s, dict) or s.get("n", 0) == 0:
        return None
    v = s.get(field)
    if v is None:
        return None
    return (int(v[0]), float(v[1]))


def _fmt_sv(sv: tuple[int, float] | None, digits: int = 4) -> str:
    if sv is None:
        return "—"
    step, val = sv
    if abs(val) < 1e-3 or abs(val) > 1e4:
        return f"{val:.{digits}g} @ ep {step}"
    return f"{val:.{digits}f} @ ep {step}"


def _fmt_float(v: float | None, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    if abs(v) < 1e-3 or abs(v) > 1e4:
        return f"{v:.{digits}g}"
    return f"{v:.{digits}f}"


def _hms(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _parse_ts(ts: str | None) -> str:
    if not ts:
        return "?"
    try:
        return datetime.strptime(ts, "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def walltime_span(record: dict) -> float:
    """Max wall_time_span across all scalar tags (a proxy for total training time)."""
    best = 0.0
    for s in record.get("scalars", {}).values():
        if isinstance(s, dict):
            best = max(best, float(s.get("wall_time_span_s", 0) or 0))
    return best


def epochs_reached(record: dict) -> int | None:
    """Last step of the first loss-like tag we find (usually per-epoch)."""
    for tag in ["Loss/MSE_train", "Loss/total_train", "Loss/L1_train", "Loss/recon"]:
        sv = _sv(record, tag, "last")
        if sv is not None:
            return sv[0] + 1  # step 0-indexed → epoch count
    # fallback: max last step across any tag
    steps = [
        s["last"][0]
        for s in record.get("scalars", {}).values()
        if isinstance(s, dict) and s.get("last")
    ]
    return max(steps) + 1 if steps else None


def peak_vram(record: dict) -> float | None:
    sv = _sv(record, "VRAM/max_allocated_GB", "max")
    return sv[1] if sv else None


def total_tflops(record: dict) -> float | None:
    sv = _sv(record, "FLOPs/TFLOPs_total", "last")
    return sv[1] if sv else None


def render_run(record: dict, memory_hint: str | None = None) -> str:
    """Emit a markdown narrative block for one run.

    Args:
        record: the JSON record from runs_tb_extracted.json.
        memory_hint: optional pre-written sentence to lead the block with
            (for runs that have matching memory entries).
    """
    lines: list[str] = []
    name = record["run_name"]
    scalars = record.get("scalars", {})
    tags = set(record.get("scalar_tags", []))

    # --- Header line ---
    lines.append(f"#### `{name}`")
    ts = _parse_ts(record.get("timestamp"))
    ep = epochs_reached(record)
    wt = walltime_span(record)
    vram = peak_vram(record)
    tflops = total_tflops(record)
    meta_bits = [f"started {ts}"]
    if ep: meta_bits.append(f"{ep} epochs")
    if wt > 0: meta_bits.append(_hms(wt))
    if tflops: meta_bits.append(f"{tflops:.1f} TFLOPs")
    if vram: meta_bits.append(f"peak VRAM {vram:.1f} GB")
    lines.append("*" + " • ".join(meta_bits) + "*")
    lines.append("")

    if memory_hint:
        lines.append(f"**Context:** {memory_hint}")
        lines.append("")

    # --- Error / empty guard ---
    errors = record.get("errors") or []
    if record.get("scalar_count", 0) == 0:
        if errors:
            lines.append(f"**Empty run** — {errors[0]}")
        else:
            lines.append("**Empty run** — no scalar tags written.")
        lines.append("")
        return "\n".join(lines)

    # --- Loss dynamics ---
    loss_bits: list[str] = []
    for tag in LOSS_TAGS:
        if tag in tags:
            last = _sv(record, tag, "last")
            mn = _sv(record, tag, "min")
            first = _sv(record, tag, "first")
            if last and first:
                loss_bits.append(
                    f"  - `{tag}`: {_fmt_float(first[1])} → {_fmt_float(last[1])}"
                    + (f" (min {_fmt_float(mn[1])} @ ep {mn[0]})" if mn else "")
                )
    if loss_bits:
        lines.append("**Loss dynamics:**")
        lines.extend(loss_bits)
        lines.append("")

    # --- Per-timestep buckets (diffusion) ---
    ts_tags = sorted([t for t in tags if t.startswith(TIMESTEP_PREFIX)])
    if ts_tags:
        lines.append("**Per-timestep MSE (final value per bucket):**")
        for t in ts_tags:
            last = _sv(record, t, "last")
            if last:
                label = t.split("/", 1)[1]
                lines.append(f"  - {label}: {_fmt_float(last[1])}")
        lines.append("")

    # --- Generation metrics ---
    gen_bits: list[str] = []
    for tag in GEN_TAGS:
        if tag in tags:
            last = _sv(record, tag, "last")
            mn = _sv(record, tag, "min")
            if last:
                gen_bits.append(
                    f"  - `{tag}`: last {_fmt_float(last[1])}"
                    + (f", best {_fmt_float(mn[1])} @ ep {mn[0]}" if mn else "")
                )
    if gen_bits:
        lines.append("**Generation metrics:**")
        lines.extend(gen_bits)
        lines.append("")

    # --- Validation quality ---
    val_tags = sorted([t for t in tags if any(t.startswith(p) for p in VAL_QUALITY_TAGS_PREFIXES)])
    if val_tags:
        lines.append("**Validation quality:**")
        for t in val_tags:
            last = _sv(record, t, "last")
            mn = _sv(record, t, "min")
            mx = _sv(record, t, "max")
            if last is None:
                continue
            lines.append(
                f"  - `{t}`: last {_fmt_float(last[1])}"
                + (f" (min {_fmt_float(mn[1])}, max {_fmt_float(mx[1])})" if mn and mx else "")
            )
        lines.append("")

    # --- Diversity ---
    div_bits: list[str] = []
    for tag in DIVERSITY_TAGS:
        if tag in tags:
            last = _sv(record, tag, "last")
            if last:
                div_bits.append(f"  - `{tag}`: {_fmt_float(last[1])}")
    if div_bits:
        lines.append("**Diversity (extended):**")
        lines.extend(div_bits)
        # Collapse warning
        lpips = _sv(record, "Generation_Diversity/extended_LPIPS", "last")
        if lpips and lpips[1] < 0.05:
            lines.append("  - ⚠️ possible mode collapse (inter-sample LPIPS < 0.05)")
        lines.append("")

    # --- Regional loss ---
    reg_tags = sorted([t for t in tags if any(t.startswith(p) for p in REGIONAL_PREFIXES)])
    if reg_tags:
        lines.append("**Regional loss (final):**")
        for t in reg_tags:
            last = _sv(record, t, "last")
            if last:
                lines.append(f"  - `{t}`: {_fmt_float(last[1])}")
        lines.append("")

    # --- PCA / morphological ---
    pca_tags = sorted([t for t in tags if any(t.startswith(p) for p in PCA_PREFIXES)])
    if pca_tags:
        lines.append("**PCA metrics:**")
        for t in pca_tags:
            last = _sv(record, t, "last")
            mn = _sv(record, t, "min")
            mx = _sv(record, t, "max")
            if last:
                extra = ""
                if "error" in t.lower() and mn:
                    extra = f" (best {_fmt_float(mn[1])} @ ep {mn[0]})"
                elif "pass_rate" in t.lower() and mx:
                    extra = f" (peak {_fmt_float(mx[1])} @ ep {mx[0]})"
                lines.append(f"  - `{t}`: {_fmt_float(last[1])}{extra}")
        lines.append("")

    morph_tags = sorted([t for t in tags if any(t.startswith(p) for p in MORPH_PREFIXES)])
    if morph_tags:
        lines.append("**Morphological metrics:**")
        for t in morph_tags:
            last = _sv(record, t, "last")
            if last:
                lines.append(f"  - `{t}`: {_fmt_float(last[1])}")
        lines.append("")

    # --- Downstream seg metrics (nnU-Net / SegResNet) ---
    # Match any tag ending in a recognized downstream suffix, for tag roots
    # like "test/", "val/", or nested "Dataset501_.../fold_0/...::test/..."
    ds_tags = sorted([
        t for t in tags
        if any(t.endswith(suf) for suf in DOWNSTREAM_METRIC_SUFFIXES)
    ])
    if ds_tags:
        lines.append("**Downstream seg metrics:**")
        for t in ds_tags:
            last = _sv(record, t, "last")
            mx = _sv(record, t, "max")
            mn = _sv(record, t, "min")
            if last is None:
                continue
            # Choose best direction: loss ↓, dice/iou/precision/recall ↑, hd95 ↓
            is_down = any(t.endswith(x) for x in ("/loss", "/hd95", "/learning_rate"))
            best = mn if is_down else mx
            extra = ""
            if best and best != last:
                extra = f" ({'min' if is_down else 'max'} {_fmt_float(best[1])} @ ep {best[0]})"
            lines.append(f"  - `{t}`: last {_fmt_float(last[1])}{extra}")
        lines.append("")

    # --- Downstream Dice/HD95 (legacy prefix-based matcher) ---
    dice_tags = sorted([t for t in tags if any(t.startswith(p) for p in DICE_PREFIXES)])
    if dice_tags:
        lines.append("**Downstream segmentation metrics:**")
        for t in dice_tags:
            last = _sv(record, t, "last")
            mx = _sv(record, t, "max")
            if last:
                lines.append(
                    f"  - `{t}`: last {_fmt_float(last[1])}"
                    + (f", best {_fmt_float(mx[1])} @ ep {mx[0]}" if mx else "")
                )
        lines.append("")

    # --- LR schedule ---
    lr_tags = sorted([t for t in tags if t.startswith(LR_PREFIX)])
    if lr_tags:
        bits = []
        for t in lr_tags:
            first = _sv(record, t, "first")
            last = _sv(record, t, "last")
            mx = _sv(record, t, "max")
            if last and mx:
                bits.append(
                    f"  - `{t}`: peak {_fmt_float(mx[1])} @ ep {mx[0]}, final {_fmt_float(last[1])}"
                )
        if bits:
            lines.append("**LR schedule:**")
            lines.extend(bits)
            lines.append("")

    # --- Training meta (AuxBin, grad norm, param norm) ---
    meta_tags = sorted([
        t for t in tags
        if t.startswith(TRAINING_PREFIX) and t not in LOSS_TAGS
    ])
    if meta_tags:
        bits = []
        for t in meta_tags:
            last = _sv(record, t, "last")
            mx = _sv(record, t, "max")
            if last:
                bits.append(f"  - `{t}`: last {_fmt_float(last[1])}"
                            + (f", max {_fmt_float(mx[1])} @ ep {mx[0]}" if mx else ""))
        if bits:
            lines.append("**Training meta:**")
            lines.extend(bits)
            lines.append("")

    # --- NaN flag ---
    nans = [
        t for t, s in scalars.items()
        if isinstance(s, dict) and s.get("nan_count", 0) > 0
    ]
    if nans:
        lines.append(f"**⚠️ NaN detected in {len(nans)} tag(s):** {', '.join(nans[:5])}"
                     + (f" (+{len(nans)-5} more)" if len(nans) > 5 else ""))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_family(records: list[dict], family_prefix: str) -> str:
    """Render all runs whose run_stem starts with family_prefix, sorted by timestamp."""
    matches = sorted(
        [r for r in records if r.get("run_stem", "").startswith(family_prefix)],
        key=lambda r: r.get("timestamp", ""),
    )
    if not matches:
        return f"No runs match prefix `{family_prefix}`.\n"
    out = [f"### {family_prefix}\n"]
    for r in matches:
        out.append(render_run(r))
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    recs = load_records(sys.argv[1] if len(sys.argv) > 1 else "runs_tb_extracted.json")
    print(f"Loaded {len(recs)} records")
    by_cat: dict[tuple[str, str], int] = {}
    for r in recs:
        by_cat[(r.get("category", "?"), r.get("mode", "?"))] = by_cat.get((r.get("category", "?"), r.get("mode", "?")), 0) + 1
    for k in sorted(by_cat):
        print(f"  {k[0]}/{k[1]}: {by_cat[k]}")
