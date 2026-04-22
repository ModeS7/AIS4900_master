"""Extract structured data from every TensorBoard run under runs_tb/.

Walks runs_tb/<category>/<mode>/<run_name>/, finds every events.out.tfevents.*
file (may be nested — e.g. nnU-Net uses fold_0/tensorboard/), groups events
files by their containing directory, and uses EventAccumulator to merge
multi-file sequences into a single series per scalar tag.

Output: runs_tb_extracted.json — one record per run with all scalar summaries,
image/text tag names, wall-time span, and parsed metadata (category, mode,
timestamp). Downstream narrative generation reads this JSON only.

Usage:
    python scripts/extract_tb.py                 # dump to runs_tb_extracted.json
    python scripts/extract_tb.py --limit 5       # test mode: first 5 runs
    python scripts/extract_tb.py --only bravo    # filter to runs whose name
                                                   # contains 'bravo'
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TS_RE = re.compile(r"_(\d{8})-(\d{6})$")
MAX_POINTS_PER_SCALAR = 20000  # cap to keep JSON reasonable

logger = logging.getLogger("extract_tb")


def find_events_dirs(run_dir: Path) -> list[Path]:
    """Return directories containing events.out.tfevents.* files, deduped.

    A single run may have multiple events files in the same dir (from restarts);
    EventAccumulator handles that when given the dir. A run may also have events
    in multiple nested dirs (unusual — e.g. nnU-Net has fold_N/tensorboard/),
    in which case we return each distinct parent dir.
    """
    seen: set[Path] = set()
    for events_file in run_dir.rglob("events.out.tfevents.*"):
        seen.add(events_file.parent)
    return sorted(seen)


def _summarize_series(series: list[tuple[int, float, float]]) -> dict[str, Any]:
    """Compute first/last/min/max + NaN flag for a scalar series.

    series: list of (step, value, wall_time)
    """
    if not series:
        return {"n": 0}
    steps_vals = [(s, v) for s, v, _ in series]
    nan_count = sum(1 for _, v, _ in series if math.isnan(v) or math.isinf(v))
    finite = [(s, v) for s, v, _ in series if math.isfinite(v)]
    min_sv = min(finite, key=lambda sv: sv[1]) if finite else None
    max_sv = max(finite, key=lambda sv: sv[1]) if finite else None
    t0, tN = series[0][2], series[-1][2]
    return {
        "n": len(series),
        "first": [steps_vals[0][0], float(steps_vals[0][1])],
        "last": [steps_vals[-1][0], float(steps_vals[-1][1])],
        "min": [min_sv[0], float(min_sv[1])] if min_sv else None,
        "max": [max_sv[0], float(max_sv[1])] if max_sv else None,
        "nan_count": nan_count,
        "wall_time_span_s": float(tN - t0) if tN >= t0 else 0.0,
    }


def extract_events_dir(events_dir: Path) -> dict[str, Any]:
    """Parse one events-containing directory and summarize all scalars."""
    ea = EventAccumulator(
        str(events_dir),
        size_guidance={"scalars": MAX_POINTS_PER_SCALAR, "tensors": 50, "images": 0},
    )
    ea.Reload()
    tags = ea.Tags()
    out: dict[str, Any] = {
        "events_dir": str(events_dir),
        "scalar_tags": sorted(tags.get("scalars", [])),
        "image_tags": sorted(tags.get("images", [])),
        "text_tags": sorted(tags.get("tensors", [])),
        "scalars": {},
    }
    for tag in tags.get("scalars", []):
        series = ea.Scalars(tag)
        triples = [(e.step, float(e.value), float(e.wall_time)) for e in series]
        out["scalars"][tag] = _summarize_series(triples)
    return out


def parse_run_metadata(run_dir: Path, root: Path) -> dict[str, Any]:
    """Derive category/mode/run_name/timestamp from path + name."""
    rel = run_dir.relative_to(root)
    parts = rel.parts  # e.g. ('diffusion_3d', 'bravo', 'exp1_1_1000_pixel_bravo_20260402-121556')
    category = parts[0] if len(parts) > 0 else "?"
    mode = parts[1] if len(parts) > 1 else "?"
    name = parts[-1]
    ts_match = TS_RE.search(name)
    if ts_match:
        date_s, time_s = ts_match.groups()
        stem = name[: ts_match.start()]
        timestamp = f"{date_s}-{time_s}"
        epoch_ts = time.mktime(time.strptime(timestamp, "%Y%m%d-%H%M%S"))
    else:
        stem = name
        timestamp = None
        epoch_ts = None
    return {
        "category": category,
        "mode": mode,
        "run_name": name,
        "run_stem": stem,
        "timestamp": timestamp,
        "timestamp_epoch": epoch_ts,
        "relpath": str(rel),
    }


def extract_run(run_dir: Path, root: Path) -> dict[str, Any]:
    """One record per run. Merges multi-events-dir runs into one structure."""
    meta = parse_run_metadata(run_dir, root)
    events_dirs = find_events_dirs(run_dir)
    if not events_dirs:
        return {**meta, "error": "no events files found", "scalar_count": 0}

    # One events dir is the common case. Multi-dir happens only for nnU-Net.
    sub_records = []
    for ed in events_dirs:
        try:
            sub_records.append(extract_events_dir(ed))
        except Exception as e:  # noqa: BLE001
            logger.exception("failed to read %s", ed)
            sub_records.append({"events_dir": str(ed), "error": str(e)})

    # Flatten: if multiple events dirs, merge their scalar tags into a single
    # dict keyed by "<subdir>/<tag>" so nothing collides.
    if len(sub_records) == 1:
        primary = sub_records[0]
        return {
            **meta,
            "events_dirs": [primary["events_dir"]],
            "scalar_tags": primary.get("scalar_tags", []),
            "image_tags": primary.get("image_tags", []),
            "text_tags": primary.get("text_tags", []),
            "scalars": primary.get("scalars", {}),
            "scalar_count": len(primary.get("scalar_tags", [])),
            "errors": [primary["error"]] if "error" in primary else [],
        }
    # Multi-dir: prefix tags by events-dir stem
    merged_scalars: dict[str, Any] = {}
    merged_tags: list[str] = []
    errors: list[str] = []
    for sr in sub_records:
        if "error" in sr:
            errors.append(f"{sr['events_dir']}: {sr['error']}")
            continue
        prefix = Path(sr["events_dir"]).relative_to(run_dir).as_posix()
        for t in sr.get("scalar_tags", []):
            key = f"{prefix}::{t}"
            merged_tags.append(key)
            merged_scalars[key] = sr["scalars"][t]
    return {
        **meta,
        "events_dirs": [sr["events_dir"] for sr in sub_records],
        "scalar_tags": sorted(merged_tags),
        "image_tags": sorted({t for sr in sub_records for t in sr.get("image_tags", [])}),
        "text_tags": sorted({t for sr in sub_records for t in sr.get("text_tags", [])}),
        "scalars": merged_scalars,
        "scalar_count": len(merged_tags),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs_tb", help="TB runs root")
    parser.add_argument("--out", default="runs_tb_extracted.json")
    parser.add_argument("--limit", type=int, default=0, help="process only first N runs")
    parser.add_argument("--only", default=None, help="only runs whose path contains this substring")
    parser.add_argument("--log", default="runs_tb_extraction.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log, mode="w"), logging.StreamHandler(sys.stdout)],
    )

    root = Path(args.root)
    if not root.exists():
        logger.error("root dir not found: %s", root)
        return 1

    # Run dirs are at depth-3: root/category/mode/run_name/
    all_runs = [p for p in sorted(root.glob("*/*/*")) if p.is_dir()]
    if args.only:
        all_runs = [p for p in all_runs if args.only in str(p)]
    if args.limit:
        all_runs = all_runs[: args.limit]
    logger.info("found %d runs under %s", len(all_runs), root)

    records: list[dict[str, Any]] = []
    # Count stats for summary
    cat_counts: dict[str, int] = defaultdict(int)
    total_scalars = 0
    nan_runs = 0
    empty_runs = 0
    started = time.time()
    for i, run in enumerate(all_runs, 1):
        rec = extract_run(run, root)
        records.append(rec)
        sc = rec.get("scalar_count", 0)
        total_scalars += sc
        cat_counts[rec.get("category", "?")] += 1
        if sc == 0:
            empty_runs += 1
        else:
            any_nan = any(
                s.get("nan_count", 0) > 0
                for s in rec.get("scalars", {}).values()
                if isinstance(s, dict)
            )
            if any_nan:
                nan_runs += 1
        logger.info(
            "[%d/%d] %s — %d scalars",
            i,
            len(all_runs),
            rec.get("relpath", "?"),
            sc,
        )

    elapsed = time.time() - started
    logger.info("=" * 60)
    logger.info("done in %.1fs — %d runs, %d total scalars", elapsed, len(records), total_scalars)
    logger.info("empty runs (no events): %d", empty_runs)
    logger.info("runs with NaN: %d", nan_runs)
    for cat, n in sorted(cat_counts.items()):
        logger.info("  %s: %d runs", cat, n)

    Path(args.out).write_text(json.dumps(records, indent=2, default=str))
    logger.info("wrote %s (%d bytes)", args.out, Path(args.out).stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
