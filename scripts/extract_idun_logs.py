"""Extract structured data from every SLURM .out (and matching .err) under IDUN/output/.

For each .out file we record:
  - Job ID, node, GPU, wall-time
  - Status (completed / oom_killed / crashed / truncated / running / empty)
  - End-of-training test metrics (best + latest variants): MSE / MS-SSIM / PSNR / LPIPS / FID / KID / CMMD
  - Eval-script-specific result lines (Best t0:, Best step:, Pass rate:, etc.)
  - "Results saved to:" output directories
  - Generation-pipeline summaries ("model: N/M volumes")
  - Resume / chain segment hints
  - Head/tail (40 lines each) for narrative context
  - Stderr tail when the run crashed/OOM'd

Output: idun_logs_extracted.json — one record per .out file.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("extract_idun")

ROOT_DEFAULT = Path("IDUN/output")

JOBID_FILENAME_RE = re.compile(r"_(\d+)\.out$")
JOB_HEADER_JOBID_RE = re.compile(r"Job(?:\s*ID)?\s*[: ]\s*(\d+)")
JOB_HEADER_NODE_RE = re.compile(r"Node\s*[: ]\s*(\S+)")
GPU_MODEL_RE = re.compile(r"NVIDIA\s+(A100|H100|H200|V100|L40|RTX\s+\w+)[^|]*?(80GB|40GB|HBM3|SXM4|PCIe)?", re.I)
REAL_TIME_RE = re.compile(r"^real\s+(\d+m[\d.]+s)", re.M)
KILLED_RE = re.compile(r"\bKilled\b|oom_kill|\bSIGKILL\b|slurmstepd:\s+error")
OOM_RE = re.compile(r"CUDA out of memory|oom_kill|RuntimeError:.*CUDA|out_of_memory")
TRACEBACK_RE = re.compile(r"^Traceback \(most recent call last\):", re.M)
TRAINING_DONE_RE = re.compile(r"Training completed in ([\d.]+)\s*hours?")
GEN_COMPLETE_RE = re.compile(r"Generation complete!|All comparisons complete")
TEST_RESULTS_BLOCK_RE = re.compile(
    r"Test Results - (best|latest) \((\d+) samples?\):\s*\n"
    r".*?MSE:\s+([\d.eE+\-]+)\s*\n"
    r".*?MS-SSIM:\s+([\d.eE+\-]+)\s*\n"
    r".*?PSNR:\s+([\d.eE+\-]+)\s*dB\s*\n"
    r".*?LPIPS:\s+([\d.eE+\-]+)",
    re.S,
)
FID_KID_CMMD_RE = re.compile(
    r"FID:\s+([\d.eE+\-]+)\s*\n"
    r".*?KID:\s+([\d.eE+\-]+)\s*\+/\-\s*([\d.eE+\-]+)\s*\n"
    r".*?CMMD:\s+([\d.eE+\-]+)",
    re.S,
)
RESULTS_SAVED_RE = re.compile(r"Results saved to:\s+(\S+)")
GENERATION_LINE_RE = re.compile(r"^\s*([\w_]+):\s+(\d+)/(\d+)\s+volumes\s+->\s+(\S+)", re.M)
BEST_METRIC_RE = re.compile(r"Best\s+([a-zA-Z_0-9]+):\s+([\d.eE+\-]+)")
RESUME_FROM_RE = re.compile(r"Resuming from checkpoint:\s+(\S+)")
CHAIN_SEGMENT_RE = re.compile(r"Chain:\s+(\d+)/(\d+)")
EPOCH_PROGRESS_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
SAVED_SAMPLES_RE = re.compile(r"Saved\s+(\d+)\s+samples?\s+to\s+(\S+)")
TRAINER_BEST_LOSS_RE = re.compile(r"Training complete\.\s*Best loss:\s*([\d.eE+\-]+)")

# JSON-result patterns (find_optimal_steps, eval_steps_pca, find_optimal_cfg, etc.)
JSON_BEST_STEPS_RE = re.compile(r'"best_steps":\s*(\d+)')
JSON_METRIC_RE = re.compile(r'"metric":\s*"([^"]+)"')
JSON_BEST_T0_RE = re.compile(r'"best_t0":\s*([\d.eE+\-]+)')
JSON_BEST_CFG_RE = re.compile(r'"best_cfg":\s*([\d.eE+\-]+)')
JSON_BEST_FREEU_RE = re.compile(r'"best_freeu":\s*\[([\d.eE+\-,\s]+)\]')
JSON_NUM_VOLUMES_RE = re.compile(r'"num_volumes":\s*(\d+)')
JSON_SEARCH_RANGE_RE = re.compile(r'"search_range":\s*\[\s*([\d.eE+\-]+)\s*,\s*([\d.eE+\-]+)\s*\]')
# Match a single evaluations entry block
JSON_EVAL_ENTRY_RE = re.compile(
    r'\{\s*"steps":\s*(\d+),\s*"fid":\s*([\d.eE+\-]+),'
    r'\s*"kid_mean":\s*([\d.eE+\-]+),\s*"kid_std":\s*([\d.eE+\-]+),'
    r'\s*"cmmd":\s*([\d.eE+\-]+)'
    r'(?:,\s*"fid_radimagenet":\s*([\d.eE+\-]+))?',
    re.S,
)
# PCA pass-rate / shape-error from eval_steps_pca and similar
JSON_PCA_PASS_RE = re.compile(r'"pass_rate":\s*([\d.eE+\-]+)')
JSON_PCA_ERROR_RE = re.compile(r'"shape_error_mean":\s*([\d.eE+\-]+)')

# Generation: "Saved 525 samples to ..." (find_optimal_steps emits "Generation took ...")
GENERATION_TOOK_RE = re.compile(r"Generation took\s+([\d.]+)\s*s")


def parse_job_tag(filename: str) -> tuple[str, str | None]:
    """Split filename `<tag>_<jobid>.out` into (tag, jobid).

    Falls back to (filename_stem, None) if the tag pattern isn't present
    (e.g. tensorboard server logs that are named just `<jobid>.out`).
    """
    m = JOBID_FILENAME_RE.search(filename)
    if not m:
        return Path(filename).stem, None
    jobid = m.group(1)
    tag = filename[: -len(f"_{jobid}.out")]
    if not tag:
        # filename is just <jobid>.out with no descriptive tag
        return Path(filename).stem, jobid
    return tag, jobid


def safe_read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception as e:
        logger.warning("read failed %s: %s", path, e)
        return ""


def determine_status(text: str, err_text: str, has_training_done: bool, has_results_saved: bool) -> str:
    """Infer SLURM status from log content."""
    if not text and not err_text:
        return "empty"
    if OOM_RE.search(text) or OOM_RE.search(err_text) or KILLED_RE.search(err_text):
        return "oom_killed"
    if TRACEBACK_RE.search(text) or TRACEBACK_RE.search(err_text):
        return "crashed"
    if has_training_done or has_results_saved or GEN_COMPLETE_RE.search(text):
        return "completed"
    if KILLED_RE.search(text):
        return "killed"
    # Heuristic: if log ends with a "Wall time approaching" / chain resubmit, mark completed-chain
    if re.search(r"Resubmitted chain segment|Wall time approaching", text):
        return "chained"
    return "truncated"  # ran but didn't finish gracefully (most likely)


def extract_log(out_path: Path, root: Path) -> dict[str, Any]:
    text = safe_read(out_path)
    err_path = out_path.with_suffix(".err")
    err_text = safe_read(err_path) if err_path.exists() else ""

    job_tag, jobid_from_name = parse_job_tag(out_path.name)
    rel = out_path.relative_to(root)
    subdir = str(rel.parent)

    rec: dict[str, Any] = {
        "out_path": str(out_path),
        "err_path": str(err_path) if err_path.exists() else None,
        "subdir": subdir,
        "job_tag": job_tag,
        "job_id": jobid_from_name,
        "size_kb": out_path.stat().st_size // 1024 if out_path.exists() else 0,
        "err_size_kb": err_path.stat().st_size // 1024 if err_path.exists() else 0,
    }

    if not text:
        rec["status"] = "empty"
        rec["head_lines"] = []
        rec["tail_lines"] = []
        rec["err_tail_lines"] = err_text.splitlines()[-20:] if err_text else []
        return rec

    # Override job_id from header when filename didn't have one
    if rec["job_id"] is None:
        m = JOB_HEADER_JOBID_RE.search(text)
        if m:
            rec["job_id"] = m.group(1)

    # Node + GPU
    m = JOB_HEADER_NODE_RE.search(text)
    rec["node"] = m.group(1) if m else None
    m = GPU_MODEL_RE.search(text)
    if m:
        gpu_parts = [g for g in m.groups() if g]
        rec["gpu"] = " ".join(gpu_parts).strip()
    else:
        rec["gpu"] = None

    # Wall-time
    m = REAL_TIME_RE.search(text)
    rec["real_time"] = m.group(1) if m else None
    m = TRAINING_DONE_RE.search(text)
    rec["training_hours"] = float(m.group(1)) if m else None

    # Resume + chain
    rec["resume_from"] = RESUME_FROM_RE.findall(text)
    m = CHAIN_SEGMENT_RE.search(text)
    rec["chain_segment"] = (int(m.group(1)), int(m.group(2))) if m else None

    # Last-seen epoch
    eps = [(int(a), int(b)) for a, b in EPOCH_PROGRESS_RE.findall(text)]
    rec["max_epoch"] = max(eps, key=lambda x: x[0])[0] if eps else None
    rec["target_epochs"] = eps[0][1] if eps else None

    # Final test metrics — capture every (best/latest) block
    test_blocks = []
    for m in TEST_RESULTS_BLOCK_RE.finditer(text):
        test_blocks.append({
            "variant": m.group(1),
            "n_samples": int(m.group(2)),
            "mse": float(m.group(3)),
            "ms_ssim": float(m.group(4)),
            "psnr": float(m.group(5)),
            "lpips": float(m.group(6)),
        })
    # FID/KID/CMMD blocks (paired with test results when emitted nearby)
    fid_blocks = []
    for m in FID_KID_CMMD_RE.finditer(text):
        fid_blocks.append({
            "fid": float(m.group(1)),
            "kid_mean": float(m.group(2)),
            "kid_std": float(m.group(3)),
            "cmmd": float(m.group(4)),
        })
    rec["final_test_results"] = test_blocks
    rec["final_fid_kid_cmmd"] = fid_blocks

    # Eval-script "Best X: Y" lines
    rec["best_metrics"] = [(name, float(val)) for name, val in BEST_METRIC_RE.findall(text)]
    rec["results_saved_to"] = RESULTS_SAVED_RE.findall(text)

    # Generation pipeline summaries
    rec["generation_summary"] = [
        {"model": m[0], "completed": int(m[1]), "total": int(m[2]), "out_dir": m[3]}
        for m in GENERATION_LINE_RE.findall(text)
    ]

    # Saved-samples lines (e.g., "Saved 10 samples to /cluster/.../exp37_3")
    rec["saved_samples"] = [
        {"count": int(c), "out_dir": d} for c, d in SAVED_SAMPLES_RE.findall(text)
    ]

    # Trainer's "Best loss" final line
    m = TRAINER_BEST_LOSS_RE.search(text)
    rec["trainer_best_loss"] = float(m.group(1)) if m else None

    # Extract JSON-search-result fields when present (find_optimal_steps, eval_steps_pca,
    # find_optimal_cfg, calibrate_degradation, etc.)
    json_results: dict[str, Any] = {}
    m = JSON_METRIC_RE.search(text)
    if m: json_results["metric"] = m.group(1)
    m = JSON_BEST_STEPS_RE.search(text)
    if m: json_results["best_steps"] = int(m.group(1))
    m = JSON_BEST_T0_RE.search(text)
    if m: json_results["best_t0"] = float(m.group(1))
    m = JSON_BEST_CFG_RE.search(text)
    if m: json_results["best_cfg"] = float(m.group(1))
    m = JSON_BEST_FREEU_RE.search(text)
    if m: json_results["best_freeu"] = [float(x) for x in m.group(1).split(",")]
    m = JSON_NUM_VOLUMES_RE.search(text)
    if m: json_results["num_volumes"] = int(m.group(1))
    m = JSON_SEARCH_RANGE_RE.search(text)
    if m: json_results["search_range"] = [float(m.group(1)), float(m.group(2))]
    m = JSON_PCA_PASS_RE.search(text)
    if m: json_results["pca_pass_rate"] = float(m.group(1))
    m = JSON_PCA_ERROR_RE.search(text)
    if m: json_results["pca_shape_error"] = float(m.group(1))
    # Find FID/KID/CMMD at best_steps (or min FID across evaluations)
    evals = JSON_EVAL_ENTRY_RE.findall(text)
    if evals:
        # Each match: (steps, fid, kid_mean, kid_std, cmmd, fid_radim_or_empty)
        parsed = []
        for e in evals:
            entry = {
                "steps": int(e[0]),
                "fid": float(e[1]),
                "kid_mean": float(e[2]),
                "kid_std": float(e[3]),
                "cmmd": float(e[4]),
            }
            if e[5]: entry["fid_radimagenet"] = float(e[5])
            parsed.append(entry)
        json_results["n_evaluations"] = len(parsed)
        # Best by metric used in JSON (prefer fid_radimagenet if present)
        if "best_steps" in json_results:
            best = next((e for e in parsed if e["steps"] == json_results["best_steps"]), None)
            if best:
                json_results["best_eval"] = best
        else:
            metric = json_results.get("metric", "fid")
            key = "fid_radimagenet" if metric == "fid_radimagenet" and any("fid_radimagenet" in p for p in parsed) else "fid"
            best = min(parsed, key=lambda p: p.get(key, 1e9))
            json_results["best_eval"] = best
        json_results["min_fid"] = min(p["fid"] for p in parsed)
        json_results["max_fid"] = max(p["fid"] for p in parsed)
    rec["json_results"] = json_results

    # Status determination
    has_training_done = rec["training_hours"] is not None
    has_results_saved = (bool(rec["results_saved_to"]) or bool(rec["saved_samples"])
                        or bool(json_results) or bool(rec["final_test_results"]))
    rec["status"] = determine_status(text, err_text, has_training_done, has_results_saved)

    # Capture tracebacks for crashed/OOM jobs
    if rec["status"] in {"crashed", "oom_killed"}:
        # Extract last traceback block
        tb_match = list(TRACEBACK_RE.finditer(text))
        if tb_match:
            start = tb_match[-1].start()
            tb_excerpt = text[start : start + 2000]  # cap at 2 KB
            rec["traceback_excerpt"] = tb_excerpt
        elif TRACEBACK_RE.search(err_text):
            tb_match = list(TRACEBACK_RE.finditer(err_text))
            start = tb_match[-1].start()
            rec["traceback_excerpt"] = err_text[start : start + 2000]
        else:
            # OOM might not have a Python traceback (SIGKILL); use err tail
            rec["traceback_excerpt"] = err_text[-2000:] if err_text else ""

    # Head/tail snippets for narrative context
    lines = text.splitlines()
    rec["head_lines"] = lines[:40]
    rec["tail_lines"] = lines[-40:] if len(lines) > 40 else lines
    err_lines = err_text.splitlines()
    rec["err_tail_lines"] = err_lines[-20:] if err_text else []

    return rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT_DEFAULT))
    p.add_argument("--out", default="idun_logs_extracted.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", default=None, help="substring filter on .out path")
    p.add_argument("--log", default="idun_logs_extraction.log")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.log, mode="w"), logging.StreamHandler(sys.stdout)],
    )

    root = Path(args.root)
    if not root.exists():
        logger.error("root not found: %s", root)
        return 1

    all_outs = sorted(root.rglob("*.out"))
    if args.only:
        all_outs = [p for p in all_outs if args.only in str(p)]
    if args.limit:
        all_outs = all_outs[: args.limit]
    logger.info("found %d .out files under %s", len(all_outs), root)

    started = time.time()
    records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    subdir_counts: dict[str, int] = {}
    for i, op in enumerate(all_outs, 1):
        try:
            rec = extract_log(op, root)
        except Exception as e:  # noqa: BLE001
            logger.exception("failed to extract %s", op)
            rec = {"out_path": str(op), "subdir": str(op.parent.relative_to(root)),
                   "status": "extractor_error", "error": str(e)}
        records.append(rec)
        status_counts[rec["status"]] = status_counts.get(rec["status"], 0) + 1
        subdir_counts[rec["subdir"]] = subdir_counts.get(rec["subdir"], 0) + 1
        if i % 100 == 0 or i == len(all_outs):
            logger.info("[%d/%d] processed (last: %s) status=%s",
                        i, len(all_outs), op.relative_to(root), rec["status"])

    logger.info("=" * 60)
    logger.info("done in %.1fs — %d files", time.time() - started, len(records))
    logger.info("status distribution:")
    for k, v in sorted(status_counts.items(), key=lambda x: -x[1]):
        logger.info("  %s: %d", k, v)
    logger.info("subdir distribution:")
    for k, v in sorted(subdir_counts.items()):
        logger.info("  %s: %d", k, v)

    Path(args.out).write_text(json.dumps(records, indent=2, default=str))
    logger.info("wrote %s (%d KB)", args.out, Path(args.out).stat().st_size // 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
