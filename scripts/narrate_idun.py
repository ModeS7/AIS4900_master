"""Helpers to render per-job markdown narratives from idun_logs_extracted.json."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("/home/mode/NTNU/AIS4900_master")


def load() -> list[dict]:
    return json.loads((ROOT / "idun_logs_extracted.json").read_text())


def jobs_in(records: list[dict], subdir_prefix: str) -> list[dict]:
    return [r for r in records if r.get("subdir", "").startswith(subdir_prefix)]


def script_key(tag: str) -> str:
    """Extract the script-name root from a job tag.

    Examples:
      "find_optimal_steps_exp1_1_1000" -> "find_optimal_steps"
      "diagnose_mean_blur_exp37_3"     -> "diagnose_mean_blur"
      "eval_time_shift_25319857"       -> "eval_time_shift"
      "phema_sweep_exp1"               -> "phema_sweep"
    """
    parts = tag.split("_")
    keep: list[str] = []
    for p in parts:
        if re.match(r"^(exp|EMA|test|val|train|wdm|ldm)\d", p, re.I):
            break
        if re.match(r"^\d+$", p) and keep:
            break
        keep.append(p)
    return "_".join(keep) if keep else tag


def family_key(tag: str) -> str:
    """Extract experiment family (expN) from a job tag if present."""
    m = re.search(r"\bexp_?(\d+)", tag)
    return f"exp{m.group(1)}" if m else "OTHER"


def fmt_status(status: str) -> str:
    icon = {
        "completed": "✅",
        "oom_killed": "💥",
        "crashed": "❌",
        "truncated": "⚠️",
        "chained": "🔗",
        "empty": "∅",
        "killed": "💀",
        "extractor_error": "🛠",
    }.get(status, "?")
    return f"{icon} {status}"


def fmt_meta(rec: dict) -> str:
    bits = []
    if rec.get("job_id"):
        bits.append(f"job {rec['job_id']}")
    if rec.get("node"):
        bits.append(rec["node"])
    if rec.get("gpu"):
        bits.append(rec["gpu"])
    if rec.get("real_time"):
        bits.append(rec["real_time"].replace("m", "m ").rstrip())
    if rec.get("training_hours"):
        bits.append(f"{rec['training_hours']}h training")
    if rec.get("max_epoch") is not None and rec.get("target_epochs"):
        bits.append(f"epoch {rec['max_epoch']}/{rec['target_epochs']}")
    if rec.get("chain_segment"):
        bits.append(f"chain {rec['chain_segment'][0]}/{rec['chain_segment'][1]}")
    if rec.get("size_kb", 0) > 0:
        bits.append(f"{rec['size_kb']}KB log")
    return " • ".join(bits)


def render_test_block(rec: dict) -> list[str]:
    """Render the final test metrics block(s)."""
    out: list[str] = []
    tr = rec.get("final_test_results") or []
    fkc = rec.get("final_fid_kid_cmmd") or []
    if not tr and not fkc:
        return out
    out.append("**Final test metrics:**")
    # Pair test_results with FID/KID/CMMD by index (best then latest, normally)
    for i, t in enumerate(tr):
        f = fkc[i] if i < len(fkc) else None
        line = (f"  - **{t['variant']}** ckpt ({t['n_samples']} samples): "
                f"MSE {t['mse']:.4g} • MS-SSIM {t['ms_ssim']:.4f} • "
                f"PSNR {t['psnr']:.2f} dB • LPIPS {t['lpips']:.4f}")
        if f:
            line += (f" • FID {f['fid']:.2f} • KID {f['kid_mean']:.4f} ± {f['kid_std']:.4f}"
                     f" • CMMD {f['cmmd']:.4f}")
        out.append(line)
    return out


def render_generation_block(rec: dict) -> list[str]:
    """Render the multi-model generation summary block."""
    out: list[str] = []
    gs = rec.get("generation_summary") or []
    ss = rec.get("saved_samples") or []
    if not gs and not ss:
        return out
    if gs:
        out.append("**Generation summary (model: completed/total):**")
        for g in gs:
            out.append(f"  - `{g['model']}`: {g['completed']}/{g['total']} → `{g['out_dir']}`")
    elif ss:
        out.append("**Saved samples:**")
        for s in ss:
            out.append(f"  - {s['count']} samples → `{s['out_dir']}`")
    return out


def render_results_saved(rec: dict) -> list[str]:
    out: list[str] = []
    rs = rec.get("results_saved_to") or []
    if not rs:
        return out
    if len(rs) == 1:
        out.append(f"**Results dir:** `{rs[0]}`")
    else:
        out.append(f"**Results dirs ({len(rs)}):**")
        for r in rs:
            out.append(f"  - `{r}`")
    return out


def render_json_results(rec: dict) -> list[str]:
    """Render JSON-search results from find_optimal_steps / eval_steps_pca / etc."""
    out: list[str] = []
    j = rec.get("json_results") or {}
    if not j:
        return out

    bits = []
    if "best_steps" in j: bits.append(f"**best_steps={j['best_steps']}**")
    if "best_t0" in j: bits.append(f"**best_t0={j['best_t0']}**")
    if "best_cfg" in j: bits.append(f"**best_cfg={j['best_cfg']}**")
    if "best_freeu" in j: bits.append(f"**best_freeu={j['best_freeu']}**")
    if "metric" in j: bits.append(f"metric=`{j['metric']}`")
    if "search_range" in j: bits.append(f"range={j['search_range']}")
    if "num_volumes" in j: bits.append(f"{j['num_volumes']} volumes")
    if "n_evaluations" in j: bits.append(f"{j['n_evaluations']} evals")
    if "pca_pass_rate" in j: bits.append(f"PCA pass {j['pca_pass_rate']*100:.1f}%")
    if "pca_shape_error" in j: bits.append(f"PCA err {j['pca_shape_error']:.4f}")

    if bits:
        out.append("**Search results:** " + " • ".join(bits))

    be = j.get("best_eval")
    if be:
        line = f"**Best eval (steps={be['steps']}):** FID {be['fid']:.3f}"
        line += f" • KID {be['kid_mean']:.4f} ± {be['kid_std']:.4f}"
        line += f" • CMMD {be['cmmd']:.4f}"
        if "fid_radimagenet" in be:
            line += f" • FID_radimagenet {be['fid_radimagenet']:.4f}"
        out.append(line)
    elif "min_fid" in j:
        out.append(f"**FID range across evals:** {j['min_fid']:.3f} – {j['max_fid']:.3f}")

    return out


def render_best_metrics(rec: dict) -> list[str]:
    out: list[str] = []
    bm = rec.get("best_metrics") or []
    if not bm:
        return out
    out.append("**Best metrics:** " + " • ".join(f"`{n}`={v:.4g}" for n, v in bm))
    return out


def render_traceback(rec: dict, max_lines: int = 8) -> list[str]:
    out: list[str] = []
    tb = rec.get("traceback_excerpt", "")
    if not tb:
        # fallback: use err_tail_lines
        et = rec.get("err_tail_lines") or []
        if et:
            out.append("**Stderr tail:**")
            out.append("```")
            out.extend(et[:max_lines])
            out.append("```")
        return out
    lines = tb.splitlines()
    out.append("**Traceback excerpt:**")
    out.append("```")
    # Keep first 2 lines + last few of traceback for the actual error line
    if len(lines) <= max_lines:
        out.extend(lines)
    else:
        out.extend(lines[:3])
        out.append("...")
        out.extend(lines[-max_lines + 3 :])
    out.append("```")
    return out


def render_resume(rec: dict) -> list[str]:
    out: list[str] = []
    rf = rec.get("resume_from") or []
    if rf:
        out.append(f"**Resumed from:** `{rf[-1]}`" + (f" (+{len(rf) - 1} earlier)" if len(rf) > 1 else ""))
    return out


def render_job(rec: dict, *, brief: bool = False) -> str:
    """Render a single job entry. `brief=True` for one-liner format used in
    bulk-summarized subdirs."""
    name = f"{rec['job_tag']}_{rec['job_id']}" if rec.get("job_id") else rec["job_tag"]
    status = fmt_status(rec.get("status", "?"))
    if brief:
        return f"- {status} `{name}` — {fmt_meta(rec)}"

    out: list[str] = []
    out.append(f"#### `{name}`  [{status}]")
    out.append(f"*{fmt_meta(rec)}*")
    out.append("")
    out.extend(render_test_block(rec))
    out.extend(render_generation_block(rec))
    out.extend(render_json_results(rec))
    out.extend(render_results_saved(rec))
    out.extend(render_best_metrics(rec))
    out.extend(render_resume(rec))
    if rec.get("trainer_best_loss") is not None:
        out.append(f"**Trainer best loss:** {rec['trainer_best_loss']:.6f}")
    if rec.get("status") in {"crashed", "oom_killed"}:
        out.extend(render_traceback(rec))
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    recs = load()
    if len(sys.argv) > 1:
        prefix = sys.argv[1]
        for r in jobs_in(recs, prefix)[:5]:
            print(render_job(r))
            print("---")
    else:
        print(f"Loaded {len(recs)} records")
