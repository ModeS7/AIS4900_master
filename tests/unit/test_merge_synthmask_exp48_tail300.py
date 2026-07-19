"""Tests for fixed-boundary exp48 head/tail dataset assembly."""

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
MERGE_SCRIPT = PROJECT_ROOT / "IDUN/generate/merge_synthmask_exp48_tail300.sh"
LABEL = "exp1_to_exp48a_t025"
TAIL_LABEL = f"{LABEL}__tail_from_00300"
HEADER = "id,bin_0,bin_1,bin_2,bin_3,bin_4,bin_5,bin_6,total_tumors"


def _write_case(dataset: Path, candidate_id: str, *, seg: bool = True, bravo: bool = True) -> None:
    sample = dataset / candidate_id
    sample.mkdir(parents=True)
    if seg:
        (sample / "seg.nii.gz").write_text(f"seg-{dataset.name}-{candidate_id}\n")
    if bravo:
        (sample / "bravo.nii.gz").write_text(f"bravo-{dataset.name}-{candidate_id}\n")


def _prepare_layout(tmp_path: Path, *, progress: int = 300) -> tuple[Path, Path, Path, dict[str, str]]:
    cluster_base = tmp_path / "cluster"
    eval_id = "test_eval"
    eval_root = cluster_base / "MedicalDataSets/evalModels" / eval_id
    head = eval_root / LABEL
    tail = eval_root / TAIL_LABEL
    head.mkdir(parents=True)
    tail.mkdir(parents=True)

    head_log = tmp_path / "head.out"
    head_log.write_text(f"[test][INFO] - Progress: {progress}/525\n")
    (tail / ".candidate_range_complete").write_text(
        "\n".join(
            [
                f"label={LABEL}",
                f"output_label={TAIL_LABEL}",
                "start_candidate=300",
                "stop_candidate=525",
                "processed_candidates=225",
                "accepted_candidates=0",
                "",
            ]
        )
    )
    (tail / "bins.csv").write_text(f"{HEADER}\n")

    env = os.environ.copy()
    env.update(
        {
            "CLUSTER_BASE": str(cluster_base),
            "EVAL_ID": eval_id,
            "MERGE_TAG": "pytest",
            "SYNTHMASK_HEAD_STOP_CONFIRMED": "true",
        }
    )
    return head, tail, head_log, env


def _run_merge(head_log: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(MERGE_SCRIPT), LABEL, str(head_log)],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def _set_tail_accepted(tail: Path, count: int) -> None:
    marker = tail / ".candidate_range_complete"
    marker.write_text(marker.read_text().replace("accepted_candidates=0", f"accepted_candidates={count}"))


def test_merge_uses_head_below_300_and_tail_from_300(tmp_path: Path) -> None:
    head, tail, head_log, env = _prepare_layout(tmp_path)
    _write_case(head, "00000")
    _write_case(head, "00299")
    # A cancellation may leave partial overlap. The tail owns this ID, so the
    # merger must ignore the head copy rather than treating it as canonical.
    _write_case(head, "00300", bravo=False)
    _write_case(tail, "00300")
    _write_case(tail, "00524")
    (tail / "bins.csv").write_text(
        f"{HEADER}\n"
        "00300,0,0,0,0,0,0,0,0\n"
        "00524,0,0,0,0,0,0,0,0\n"
    )
    _set_tail_accepted(tail, 2)

    result = _run_merge(head_log, env)

    assert result.returncode == 0, result.stderr
    assert {path.name for path in head.iterdir() if path.is_dir()} == {
        "00000",
        "00299",
        "00300",
        "00524",
    }
    assert (head / "00300/bravo.nii.gz").read_text().startswith(f"bravo-{TAIL_LABEL}")
    assert (head / "bins.csv").read_text().splitlines() == [
        HEADER,
        "00000,0,0,0,0,0,0,0,0",
        "00299,0,0,0,0,0,0,0,0",
        "00300,0,0,0,0,0,0,0,0",
        "00524,0,0,0,0,0,0,0,0",
    ]
    backup = head.parent / f"{LABEL}.__head_before_tail_00300_pytest"
    assert (backup / "00300/seg.nii.gz").is_file()
    assert not (backup / "00300/bravo.nii.gz").exists()
    assert tail.is_dir()


def test_merge_rejects_incomplete_owned_head_candidate(tmp_path: Path) -> None:
    head, tail, head_log, env = _prepare_layout(tmp_path)
    _write_case(head, "00299", bravo=False)
    _write_case(tail, "00300")
    (tail / "bins.csv").write_text(f"{HEADER}\n00300,0,0,0,0,0,0,0,0\n")
    _set_tail_accepted(tail, 1)

    result = _run_merge(head_log, env)

    assert result.returncode != 0
    assert "owned head candidate 00299 has no complete BRAVO volume" in result.stderr
    assert head.is_dir()
    assert not (head.parent / f"{LABEL}.__head_before_tail_00300_pytest").exists()


def test_merge_requires_log_proof_that_head_reached_split(tmp_path: Path) -> None:
    head, tail, head_log, env = _prepare_layout(tmp_path, progress=290)
    _write_case(head, "00000")
    _write_case(tail, "00300")
    (tail / "bins.csv").write_text(f"{HEADER}\n00300,0,0,0,0,0,0,0,0\n")
    _set_tail_accepted(tail, 1)

    result = _run_merge(head_log, env)

    assert result.returncode != 0
    assert "head log reached only 290/525" in result.stderr
    assert head.is_dir()


def test_merge_rejects_stale_tail_completion_marker(tmp_path: Path) -> None:
    head, tail, head_log, env = _prepare_layout(tmp_path)
    _write_case(head, "00000")
    _write_case(tail, "00300")
    (tail / "bins.csv").write_text(f"{HEADER}\n00300,0,0,0,0,0,0,0,0\n")

    result = _run_merge(head_log, env)

    assert result.returncode != 0
    assert "tail marker accepted-candidate count mismatch" in result.stderr
    assert head.is_dir()
