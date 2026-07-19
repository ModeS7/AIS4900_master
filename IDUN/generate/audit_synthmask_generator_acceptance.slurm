#!/usr/bin/env bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=0-00:10:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name="audit_sm_acceptance"
#SBATCH --mail-user=modestas@stud.ntnu.no
#SBATCH --mail-type=END,FAIL
#SBATCH --output=IDUN/output/generate/audit_sm_acceptance_%j.out
#SBATCH --error=IDUN/output/generate/audit_sm_acceptance_%j.err

# Read-only audit of which generator(s) reduce the shared candidate count.

set -Eeuo pipefail

python3 - <<'PY'
from itertools import combinations
from pathlib import Path

root = Path(
    "/cluster/work/modestas/MedicalDataSets/evalModels/"
    "source_selection_synthmask525_seed42_euler100_pca2p323mm_try3"
)

labels = [
    "exp1_1_1000",
    "exp1_1_1000plus",
    "exp32_2_1000",
    "exp32_3_1000",
    "exp47a",
    "exp47b",
    "exp47c",
    "exp47d",
    "exp47e",
    "exp1_to_exp48a_t025",
    "exp1_to_exp48b_t025",
    "exp1_to_exp48c_t025",
    "exp1_to_exp48d_t025",
    "exp1_to_exp48e_t025",
]

core_five = [
    "exp1_1_1000",
    "exp1_1_1000plus",
    "exp32_2_1000",
    "exp1_to_exp48c_t025",
    "exp1_to_exp48d_t025",
]

candidate_ids = {f"{index:05d}" for index in range(525)}
selected_file = root / "matched_common105/common_candidate_ids.txt"

if not selected_file.is_file():
    raise SystemExit(f"FATAL: selected-ID file not found: {selected_file}")

selected_ids = selected_file.read_text().splitlines()
if len(selected_ids) != 105 or len(set(selected_ids)) != 105:
    raise SystemExit(
        f"FATAL: expected 105 unique IDs in {selected_file}, got "
        f"{len(selected_ids)} rows and {len(set(selected_ids))} unique IDs"
    )
if not set(selected_ids) <= candidate_ids:
    raise SystemExit(f"FATAL: invalid candidate ID in {selected_file}")


def is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


accepted = {}
for label in labels:
    dataset = root / label
    if not dataset.is_dir():
        raise SystemExit(f"FATAL: generator dataset not found: {dataset}")
    if not is_nonempty_file(dataset / "bins.csv"):
        raise SystemExit(f"FATAL: missing completion file: {dataset / 'bins.csv'}")

    accepted_ids = set()
    incomplete_ids = []
    for candidate_id in candidate_ids:
        sample = dataset / candidate_id
        has_seg = is_nonempty_file(sample / "seg.nii.gz")
        has_bravo = is_nonempty_file(sample / "bravo.nii.gz")
        if has_seg != has_bravo:
            incomplete_ids.append(candidate_id)
        elif has_seg:
            accepted_ids.add(candidate_id)

    if incomplete_ids:
        raise SystemExit(
            f"FATAL: {label} has incomplete pairs: "
            + " ".join(sorted(incomplete_ids))
        )
    accepted[label] = accepted_ids


def common(kept_labels):
    result = candidate_ids.copy()
    for label in kept_labels:
        result &= accepted[label]
    return result


all_common = common(labels)
core_common = common(core_five)
selected_set = set(selected_ids)

if len(core_common) != 106:
    raise SystemExit(
        f"FATAL: expected the core-five intersection to contain 106 IDs, "
        f"found {len(core_common)}"
    )
expected_selected = sorted(core_common)[:105]
if selected_ids != expected_selected:
    raise SystemExit(
        "FATAL: common_candidate_ids.txt is not the first 105 IDs from the "
        "core-five intersection"
    )

print(f"All 14 common: {len(all_common)}/525")
print(f"Core five common: {len(core_common)}/525")
print()
print(
    f"{'generator':30} {'accepted':>8} {'of core':>8} "
    f"{'of 105':>7} {'without it':>10} {'gain':>6}"
)

leave_one_out = {}
for label in labels:
    without_label = common([other for other in labels if other != label])
    leave_one_out[label] = len(without_label)
    print(
        f"{label:30} {len(accepted[label]):8d} "
        f"{len(accepted[label] & core_common):8d} "
        f"{len(accepted[label] & selected_set):7d} "
        f"{len(without_label):10d} "
        f"{len(without_label) - len(all_common):+6d}"
    )

print()
print("Missing IDs from the frozen 105:")
for label in labels:
    missing = sorted(selected_set - accepted[label])
    if missing:
        print(f"{label:30} {len(missing):2d}: {' '.join(missing)}")

print()
print("Core-five candidates blocked by the other nine generators:")
non_core = [label for label in labels if label not in core_five]
blocked_count = 0
for candidate_id in sorted(core_common):
    blockers = [
        label for label in non_core if candidate_id not in accepted[label]
    ]
    if blockers:
        blocked_count += 1
        print(f"{candidate_id}: {', '.join(blockers)}")
print(f"Blocked core-five candidates: {blocked_count}/{len(core_common)}")

single_removals = [
    (label, count)
    for label, count in leave_one_out.items()
    if label not in core_five and count >= 105
]

print()
if single_removals:
    print("A single non-core generator can be removed while retaining >=105 common cases:")
    for label, count in sorted(single_removals, key=lambda item: (-item[1], item[0])):
        print(f"  remove {label}: {count}/525 remain common")
else:
    print("No single non-core generator removal reaches 105 common cases.")

removable = non_core
for removal_count in range(len(removable) + 1):
    solutions = []
    for removed in combinations(removable, removal_count):
        kept = [label for label in labels if label not in removed]
        count = len(common(kept))
        if count >= 105:
            solutions.append((count, removed))
    if solutions:
        solutions.sort(key=lambda item: (-item[0], item[1]))
        best_count, best_removed = solutions[0]
        print(
            "Minimum non-core removals needed for >=105 common cases: "
            f"{removal_count}"
        )
        print(
            f"Best retained intersection: {best_count}/525; remove: "
            + (", ".join(best_removed) if best_removed else "none")
        )
        if len(solutions) > 1:
            print(f"Other feasible removal sets at this size: {len(solutions) - 1}")
        break
PY
