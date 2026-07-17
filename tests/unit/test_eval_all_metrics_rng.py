"""Deterministic sampling tests for the combined generation-metric panel."""

from unittest.mock import patch

import torch

from medgen.metrics.generation import compute_cmmd
from medgen.scripts.eval_all_metrics import (
    derive_seed,
    reference_independent,
    scoped_numpy_rng,
    two_sample_metrics,
)


def test_derived_streams_are_stable_and_scope_specific():
    """Stable hashing gives repeatable streams without relying on Python hash()."""
    first = derive_seed(42, "two_sample", "dataset-a", "train")
    assert first == derive_seed(42, "two_sample", "dataset-a", "train")
    assert first != derive_seed(42, "two_sample", "dataset-b", "train")
    assert first != derive_seed(42, "two_sample", "dataset-a", "test")
    assert 0 <= first <= (1 << 63) - 1


def test_two_sample_metrics_uses_independent_metric_streams():
    """Every stochastic metric receives a distinct child of the pair seed."""
    feats = {name: torch.zeros(2, 2) for name in ("in", "rin", "clip", "med3d")}
    pair_seed = derive_seed(42, "two_sample", "dataset-a", "train")

    with (
        patch("medgen.scripts.eval_all_metrics.compute_fid", side_effect=[1.0, 2.0]),
        patch(
            "medgen.scripts.eval_all_metrics.compute_kid",
            side_effect=[(0.1, 0.01), (0.2, 0.02)],
        ) as kid,
        patch("medgen.scripts.eval_all_metrics.compute_cmmd", side_effect=[0.3, 0.4]) as cmmd,
    ):
        result = two_sample_metrics(
            feats, feats, torch.device("cpu"), bandwidth=1.0, seed=pair_seed,
        )

    assert result["kid_imagenet"] == 0.1
    assert kid.call_args_list[0].kwargs["seed"] == derive_seed(pair_seed, "kid_imagenet")
    assert kid.call_args_list[1].kwargs["seed"] == derive_seed(pair_seed, "kid_radimagenet")
    assert cmmd.call_args_list[0].kwargs["seed"] == derive_seed(pair_seed, "cmmd")
    assert cmmd.call_args_list[1].kwargs["seed"] == derive_seed(pair_seed, "med3d_mmd")


def test_diversity_subsample_is_independent_of_dataset_order():
    """Per-dataset streams select the same volumes regardless of loop order."""
    vols = torch.arange(8, dtype=torch.float32).reshape(8, 1, 1, 1, 1)

    def selections(order):
        chosen = {}
        for label in order:
            captured = []

            def capture(div_vols, *_args, **_kwargs):
                captured.extend(int(v) for v in div_vols[:, 0, 0, 0, 0])
                return 0.5

            rng = scoped_numpy_rng(42, "diversity", label)
            with patch("medgen.scripts.eval_all_metrics.pairwise_diversity_3d", capture):
                reference_independent(
                    vols, torch.device("cpu"), None, 0.05, diversity_cap=4, rng=rng,
                )
            chosen[label] = captured
        return chosen

    assert selections(["dataset-a", "dataset-b"]) == selections(["dataset-b", "dataset-a"])


def test_stochastic_metric_values_are_independent_of_dataset_order():
    """KID and capped CMMD retain each dataset's values when loop order changes."""
    names = ("in", "rin", "clip", "med3d")
    data_rng = torch.Generator().manual_seed(71)
    reference = {name: torch.randn(120, 4, generator=data_rng) for name in names}
    generated = {
        label: {name: torch.randn(120, 4, generator=data_rng) for name in names}
        for label in ("dataset-a", "dataset-b")
    }

    def capped_cmmd(*args, **kwargs):
        kwargs["max_samples"] = 12
        return compute_cmmd(*args, **kwargs)

    def score(order):
        results = {}
        with (
            patch("medgen.scripts.eval_all_metrics.compute_fid", return_value=0.0),
            patch("medgen.scripts.eval_all_metrics.compute_cmmd", side_effect=capped_cmmd),
        ):
            for label in order:
                _ = torch.rand(13)  # Unrelated global use must not perturb local streams.
                pair_seed = derive_seed(42, "two_sample", label, "train")
                results[label] = two_sample_metrics(
                    generated[label], reference, torch.device("cpu"),
                    bandwidth=1.0, seed=pair_seed,
                )
        return results

    torch.manual_seed(1)
    forward = score(["dataset-a", "dataset-b"])
    torch.manual_seed(999)
    reverse = score(["dataset-b", "dataset-a"])
    assert forward == reverse
