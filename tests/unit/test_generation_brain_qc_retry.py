"""Integration tests for fixed-mask BRAVO rejection and deterministic retries."""

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import medgen.scripts.generate as generation


class _StrategyStub:
    """Minimal strategy used to keep the pipeline test independent of CUDA."""

    def setup_scheduler(self, **_kwargs: object) -> None:
        self.scheduler = SimpleNamespace()


def _write_mask(path: Path, foreground_xyz: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros((4, 4, 3), dtype=np.float32)
    mask[foreground_xyz] = 1.0
    path.parent.mkdir(parents=True)
    nib.save(nib.Nifti1Image(mask, np.eye(4, dtype=np.float32)), path)
    return mask


def _pipeline_config(
    tmp_path: Path,
    real_seg_dir: Path,
    *,
    num_images: int,
    max_image_attempts: int,
    generation_depth: int = 3,
    trim_slices: int = 0,
    expected_real_depth: int = 3,
):
    checkpoint = tmp_path / "image.pt"
    torch.save(
        {
            "config": {
                "sigma_data": 0.0,
                "out_channels": 1,
                "pixel": {},
                "offset_noise": {},
            }
        },
        checkpoint,
    )
    config = OmegaConf.load(Path(__file__).parents[2] / "configs" / "generate.yaml")
    return OmegaConf.merge(
        config,
        {
            "spatial_dims": 3,
            "strategy": "rflow",
            "gen_mode": "bravo",
            "image_model": str(checkpoint),
            "image_model_high_t": None,
            "image_size": 4,
            "depth": generation_depth,
            "trim_slices": trim_slices,
            "fov_mm": 4.0,
            "num_images": num_images,
            "num_steps": 2,
            "num_steps_bravo": 2,
            "current_image": 0,
            "seed": 42,
            "real_seg_dir": str(real_seg_dir),
            "expected_real_cases": num_images,
            "expected_real_depth": expected_real_depth,
            "require_real_bravo_pairs": False,
            "validate_real_seg_masks": True,
            "validate_size_bins": False,
            "brain_atlas_path": None,
            "brain_pca_path": None,
            "seg_pca_path": None,
            "validate_brain_mask": True,
            "brain_threshold": 0.05,
            "conditioning_brain_qc_mode": "reject",
            "brain_containment_margin_mm": 0.0,
            "max_image_attempts_per_mask": max_image_attempts,
            "mask_outside_brain": False,
            "verbose": False,
        },
    )


def _patch_non_sampling_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generation, "_create_strategy", lambda _name: _StrategyStub())
    monkeypatch.setattr(
        generation,
        "_load_image_model_with_optional_handoff",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(generation, "_build_diffrs", lambda *_args: (None, None))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def _image_with_tissue(voxel_dhw: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((3, 4, 4), dtype=np.float32)
    image[voxel_dhw] = 1.0
    return image


def test_reject_mode_retries_same_mask_with_deterministic_seed_and_resets_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real_seg_dir = tmp_path / "masks"
    source_masks = [
        _write_mask(real_seg_dir / "Mets_001" / "seg.nii.gz", (1, 1, 1)),
        _write_mask(real_seg_dir / "Mets_002" / "seg.nii.gz", (2, 1, 1)),
    ]
    config = _pipeline_config(
        tmp_path,
        real_seg_dir,
        num_images=2,
        max_image_attempts=2,
    )
    output_dir = tmp_path / "generated"
    _patch_non_sampling_dependencies(monkeypatch)

    # Sample 0 first produces tissue far from its conditioning lesion, then a
    # valid image. Sample 1 succeeds on its first deterministic draw.
    generated_images = iter(
        [
            _image_with_tissue((1, 3, 3)),
            _image_with_tissue((1, 1, 1)),
            _image_with_tissue((1, 2, 1)),
        ]
    )
    masks_seen: list[np.ndarray] = []
    seeds_seen: list[int | None] = []

    def fake_generate_bravo(
        seg_binary: np.ndarray,
        *_args: object,
        noise_seed: int | None = None,
        **_kwargs: object,
    ) -> np.ndarray:
        masks_seen.append(seg_binary.copy())
        seeds_seen.append(noise_seed)
        return next(generated_images)

    monkeypatch.setattr(generation, "_generate_bravo", fake_generate_bravo)

    generation.run_3d_pipeline(config, output_dir)

    expected_dhw = [np.transpose(mask, (2, 0, 1)) for mask in source_masks]
    assert seeds_seen == [42, 1_000_042, 43]
    np.testing.assert_array_equal(masks_seen[0], expected_dhw[0])
    np.testing.assert_array_equal(masks_seen[1], expected_dhw[0])
    np.testing.assert_array_equal(masks_seen[2], expected_dhw[1])

    # Rejection never edits the conditioning source or the mask paired with
    # the accepted image.
    for index, source_mask in enumerate(source_masks):
        source_path = real_seg_dir / f"Mets_{index + 1:03d}" / "seg.nii.gz"
        saved_path = output_dir / f"{index:05d}" / "seg.nii.gz"
        np.testing.assert_array_equal(nib.load(source_path).get_fdata(), source_mask)
        np.testing.assert_array_equal(nib.load(saved_path).get_fdata(), source_mask)

def test_reject_mode_fails_closed_when_fixed_mask_attempt_cap_is_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real_seg_dir = tmp_path / "masks"
    source_mask = _write_mask(
        real_seg_dir / "Mets_001" / "seg.nii.gz",
        (1, 1, 1),
    )
    config = _pipeline_config(
        tmp_path,
        real_seg_dir,
        num_images=1,
        max_image_attempts=2,
    )
    output_dir = tmp_path / "generated"
    _patch_non_sampling_dependencies(monkeypatch)

    masks_seen: list[np.ndarray] = []
    seeds_seen: list[int | None] = []

    def always_invalid(
        seg_binary: np.ndarray,
        *_args: object,
        noise_seed: int | None = None,
        **_kwargs: object,
    ) -> np.ndarray:
        masks_seen.append(seg_binary.copy())
        seeds_seen.append(noise_seed)
        return _image_with_tissue((1, 3, 3))

    monkeypatch.setattr(generation, "_generate_bravo", always_invalid)

    with pytest.raises(
        RuntimeError,
        match="no image passed quality control after 2 deterministic draws",
    ):
        generation.run_3d_pipeline(config, output_dir)

    expected_dhw = np.transpose(source_mask, (2, 0, 1))
    assert seeds_seen == [42, 1_000_042]
    assert len(masks_seen) == 2
    np.testing.assert_array_equal(masks_seen[0], expected_dhw)
    np.testing.assert_array_equal(masks_seen[1], expected_dhw)
    np.testing.assert_array_equal(
        nib.load(real_seg_dir / "Mets_001" / "seg.nii.gz").get_fdata(),
        source_mask,
    )
    assert not (output_dir / "00000").exists()


def test_reject_qc_ignores_the_discarded_padding_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    real_seg_dir = tmp_path / "masks"
    source_mask = _write_mask(
        real_seg_dir / "Mets_001" / "seg.nii.gz",
        (1, 1, 1),
    )
    config = _pipeline_config(
        tmp_path,
        real_seg_dir,
        num_images=1,
        max_image_attempts=2,
        generation_depth=4,
        trim_slices=1,
        expected_real_depth=3,
    )
    output_dir = tmp_path / "generated"
    _patch_non_sampling_dependencies(monkeypatch)

    generated = np.zeros((4, 4, 4), dtype=np.float32)
    generated[1, 1, 1] = 1.0
    generated[3, 3, 3] = 1.0  # disconnected artifact in discarded padding only
    calls = 0

    def generate_once(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return generated.copy()

    monkeypatch.setattr(generation, "_generate_bravo", generate_once)
    generation.run_3d_pipeline(config, output_dir)

    assert calls == 1
    np.testing.assert_array_equal(
        nib.load(output_dir / "00000" / "seg.nii.gz").get_fdata(),
        source_mask,
    )
