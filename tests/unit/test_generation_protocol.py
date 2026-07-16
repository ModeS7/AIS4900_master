"""Focused tests for deterministic, auditable generation protocol helpers."""

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from medgen.models.handoff import HandoffWrapper
from medgen.scripts.generate import (
    _derive_sample_seed,
    _discover_real_seg_files,
    _load_image_model_with_optional_handoff,
    _preflight_real_seg_files,
    _randn,
    _save_sample_directory_atomic,
    _validate_expected_strategy,
    _validate_real_seg_volume,
)


def _checkpoint(path: Path, *, spatial_dims: int, in_channels: int, out_channels: int,
                pixel_shift: float = 0.0) -> None:
    torch.save(
        {
            'model_state_dict': {},
            'config': {
                'strategy': 'rflow',
                'spatial_dims': spatial_dims,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'sigma_data': 0.5,
                'pixel': {
                    'rescale': False,
                    'pixel_shift': pixel_shift,
                    'pixel_scale': 1.0,
                },
            },
        },
        path,
    )


def test_generation_config_preserves_stochastic_default_and_exposes_processing_flags():
    config = OmegaConf.load(Path(__file__).parents[2] / 'configs' / 'generate.yaml')
    assert config.seed is None
    assert config.expected_strategy is None
    assert config.mask_outside_brain is True
    assert config.expected_real_cases is None
    assert config.expected_real_depth is None
    assert config.require_real_bravo_pairs is False


def test_expected_strategy_assertion_fails_when_protocol_and_config_differ():
    _validate_expected_strategy(OmegaConf.create({'strategy': 'rflow', 'expected_strategy': 'rflow'}))
    with pytest.raises(ValueError, match='protocol requires rflow'):
        _validate_expected_strategy(
            OmegaConf.create({'strategy': 'ddpm', 'expected_strategy': 'rflow'})
        )


def test_sample_seed_streams_are_stable_and_first_bravo_seed_is_base_plus_index():
    assert _derive_sample_seed(None, 4) is None
    assert _derive_sample_seed(42, 4, stream='bravo', attempt=0) == 46
    assert _derive_sample_seed(42, 4, stream='bravo', attempt=1) == 1_000_046
    assert _derive_sample_seed(42, 4, stream='seg', attempt=0) == 1_000_000_046


def test_local_seeded_noise_is_independent_of_global_rng_state():
    device = torch.device('cpu')
    first = _randn((2, 3, 4), device, seed=123)
    torch.manual_seed(999)
    _ = torch.randn(100)
    second = _randn((2, 3, 4), device, seed=123)
    third = _randn((2, 3, 4), device, seed=124)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first, third)


def test_real_mask_pool_requires_exact_count_and_never_cycles(tmp_path: Path):
    for patient_id in ('Mets_002', 'Mets_001'):
        patient = tmp_path / patient_id
        patient.mkdir()
        (patient / 'seg.nii.gz').touch()

    files = _discover_real_seg_files(tmp_path, num_images=2, expected_cases=2)
    assert [path.parent.name for path in files] == ['Mets_001', 'Mets_002']

    with pytest.raises(ValueError, match='Expected exactly 3'):
        _discover_real_seg_files(tmp_path, num_images=2, expected_cases=3)
    with pytest.raises(ValueError, match='not cycled'):
        _discover_real_seg_files(tmp_path, num_images=3)


def test_real_mask_and_bravo_patient_ids_must_match_exactly(tmp_path: Path):
    for patient_id in ('Mets_001', 'Mets_002'):
        patient = tmp_path / patient_id
        patient.mkdir()
        (patient / 'seg.nii.gz').touch()
    (tmp_path / 'Mets_001' / 'bravo.nii.gz').touch()
    extra = tmp_path / 'Mets_003'
    extra.mkdir()
    (extra / 'bravo.nii.gz').touch()

    with pytest.raises(ValueError, match='identifiers do not match'):
        _discover_real_seg_files(
            tmp_path,
            num_images=2,
            expected_cases=2,
            require_bravo_pairs=True,
        )


def test_selected_real_masks_are_preflighted_before_generation(tmp_path: Path):
    valid = tmp_path / 'Mets_001' / 'seg.nii.gz'
    invalid = tmp_path / 'Mets_002' / 'seg.nii.gz'
    valid.parent.mkdir()
    invalid.parent.mkdir()
    affine = np.eye(4, dtype=np.float32)
    valid_data = np.zeros((4, 4, 3), dtype=np.float32)
    valid_data[1, 1, 1] = 1.0
    nib.save(nib.Nifti1Image(valid_data, affine), valid)
    nib.save(nib.Nifti1Image(np.full((4, 4, 3), 0.5, dtype=np.float32), affine), invalid)

    with pytest.raises(ValueError, match='not binary'):
        _preflight_real_seg_files([valid, invalid], num_images=2, image_size=4)

    # Only selected masks are preflighted, permitting a small smoke subset of
    # a larger exact input pool.
    _preflight_real_seg_files([valid, invalid], num_images=1, image_size=4)


@pytest.mark.parametrize(
    ('volume', 'message'),
    [
        (np.zeros((4, 4, 3), dtype=np.float32), 'empty'),
        (np.full((4, 4, 3), 0.5, dtype=np.float32), 'not binary'),
        (np.full((4, 4, 3), np.nan, dtype=np.float32), 'non-finite'),
        (np.ones((3, 4, 3), dtype=np.float32), 'in-plane shape'),
    ],
)
def test_real_mask_validation_rejects_malformed_inputs(
    volume: np.ndarray,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        _validate_real_seg_volume(volume, source=Path('case/seg.nii.gz'), image_size=4)


def test_real_mask_validation_can_lock_expected_depth():
    volume = np.zeros((4, 4, 3), dtype=np.float32)
    volume[1, 1, 1] = 1.0
    with pytest.raises(ValueError, match='depth 3, expected 4'):
        _validate_real_seg_volume(
            volume,
            source=Path('case/seg.nii.gz'),
            image_size=4,
            expected_depth=4,
        )


def test_atomic_sample_publish_reloads_all_niftis_and_refuses_overwrite(tmp_path: Path):
    seg = np.zeros((4, 4, 3), dtype=np.float32)
    seg[1, 1, 1] = 1.0
    bravo = np.linspace(0, 1, seg.size, dtype=np.float32).reshape(seg.shape)

    published = _save_sample_directory_atomic(
        tmp_path,
        0,
        {'seg.nii.gz': seg, 'bravo.nii.gz': bravo},
        voxel_size=(1.0, 1.0, 1.0),
    )
    assert published == tmp_path / '00000'
    assert (published / 'seg.nii.gz').is_file()
    assert (published / 'bravo.nii.gz').is_file()
    assert not list(tmp_path.glob('.*.tmp-*'))

    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        _save_sample_directory_atomic(
            tmp_path,
            0,
            {'seg.nii.gz': seg, 'bravo.nii.gz': bravo},
            voxel_size=(1.0, 1.0, 1.0),
        )


@pytest.mark.parametrize(
    ('spatial_dims', 'in_channels', 'out_channels'),
    [(2, 2, 1), (3, 2, 1)],
)
def test_shared_image_loader_builds_handoff_for_2d_and_3d(
    tmp_path: Path,
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
):
    low = tmp_path / 'low.pt'
    high = tmp_path / 'high.pt'
    _checkpoint(
        low,
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    _checkpoint(
        high,
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
    )
    config = OmegaConf.create({
        'image_model': str(low),
        'image_model_high_t': str(high),
        'handoff_t': 0.25,
        'strategy': 'rflow',
    })
    loaded = [torch.nn.Identity(), torch.nn.Identity()]

    with patch('medgen.scripts.generate.load_diffusion_model', side_effect=loaded) as loader:
        model = _load_image_model_with_optional_handoff(
            config,
            torch.device('cpu'),
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=spatial_dims,
        )

    assert isinstance(model, HandoffWrapper)
    assert loader.call_count == 2
    assert all(call.kwargs['spatial_dims'] == spatial_dims for call in loader.call_args_list)


def test_handoff_loader_rejects_incompatible_pixel_normalization(tmp_path: Path):
    low = tmp_path / 'low.pt'
    high = tmp_path / 'high.pt'
    _checkpoint(low, spatial_dims=3, in_channels=2, out_channels=1, pixel_shift=0.0)
    _checkpoint(high, spatial_dims=3, in_channels=2, out_channels=1, pixel_shift=0.5)
    config = OmegaConf.create({
        'image_model': str(low),
        'image_model_high_t': str(high),
        'handoff_t': 0.25,
        'strategy': 'rflow',
    })

    with patch('medgen.scripts.generate.load_diffusion_model', return_value=torch.nn.Identity()):
        with pytest.raises(ValueError, match='disagree on pixel'):
            _load_image_model_with_optional_handoff(
                config,
                torch.device('cpu'),
                in_channels=2,
                out_channels=1,
                spatial_dims=3,
            )


class _CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs):
        self.calls += 1
        return x


def test_100_step_quarter_handoff_routes_75_high_and_25_low_calls():
    high = _CountingModel()
    low = _CountingModel()
    wrapper = HandoffWrapper(high, low, handoff_t=0.25, num_train_timesteps=1000)
    x = torch.zeros(1, 1, 2, 2)

    # RFlow's 100-step grid is 1000, 990, ..., 10. The threshold is 250.
    for timestep in range(1000, 0, -10):
        wrapper(x=x, timesteps=torch.tensor([float(timestep)]))

    assert high.calls == 75
    assert low.calls == 25
