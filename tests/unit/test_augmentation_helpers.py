"""Unit tests for augmentation helper modules.

Modules covered:
- augmentation/score_aug_patterns.py
- augmentation/score_aug_omega.py
- augmentation/score_aug_wrapper.py
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch


# ============================================================================
# augmentation/score_aug_patterns.py
# ============================================================================

class TestCheckerboardMask:
    """Tests for _checkerboard_mask()."""

    def test_shape(self):
        from medgen.augmentation.score_aug_patterns import _checkerboard_mask
        mask = _checkerboard_mask(64, 64, grid_size=4, offset=False)
        assert mask.shape == (64, 64)

    def test_binary_values(self):
        from medgen.augmentation.score_aug_patterns import _checkerboard_mask
        mask = _checkerboard_mask(64, 64, grid_size=4, offset=False)
        unique = torch.unique(mask)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_approximately_half(self):
        from medgen.augmentation.score_aug_patterns import _checkerboard_mask
        mask = _checkerboard_mask(64, 64, grid_size=4, offset=False)
        frac = mask.sum().item() / mask.numel()
        assert 0.45 <= frac <= 0.55

    def test_offset_inverts(self):
        from medgen.augmentation.score_aug_patterns import _checkerboard_mask
        m1 = _checkerboard_mask(64, 64, grid_size=4, offset=False)
        m2 = _checkerboard_mask(64, 64, grid_size=4, offset=True)
        combined = m1 + m2
        assert torch.all(combined == 1.0)


class TestGridDropoutMask:
    """Tests for _grid_dropout_mask()."""

    def test_shape_and_binary(self):
        from medgen.augmentation.score_aug_patterns import _grid_dropout_mask
        mask = _grid_dropout_mask(64, 64, grid_size=4, drop_ratio=0.25, seed=42)
        assert mask.shape == (64, 64)
        unique = torch.unique(mask)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_drop_ratio(self):
        from medgen.augmentation.score_aug_patterns import _grid_dropout_mask
        mask = _grid_dropout_mask(64, 64, grid_size=4, drop_ratio=0.25, seed=42)
        # 4x4 = 16 cells, 25% drop = 4 cells dropped
        # Each cell = 16x16 = 256 pixels, so ~4*256 = 1024 dropped pixels
        dropped = mask.sum().item()
        total_pixels = 64 * 64
        ratio = dropped / total_pixels
        assert 0.20 <= ratio <= 0.30

    def test_deterministic(self):
        from medgen.augmentation.score_aug_patterns import _grid_dropout_mask
        m1 = _grid_dropout_mask(64, 64, grid_size=4, drop_ratio=0.25, seed=42)
        m2 = _grid_dropout_mask(64, 64, grid_size=4, drop_ratio=0.25, seed=42)
        assert torch.equal(m1, m2)


class TestCoarseDropoutMask:
    """Tests for _coarse_dropout_mask()."""

    def test_shape_and_binary(self):
        from medgen.augmentation.score_aug_patterns import _coarse_dropout_mask
        mask = _coarse_dropout_mask(64, 64, pattern_id=0)
        assert mask.shape == (64, 64)
        unique = torch.unique(mask)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_pattern_0_has_two_holes(self):
        from medgen.augmentation.score_aug_patterns import _coarse_dropout_mask
        mask = _coarse_dropout_mask(64, 64, pattern_id=0)
        # Top-left and bottom-right corners should have 1s
        assert mask[0, 0] == 1.0
        assert mask[-1, -1] == 1.0
        assert mask.sum().item() > 0

    def test_all_4_patterns_valid(self):
        from medgen.augmentation.score_aug_patterns import _coarse_dropout_mask
        for pid in range(4):
            mask = _coarse_dropout_mask(64, 64, pattern_id=pid)
            assert mask.sum().item() > 0, f"Pattern {pid} has no dropped pixels"


class TestPatchDropoutMask:
    """Tests for _patch_dropout_mask()."""

    def test_shape_and_binary(self):
        from medgen.augmentation.score_aug_patterns import _patch_dropout_mask
        mask = _patch_dropout_mask(64, 64, patch_size=8, drop_ratio=0.25, seed=42)
        assert mask.shape == (64, 64)
        unique = torch.unique(mask)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_deterministic(self):
        from medgen.augmentation.score_aug_patterns import _patch_dropout_mask
        m1 = _patch_dropout_mask(64, 64, patch_size=8, drop_ratio=0.25, seed=42)
        m2 = _patch_dropout_mask(64, 64, patch_size=8, drop_ratio=0.25, seed=42)
        assert torch.equal(m1, m2)


class TestGeneratePatternMask:
    """Tests for generate_pattern_mask()."""

    def test_all_16_patterns_2d(self):
        from medgen.augmentation.score_aug_patterns import generate_pattern_mask
        for pid in range(16):
            mask = generate_pattern_mask(pid, H=64, W=64)
            assert mask.shape == (64, 64), f"Pattern {pid} wrong shape"
            unique = torch.unique(mask)
            assert all(v in (0.0, 1.0) for v in unique.tolist()), f"Pattern {pid} non-binary"

    def test_3d_expands_depth(self):
        from medgen.augmentation.score_aug_patterns import generate_pattern_mask
        mask = generate_pattern_mask(0, H=64, W=64, D=16, spatial_dims=3)
        assert mask.shape == (16, 64, 64)

    def test_3d_requires_D(self):
        from medgen.augmentation.score_aug_patterns import generate_pattern_mask
        with pytest.raises(ValueError, match="D required"):
            generate_pattern_mask(0, H=64, W=64, D=None, spatial_dims=3)

    def test_3d_uniform_across_depth(self):
        from medgen.augmentation.score_aug_patterns import generate_pattern_mask
        mask = generate_pattern_mask(0, H=64, W=64, D=16, spatial_dims=3)
        # All depth slices should be identical
        for d in range(1, 16):
            assert torch.equal(mask[0], mask[d])

    def test_out_of_range_returns_zeros(self):
        from medgen.augmentation.score_aug_patterns import generate_pattern_mask
        mask = generate_pattern_mask(16, H=64, W=64)
        assert torch.all(mask == 0)

    def test_constants(self):
        from medgen.augmentation.score_aug_patterns import NUM_PATTERNS, PATTERN_NAMES
        assert NUM_PATTERNS == 16
        assert len(PATTERN_NAMES) == 16


class TestClearPatternCache:
    """Tests for clear_pattern_cache()."""

    def test_clears_without_error(self):
        from medgen.augmentation.score_aug_patterns import clear_pattern_cache
        clear_pattern_cache()
        clear_pattern_cache()  # second call should also succeed


# ============================================================================
# augmentation/score_aug_omega.py
# ============================================================================

class TestModeIntensityScale:
    """Tests for MODE_INTENSITY_SCALE and apply/inverse functions."""

    def test_known_scale_values(self):
        from medgen.augmentation.score_aug_omega import MODE_INTENSITY_SCALE
        assert MODE_INTENSITY_SCALE == {0: 0.85, 1: 1.15, 2: 0.92, 3: 1.08}

    def test_apply_none_mode_id(self):
        from medgen.augmentation.score_aug_omega import apply_mode_intensity_scale
        x = torch.ones(2, 1, 64, 64)
        scaled, scales = apply_mode_intensity_scale(x, mode_id=None)
        assert torch.equal(scaled, x)
        assert scales.item() == 1.0

    def test_apply_scales_correctly(self):
        from medgen.augmentation.score_aug_omega import apply_mode_intensity_scale
        x = torch.ones(2, 1, 64, 64)
        mode_id = torch.tensor([0, 0])  # bravo -> 0.85
        scaled, scales = apply_mode_intensity_scale(x, mode_id, spatial_dims=2)
        assert torch.allclose(scaled, torch.full_like(x, 0.85))

    def test_apply_3d_shape(self):
        from medgen.augmentation.score_aug_omega import apply_mode_intensity_scale
        x = torch.ones(2, 1, 16, 64, 64)
        mode_id = torch.tensor([1, 1])  # flair -> 1.15
        scaled, scales = apply_mode_intensity_scale(x, mode_id, spatial_dims=3)
        assert scales.shape == (2, 1, 1, 1, 1)
        assert torch.allclose(scaled, torch.full_like(x, 1.15))

    def test_inverse_roundtrip(self):
        from medgen.augmentation.score_aug_omega import (
            apply_mode_intensity_scale,
            inverse_mode_intensity_scale,
        )
        x = torch.rand(2, 1, 64, 64)
        mode_id = torch.tensor([2, 3])
        scaled, scales = apply_mode_intensity_scale(x, mode_id, spatial_dims=2)
        recovered = inverse_mode_intensity_scale(scaled, scales)
        assert torch.allclose(recovered, x, atol=1e-6)


class TestEncodeOmega:
    """Tests for encode_omega()."""

    def test_none_returns_identity(self):
        from medgen.augmentation.score_aug_omega import encode_omega
        enc = encode_omega(None, torch.device('cpu'))
        assert enc[0, 0] == 1.0  # identity marker

    def test_output_shape(self):
        from medgen.augmentation.score_aug_omega import encode_omega
        enc = encode_omega(None, torch.device('cpu'))
        assert enc.shape == (1, 36)

    def test_mode_id_one_hot(self):
        from medgen.augmentation.score_aug_omega import encode_omega
        mode_id = torch.tensor([2])
        enc = encode_omega(None, torch.device('cpu'), mode_id=mode_id)
        assert enc[0, 34] == 1.0  # dim 32 + 2 = 34

    def test_v2_rot90(self):
        from medgen.augmentation.score_aug_omega import encode_omega
        omega = {'v2': True, 'transforms': [('rot90', {'k': 2})]}
        enc = encode_omega(omega, torch.device('cpu'))
        assert enc[0, 0] == 1.0   # spatial active
        assert enc[0, 4] == 1.0   # rot90 type
        assert abs(enc[0, 10].item() - 2/3) < 1e-6  # k/3

    def test_v2_hflip(self):
        from medgen.augmentation.score_aug_omega import encode_omega
        omega = {'v2': True, 'transforms': [('hflip', {})]}
        enc = encode_omega(omega, torch.device('cpu'))
        assert enc[0, 5] == 1.0  # hflip type

    def test_v2_translate(self):
        from medgen.augmentation.score_aug_omega import encode_omega
        omega = {'v2': True, 'transforms': [('translate', {'dx': 0.1, 'dy': 0.2})]}
        enc = encode_omega(omega, torch.device('cpu'))
        assert enc[0, 1] == 1.0   # translation active
        assert abs(enc[0, 11].item() - 0.1) < 1e-6
        assert abs(enc[0, 12].item() - 0.2) < 1e-6

    def test_v2_pattern(self):
        from medgen.augmentation.score_aug_omega import encode_omega
        omega = {'v2': True, 'transforms': [('pattern', {'pattern_id': 5})]}
        enc = encode_omega(omega, torch.device('cpu'))
        assert enc[0, 3] == 1.0      # pattern active
        assert enc[0, 16 + 5] == 1.0  # one-hot at index 21

    def test_3d_rot90(self):
        from medgen.augmentation.score_aug_omega import encode_omega
        omega = {'v2': True, 'transforms': [('rot90_3d', {'axis': 'd', 'k': 1})]}
        enc = encode_omega(omega, torch.device('cpu'), spatial_dims=3)
        assert enc[0, 4] == 1.0  # rot90_d


# ============================================================================
# augmentation/score_aug_wrapper.py
# ============================================================================

class TestOmegaTimeEmbed:
    """Tests for OmegaTimeEmbed."""

    def test_buffer_exists(self):
        from medgen.augmentation.score_aug_wrapper import OmegaTimeEmbed
        original = nn.Linear(128, 128)
        ote = OmegaTimeEmbed(original, embed_dim=128)
        assert hasattr(ote, '_omega_encoding')
        assert ote._omega_encoding.shape == (1, 36)

    def test_set_encoding(self):
        from medgen.augmentation.score_aug_wrapper import OmegaTimeEmbed
        original = nn.Linear(128, 128)
        ote = OmegaTimeEmbed(original, embed_dim=128)
        new_enc = torch.ones(1, 36)
        ote.set_omega_encoding(new_enc)
        assert torch.equal(ote._omega_encoding, new_enc)

    def test_forward_adds_omega(self):
        from medgen.augmentation.score_aug_wrapper import OmegaTimeEmbed
        original = nn.Linear(128, 128)
        ote = OmegaTimeEmbed(original, embed_dim=128)
        t_emb = torch.randn(4, 128)

        # Get baseline from original only
        with torch.no_grad():
            original_out = original(t_emb)
            omega_out = ote.omega_mlp(ote._omega_encoding)
            expected = original_out + omega_out
            actual = ote(t_emb)

        assert torch.allclose(actual, expected, atol=1e-6)


class TestScoreAugModelWrapper:
    """Tests for ScoreAugModelWrapper."""

    def _make_mock_model(self):
        """Create a minimal mock model with time_embed."""
        model = nn.Module()
        model.time_embed = nn.Linear(128, 128)
        # Add a simple forward method
        model.forward = lambda x, timesteps: x
        model.register_forward_hook = lambda *a, **kw: None
        return model

    def test_replaces_time_embed(self):
        from medgen.augmentation.score_aug_wrapper import OmegaTimeEmbed, ScoreAugModelWrapper
        model = self._make_mock_model()
        wrapper = ScoreAugModelWrapper(model, embed_dim=128)
        assert isinstance(model.time_embed, OmegaTimeEmbed)

    def test_forward_calls_model(self):
        from medgen.augmentation.score_aug_wrapper import ScoreAugModelWrapper
        model = self._make_mock_model()
        # Replace forward with a mock to track calls
        call_log = []
        original_forward = model.forward
        def tracking_forward(x, timesteps):
            call_log.append((x.shape, timesteps.shape))
            return x
        model.forward = tracking_forward

        wrapper = ScoreAugModelWrapper(model, embed_dim=128)
        x = torch.randn(2, 1, 64, 64)
        t = torch.randint(0, 100, (2,))
        result = wrapper(x, t)
        assert len(call_log) == 1

    def test_parameters_without_omega(self):
        from medgen.augmentation.score_aug_wrapper import ScoreAugModelWrapper
        model = self._make_mock_model()
        wrapper = ScoreAugModelWrapper(model, embed_dim=128)

        all_params = set(id(p) for p in wrapper.parameters())
        omega_params = set(id(p) for p in wrapper.omega_time_embed.omega_mlp.parameters())
        without_omega = set(id(p) for p in wrapper.parameters_without_omega)

        # parameters_without_omega should exclude omega MLP params
        assert omega_params.issubset(all_params)
        assert len(without_omega & omega_params) == 0
