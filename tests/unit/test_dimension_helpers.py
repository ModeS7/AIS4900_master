"""Unit tests for dimension helper methods in DiffusionTrainerBase.

Tests the new unified dimension-aware helper methods that enable
2D/3D code sharing.
"""

import pytest
import torch


class MockTrainer2D:
    """Mock 2D trainer for testing dimension helpers."""

    spatial_dims = 2
    image_size = 64

    def __init__(self):
        from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase
        # Copy helper methods from base class
        self._expand_timesteps = DiffusionTrainerBase._expand_timesteps.__get__(self)
        self._get_spatial_shape = DiffusionTrainerBase._get_spatial_shape.__get__(self)
        self._get_noise_shape = DiffusionTrainerBase._get_noise_shape.__get__(self)
        self._extract_center_slice = DiffusionTrainerBase._extract_center_slice.__get__(self)
        self._validate_tensor_shape = DiffusionTrainerBase._validate_tensor_shape.__get__(self)


class MockTrainer3D:
    """Mock 3D trainer for testing dimension helpers."""

    spatial_dims = 3
    volume_depth = 32
    volume_height = 64
    volume_width = 64

    def __init__(self):
        from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase
        # Copy helper methods from base class
        self._expand_timesteps = DiffusionTrainerBase._expand_timesteps.__get__(self)
        self._get_spatial_shape = DiffusionTrainerBase._get_spatial_shape.__get__(self)
        self._get_noise_shape = DiffusionTrainerBase._get_noise_shape.__get__(self)
        self._extract_center_slice = DiffusionTrainerBase._extract_center_slice.__get__(self)
        self._validate_tensor_shape = DiffusionTrainerBase._validate_tensor_shape.__get__(self)


class TestExpandTimesteps:
    """Test _expand_timesteps helper."""

    def test_2d_from_1d(self):
        """2D: [B] -> [B, 1, 1, 1]"""
        trainer = MockTrainer2D()
        t = torch.tensor([0.5, 0.7])
        expanded = trainer._expand_timesteps(t)
        assert expanded.shape == (2, 1, 1, 1)

    def test_2d_from_2d(self):
        """2D: [B, 1] -> [B, 1, 1, 1]"""
        trainer = MockTrainer2D()
        t = torch.tensor([[0.5], [0.7]])
        expanded = trainer._expand_timesteps(t)
        assert expanded.shape == (2, 1, 1, 1)

    def test_3d_from_1d(self):
        """3D: [B] -> [B, 1, 1, 1, 1]"""
        trainer = MockTrainer3D()
        t = torch.tensor([0.5, 0.7])
        expanded = trainer._expand_timesteps(t)
        assert expanded.shape == (2, 1, 1, 1, 1)

    def test_3d_from_2d(self):
        """3D: [B, 1] -> [B, 1, 1, 1, 1]"""
        trainer = MockTrainer3D()
        t = torch.tensor([[0.5], [0.7]])
        expanded = trainer._expand_timesteps(t)
        assert expanded.shape == (2, 1, 1, 1, 1)

    def test_preserves_values(self):
        """Expanded timesteps preserve original values."""
        trainer = MockTrainer2D()
        t = torch.tensor([0.3, 0.8])
        expanded = trainer._expand_timesteps(t)
        assert torch.allclose(expanded.squeeze(), t)


class TestGetSpatialShape:
    """Test _get_spatial_shape helper."""

    def test_2d_shape(self):
        """2D returns (H, W)."""
        trainer = MockTrainer2D()
        shape = trainer._get_spatial_shape()
        assert shape == (64, 64)

    def test_3d_shape(self):
        """3D returns (D, H, W)."""
        trainer = MockTrainer3D()
        shape = trainer._get_spatial_shape()
        assert shape == (32, 64, 64)


class TestGetNoiseShape:
    """Test _get_noise_shape helper."""

    def test_2d_shape(self):
        """2D returns [B, C, H, W]."""
        trainer = MockTrainer2D()
        shape = trainer._get_noise_shape(batch_size=4, channels=2)
        assert shape == (4, 2, 64, 64)

    def test_3d_shape(self):
        """3D returns [B, C, D, H, W]."""
        trainer = MockTrainer3D()
        shape = trainer._get_noise_shape(batch_size=2, channels=1)
        assert shape == (2, 1, 32, 64, 64)


class TestExtractCenterSlice:
    """Test _extract_center_slice helper."""

    def test_2d_noop(self):
        """2D: tensor is returned unchanged."""
        trainer = MockTrainer2D()
        tensor = torch.randn(4, 2, 64, 64)
        result = trainer._extract_center_slice(tensor)
        assert result is tensor  # Same object
        assert result.shape == (4, 2, 64, 64)

    def test_3d_extracts_center(self):
        """3D: extracts center slice along depth dimension."""
        trainer = MockTrainer3D()
        tensor = torch.randn(2, 1, 32, 64, 64)
        # Set a distinctive value at center slice
        tensor[:, :, 16, :, :] = 42.0
        result = trainer._extract_center_slice(tensor)
        assert result.shape == (2, 1, 64, 64)
        assert torch.all(result == 42.0)

    def test_3d_odd_depth(self):
        """3D with odd depth uses floor division for center."""
        trainer = MockTrainer3D()
        tensor = torch.randn(2, 1, 33, 64, 64)  # Odd depth
        result = trainer._extract_center_slice(tensor)
        assert result.shape == (2, 1, 64, 64)


class TestValidateTensorShape:
    """Test _validate_tensor_shape helper."""

    def test_2d_valid(self):
        """2D: 4D tensor passes."""
        trainer = MockTrainer2D()
        tensor = torch.randn(4, 2, 64, 64)
        trainer._validate_tensor_shape(tensor, "test_tensor")  # Should not raise

    def test_2d_invalid_3d(self):
        """2D: 3D tensor fails."""
        trainer = MockTrainer2D()
        tensor = torch.randn(4, 2, 64)
        with pytest.raises(ValueError, match="has 3 dims, expected 4"):
            trainer._validate_tensor_shape(tensor, "test_tensor")

    def test_2d_invalid_5d(self):
        """2D: 5D tensor fails."""
        trainer = MockTrainer2D()
        tensor = torch.randn(4, 2, 64, 64, 1)
        with pytest.raises(ValueError, match="has 5 dims, expected 4"):
            trainer._validate_tensor_shape(tensor, "test_tensor")

    def test_3d_valid(self):
        """3D: 5D tensor passes."""
        trainer = MockTrainer3D()
        tensor = torch.randn(2, 1, 32, 64, 64)
        trainer._validate_tensor_shape(tensor, "test_tensor")  # Should not raise

    def test_3d_invalid_4d(self):
        """3D: 4D tensor fails."""
        trainer = MockTrainer3D()
        tensor = torch.randn(2, 1, 64, 64)
        with pytest.raises(ValueError, match="has 4 dims, expected 5"):
            trainer._validate_tensor_shape(tensor, "test_tensor")

    def test_error_includes_name(self):
        """Error message includes tensor name."""
        trainer = MockTrainer2D()
        tensor = torch.randn(4, 2, 64)
        with pytest.raises(ValueError, match="my_images"):
            trainer._validate_tensor_shape(tensor, "my_images")


class TestMsssimFactory:
    """Test _create_msssim_fn factory."""

    def test_2d_returns_partial_with_dims_2(self):
        """2D trainer gets MS-SSIM function with spatial_dims=2."""
        trainer = MockTrainer2D()
        from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase
        fn = DiffusionTrainerBase._create_msssim_fn(trainer)
        # Verify it's a partial with correct spatial_dims
        assert fn.keywords.get('spatial_dims') == 2

    def test_3d_returns_partial_with_dims_3(self):
        """3D trainer gets MS-SSIM function with spatial_dims=3."""
        trainer = MockTrainer3D()
        from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase
        fn = DiffusionTrainerBase._create_msssim_fn(trainer)
        # Verify it's a partial with correct spatial_dims
        assert fn.keywords.get('spatial_dims') == 3


class TestLpipsFactory:
    """Test _create_lpips_fn factory."""

    def test_disabled_returns_none(self):
        """Returns None when log_lpips is False."""
        trainer = MockTrainer2D()
        trainer.log_lpips = False
        from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase
        fn = DiffusionTrainerBase._create_lpips_fn(trainer)
        assert fn is None

    def test_2d_returns_2d_function(self):
        """2D trainer with log_lpips=True gets 2D LPIPS function."""
        trainer = MockTrainer2D()
        trainer.log_lpips = True
        from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase
        fn = DiffusionTrainerBase._create_lpips_fn(trainer)
        from medgen.metrics import compute_lpips
        assert fn is compute_lpips

    def test_3d_returns_3d_function(self):
        """3D trainer with log_lpips=True gets 3D LPIPS function."""
        trainer = MockTrainer3D()
        trainer.log_lpips = True
        from medgen.pipeline.diffusion_trainer_base import DiffusionTrainerBase
        fn = DiffusionTrainerBase._create_lpips_fn(trainer)
        from medgen.metrics import compute_lpips_3d
        assert fn is compute_lpips_3d


class TestSpatialDimsParameter:
    """Test DiffusionTrainer spatial_dims parameter."""

    def test_default_is_2d(self):
        """Default spatial_dims is 2."""
        from medgen.pipeline.trainer import DiffusionTrainer
        # Just check the default value in signature
        import inspect
        sig = inspect.signature(DiffusionTrainer.__init__)
        assert sig.parameters['spatial_dims'].default == 2

    def test_invalid_spatial_dims_raises(self):
        """Invalid spatial_dims raises ValueError."""
        from medgen.pipeline.trainer import DiffusionTrainer
        from omegaconf import OmegaConf

        # Create minimal config
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'strategy': {'name': 'rflow', 'num_train_timesteps': 1000},
            'model': {'image_size': 64},
            'training': {'batch_size': 4},
            'paths': {},
        })

        with pytest.raises(ValueError, match="spatial_dims must be 2 or 3"):
            DiffusionTrainer(cfg, spatial_dims=4)

    def test_convenience_constructors_exist(self):
        """Convenience constructors are available."""
        from medgen.pipeline.trainer import DiffusionTrainer
        assert hasattr(DiffusionTrainer, 'create_2d')
        assert hasattr(DiffusionTrainer, 'create_3d')
        assert callable(DiffusionTrainer.create_2d)
        assert callable(DiffusionTrainer.create_3d)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
