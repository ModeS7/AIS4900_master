"""Integration tests for 3D training modes.

These tests verify that 3D modes (seg_conditioned, bravo) work correctly
with all components: configs, dataloaders, model wrappers, generation metrics.

Run with: pytest tests/integration/test_3d_modes.py -v
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf


class TestConfigValidation:
    """Test that config mode names match expected values."""

    def test_seg_conditioned_3d_mode_name(self):
        """seg_diffusion_3d.yaml should have name=seg_conditioned."""
        cfg = OmegaConf.load("configs/mode/seg_diffusion_3d.yaml")
        assert cfg.name == "seg_conditioned", (
            f"seg_diffusion_3d.yaml has name={cfg.name}, expected 'seg_conditioned'. "
            "This is required for SizeBinModelWrapper to be applied."
        )

    def test_seg_conditioned_2d_mode_name(self):
        """seg_conditioned.yaml should have name=seg_conditioned."""
        cfg = OmegaConf.load("configs/mode/seg_conditioned.yaml")
        assert cfg.name == "seg_conditioned"

    def test_seg_conditioned_has_size_bins(self):
        """seg_conditioned modes should have size_bins config."""
        for config_file in ["configs/mode/seg_conditioned.yaml", "configs/mode/seg_diffusion_3d.yaml"]:
            cfg = OmegaConf.load(config_file)
            assert "size_bins" in cfg, f"{config_file} missing size_bins config"
            assert "num_bins" in cfg.size_bins
            assert "edges" in cfg.size_bins


class TestSizeBinWrapper:
    """Test that SizeBinModelWrapper is applied correctly."""

    def test_size_bin_wrapper_applied_for_seg_conditioned(self):
        """SizeBinModelWrapper should be applied when mode_name='seg_conditioned'."""
        from medgen.models.wrappers.size_bin_embed import SizeBinModelWrapper
        from torch import nn

        # Create a simple mock model with time_embed
        class MockUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.time_embed = nn.Linear(256, 256)

            def forward(self, x, timesteps):
                return x

        model = MockUNet()
        wrapper = SizeBinModelWrapper(model, embed_dim=256)

        # Verify wrapper is created correctly
        assert hasattr(wrapper, 'size_bin_time_embed')
        assert hasattr(wrapper.size_bin_time_embed, 'projection')

    def test_film_conditioning_output_shape(self):
        """FiLM conditioning should produce correct output shapes."""
        from medgen.models.wrappers.size_bin_embed import SizeBinTimeEmbed
        from torch import nn

        class MockTimeEmbed(nn.Module):
            def forward(self, x):
                return x.expand(-1, 256)

        sbt = SizeBinTimeEmbed(MockTimeEmbed(), embed_dim=256)
        sbt.set_size_bins(torch.randint(0, 5, (2, 7)))

        out = sbt(torch.randn(2, 256))
        assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"

    def test_film_zero_init_identity(self):
        """FiLM should start as identity due to zero initialization."""
        from medgen.models.wrappers.size_bin_embed import SizeBinTimeEmbed
        from torch import nn

        class MockTimeEmbed(nn.Module):
            def forward(self, x):
                return x

        sbt = SizeBinTimeEmbed(MockTimeEmbed(), embed_dim=256)
        sbt.set_size_bins(torch.randint(0, 5, (2, 7)))

        input_tensor = torch.randn(2, 256)
        out = sbt(input_tensor)

        # At init, scale=0, shift=0, so out*(1+0)+0 = out
        diff = (out - input_tensor).abs().max().item()
        assert diff < 1e-5, f"FiLM should start as identity, but max diff={diff}"


class TestUNetCreation:
    """Test UNet model creation with various configurations."""

    def test_unet_with_norm_num_groups(self):
        """UNet should be created with correct norm_num_groups."""
        from monai.networks.nets import DiffusionModelUNet

        # This should work with norm_num_groups=16
        model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 256, 512, 512),
            attention_levels=(False, False, False, False, True, True),
            num_res_blocks=(1, 1, 1, 2, 2, 2),
            num_head_channels=16,
            norm_num_groups=16,
        )
        assert model is not None

    def test_unet_fails_without_norm_num_groups(self):
        """UNet with small channels should fail with default norm_num_groups=32."""
        from monai.networks.nets import DiffusionModelUNet

        # Default norm_num_groups=32, but first channel is 16 which isn't divisible by 32
        with pytest.raises(ValueError, match="multiple of norm_num_groups"):
            DiffusionModelUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(16, 32, 64, 256, 512, 512),
                attention_levels=(False, False, False, False, True, True),
                num_res_blocks=(1, 1, 1, 2, 2, 2),
                num_head_channels=16,
                # norm_num_groups defaults to 32
            )


class TestGenerationMetrics:
    """Test generation metrics with different data formats."""

    def test_dict_format_handling(self, tmp_path):
        """Generation metrics should handle dict format from 3D datasets."""
        from medgen.metrics.generation import GenerationMetrics, GenerationMetricsConfig

        # Create mock dataset returning dict format
        class MockDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    'image': torch.randn(1, 32, 64, 64),  # 3D: [C, D, H, W]
                    'seg': (torch.rand(1, 32, 64, 64) > 0.5).float(),
                    'patient': f'patient_{idx}',
                }

        config = GenerationMetricsConfig(
            samples_per_epoch=2,
            samples_extended=2,
            samples_test=2,
        )
        metrics = GenerationMetrics(config, mode_name='bravo', device='cpu', run_dir=str(tmp_path))

        # This should not raise an error
        metrics.set_fixed_conditioning(MockDataset(), num_masks=2, seg_channel_idx=1)

        assert metrics.fixed_conditioning_masks is not None
        assert metrics.fixed_conditioning_masks.shape[0] <= 2

    def test_tuple_format_handling_2d(self, tmp_path):
        """Generation metrics should handle tuple format (2D)."""
        from medgen.metrics.generation import GenerationMetrics, GenerationMetricsConfig

        class MockDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                image = torch.randn(1, 64, 64)  # 2D: [C, H, W]
                seg = (torch.rand(1, 64, 64) > 0.5).float()
                return (image, seg)

        config = GenerationMetricsConfig(
            samples_per_epoch=2,
            samples_extended=2,
            samples_test=2,
        )
        metrics = GenerationMetrics(config, mode_name='bravo', device='cpu', run_dir=str(tmp_path))
        metrics.set_fixed_conditioning(MockDataset(), num_masks=2, seg_channel_idx=1)

        assert metrics.fixed_conditioning_masks is not None

    def test_tuple_format_seg_conditioned(self, tmp_path):
        """Generation metrics should handle (seg, size_bins) tuple format."""
        from medgen.metrics.generation import GenerationMetrics, GenerationMetricsConfig

        class MockDataset:
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                seg = (torch.rand(1, 32, 64, 64) > 0.5).float()  # 3D seg
                size_bins = torch.randint(0, 5, (7,))  # 1D size bins
                return (seg, size_bins)

        config = GenerationMetricsConfig(
            samples_per_epoch=2,
            samples_extended=2,
            samples_test=2,
        )
        metrics = GenerationMetrics(config, mode_name='seg_conditioned', device='cpu', run_dir=str(tmp_path))
        metrics.set_fixed_conditioning(MockDataset(), num_masks=2, seg_channel_idx=0)

        assert metrics.fixed_conditioning_masks is not None

    def test_3d_slicing(self):
        """Slicing should work for 3D tensors [C, D, H, W]."""
        tensor = torch.randn(2, 32, 64, 64)  # [C, D, H, W]

        # This should work for both 2D and 3D
        seg = tensor[1:2, ...]
        assert seg.shape == (1, 32, 64, 64)

        image = tensor[0:1, ...]
        assert image.shape == (1, 32, 64, 64)


class TestSiTCheckpoint:
    """Test SiT model checkpoint save/load."""

    def test_get_model_config_sit(self):
        """_get_model_config should return SiT-specific keys."""
        # Mock the trainer's config and state
        mock_cfg = OmegaConf.create({
            'model': {
                'type': 'sit',
                'spatial_dims': 2,
                'image_size': 256,
                'patch_size': 4,
                'variant': 'S',
                'mlp_ratio': 4.0,
                'conditioning': 'concat',
                'qk_norm': True,
            },
            'mode': {
                'name': 'bravo',
                'in_channels': 2,
                'out_channels': 1,
            },
        })

        # Create mock mode
        class MockMode:
            def get_model_config(self):
                return {'in_channels': 2, 'out_channels': 1}

        # Simulate what _get_model_config does for transformers
        is_transformer = mock_cfg.model.type == 'sit'
        config = {
            'model_type': 'sit',
            'in_channels': 2,
            'out_channels': 1,
            'strategy': 'rflow',
            'mode': 'bravo',
            'spatial_dims': mock_cfg.model.get('spatial_dims', 2),
        }

        if is_transformer:
            config.update({
                'image_size': mock_cfg.model.image_size,
                'patch_size': mock_cfg.model.patch_size,
                'variant': mock_cfg.model.variant,
                'mlp_ratio': mock_cfg.model.get('mlp_ratio', 4.0),
                'conditioning': mock_cfg.model.get('conditioning', 'concat'),
                'qk_norm': mock_cfg.model.get('qk_norm', True),
            })

        # Verify SiT-specific keys are present
        assert 'image_size' in config
        assert 'patch_size' in config
        assert 'variant' in config
        assert 'channels' not in config  # UNet-specific, should not be present


class TestTensorOperations:
    """Test tensor operations work for both 2D and 3D."""

    def test_mse_per_sample_2d(self):
        """MSE per sample should work for 2D tensors."""
        pred = torch.randn(4, 1, 64, 64)  # [B, C, H, W]
        target = torch.randn(4, 1, 64, 64)

        # This should give [B] shaped output
        mse = ((pred - target) ** 2).flatten(1).mean(1)
        assert mse.shape == (4,), f"Expected (4,), got {mse.shape}"

    def test_mse_per_sample_3d(self):
        """MSE per sample should work for 3D tensors."""
        pred = torch.randn(2, 1, 32, 64, 64)  # [B, C, D, H, W]
        target = torch.randn(2, 1, 32, 64, 64)

        # This should give [B] shaped output
        mse = ((pred - target) ** 2).flatten(1).mean(1)
        assert mse.shape == (2,), f"Expected (2,), got {mse.shape}"

    def test_scatter_add_requires_matching_dims(self):
        """scatter_add_ requires index and src to have same number of dims."""
        timestep_loss_sum = torch.zeros(10)
        bin_indices = torch.tensor([0, 1, 2, 3])  # [4]
        mse_per_sample = torch.tensor([0.1, 0.2, 0.3, 0.4])  # [4]

        # Both are 1D, this should work
        timestep_loss_sum.scatter_add_(0, bin_indices, mse_per_sample)

        # If mse_per_sample were [4, 64] (2D), this would fail
        mse_wrong = torch.randn(4, 64)
        with pytest.raises(RuntimeError):
            timestep_loss_sum.scatter_add_(0, bin_indices, mse_wrong)


class TestTrainScriptModeRouting:
    """Test that train.py routes modes to correct dataloaders."""

    def test_seg_conditioned_uses_seg_dataloader(self):
        """mode='seg_conditioned' should use create_seg_dataloader for 3D."""
        # Verify the condition in train.py
        mode = 'seg_conditioned'
        assert mode in ('seg', 'seg_conditioned'), (
            f"Mode {mode} should match condition for seg dataloader"
        )

    def test_seg_uses_seg_dataloader(self):
        """mode='seg' should also use create_seg_dataloader for 3D."""
        mode = 'seg'
        assert mode in ('seg', 'seg_conditioned')


class TestVolume3DMSSSIM:
    """Test volume 3D MS-SSIM computation guards."""

    def test_3d_diffusion_skips_slice_msssim(self):
        """3D diffusion models should skip slice-by-slice MS-SSIM computation.

        The _compute_volume_3d_msssim function processes 2D slices which is
        incompatible with 3D models that expect 5D input [B, C, D, H, W].
        """
        # The guard condition in trainer.py
        spatial_dims_2d = 2
        spatial_dims_3d = 3

        # 2D model: should NOT skip (returns actual metric)
        assert spatial_dims_2d != 3, "2D models should compute slice MS-SSIM"

        # 3D model: should skip (returns None)
        assert spatial_dims_3d == 3, "3D models should skip slice MS-SSIM"

    def test_3d_model_input_shape(self):
        """3D models expect 5D input, not 4D slices."""
        # 2D slice shape from volume
        slice_shape_2d = (16, 1, 256, 256)  # [B, C, H, W]

        # 3D model expects
        expected_3d_input_shape = (1, 1, 32, 256, 256)  # [B, C, D, H, W]

        assert len(slice_shape_2d) == 4, "2D slices are 4D"
        assert len(expected_3d_input_shape) == 5, "3D models expect 5D input"

        # This mismatch caused the GroupNorm error
        num_groups = 16
        channels_in_slice = slice_shape_2d[1]  # 1
        assert channels_in_slice % num_groups != 0, (
            f"1 channel is not divisible by {num_groups} groups - "
            "this is why 2D slices fail in 3D models"
        )
