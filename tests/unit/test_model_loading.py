"""Unit tests for diffusion model loading utilities.

Tests the detect_wrapper_type function and load_diffusion_model
with various checkpoint formats and wrapper types.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from medgen.diffusion.loading import (
    detect_wrapper_type,
    load_diffusion_model,
    load_diffusion_model_with_metadata,
    LoadedModel,
    _resolve_channels,
)


class TestDetectWrapperType:
    """Test wrapper type detection from state dict keys."""

    def test_raw_model(self):
        """Raw model has no 'model.' prefix in keys."""
        state_dict = {
            'time_embed.0.weight': torch.randn(256, 128),
            'time_embed.0.bias': torch.randn(256),
            'down_blocks.0.resnets.0.conv1.weight': torch.randn(128, 1, 3, 3),
        }
        assert detect_wrapper_type(state_dict) == 'raw'

    def test_score_aug_wrapper(self):
        """ScoreAug wrapper has 'model.' prefix and omega_mlp keys."""
        state_dict = {
            'model.time_embed.0.weight': torch.randn(256, 128),
            'model.down_blocks.0.resnets.0.conv1.weight': torch.randn(128, 1, 3, 3),
            'combined_time_embed.omega_mlp.0.weight': torch.randn(256, 16),
            'combined_time_embed.omega_mlp.0.bias': torch.randn(256),
        }
        assert detect_wrapper_type(state_dict) == 'score_aug'

    def test_mode_embed_wrapper(self):
        """ModeEmbed wrapper has 'model.' prefix and mode_mlp keys."""
        state_dict = {
            'model.time_embed.0.weight': torch.randn(256, 128),
            'model.down_blocks.0.resnets.0.conv1.weight': torch.randn(128, 1, 3, 3),
            'mode_time_embed.mode_mlp.0.weight': torch.randn(256, 4),
            'mode_time_embed.mode_mlp.0.bias': torch.randn(256),
        }
        assert detect_wrapper_type(state_dict) == 'mode_embed'

    def test_combined_wrapper(self):
        """Combined wrapper has both omega_mlp and mode_mlp keys."""
        state_dict = {
            'model.time_embed.0.weight': torch.randn(256, 128),
            'model.down_blocks.0.resnets.0.conv1.weight': torch.randn(128, 1, 3, 3),
            'combined_time_embed.omega_mlp.0.weight': torch.randn(256, 16),
            'combined_time_embed.omega_mlp.0.bias': torch.randn(256),
            'combined_time_embed.mode_mlp.0.weight': torch.randn(256, 4),
            'combined_time_embed.mode_mlp.0.bias': torch.randn(256),
        }
        assert detect_wrapper_type(state_dict) == 'combined'

    def test_wrapped_but_no_mlps_warns_and_returns_raw(self):
        """Wrapped model without recognized MLPs returns raw with warning."""
        state_dict = {
            'model.time_embed.0.weight': torch.randn(256, 128),
            'model.down_blocks.0.resnets.0.conv1.weight': torch.randn(128, 1, 3, 3),
            # No omega_mlp or mode_mlp
        }
        # Should log warning and return 'raw'
        result = detect_wrapper_type(state_dict)
        assert result == 'raw'


class TestResolveChannels:
    """Test _resolve_channels helper function."""

    def test_arg_takes_precedence(self):
        """Argument value overrides checkpoint config."""
        result = _resolve_channels(
            'in_channels',
            arg_value=3,
            model_config={'in_channels': 2},
            required=True,
        )
        assert result == 3

    def test_checkpoint_fallback(self):
        """Uses checkpoint config when arg not provided."""
        result = _resolve_channels(
            'in_channels',
            arg_value=None,
            model_config={'in_channels': 2},
            required=True,
        )
        assert result == 2

    def test_raises_when_required_and_missing(self):
        """Raises ValueError when required and not found anywhere."""
        with pytest.raises(ValueError, match="in_channels not provided"):
            _resolve_channels(
                'in_channels',
                arg_value=None,
                model_config={},
                required=True,
            )


class TestLoadedModel:
    """Test LoadedModel dataclass."""

    def test_dataclass_fields(self):
        """LoadedModel has expected fields."""
        model = MagicMock()
        config = {'training': {'batch_size': 8}}

        result = LoadedModel(
            model=model,
            config=config,
            wrapper_type='raw',
            epoch=10,
            checkpoint_path='/path/to/checkpoint.pt',
        )

        assert result.model is model
        assert result.config == config
        assert result.wrapper_type == 'raw'
        assert result.epoch == 10
        assert result.checkpoint_path == '/path/to/checkpoint.pt'


class TestLoadDiffusionModelWithCheckpoint:
    """Integration-style tests for load_diffusion_model (mocked).

    These tests verify the loading logic without creating real models.
    """

    @patch('medgen.diffusion.loading.torch.load')
    @patch('medgen.diffusion.loading.DiffusionModelUNet')
    @patch('medgen.diffusion.loading.create_conditioning_wrapper')
    def test_load_raw_model(self, mock_wrapper, mock_unet_cls, mock_torch_load):
        """Load raw model (no wrapper)."""
        # Setup mocks
        mock_model = MagicMock()
        mock_unet_cls.return_value = mock_model
        mock_model.load_state_dict = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        mock_torch_load.return_value = {
            'model_state_dict': {
                'time_embed.0.weight': torch.randn(512, 128),
            },
            'config': {},
            'model_config': {
                'channels': [128, 256, 256],
                'attention_levels': [False, True, True],
                'num_res_blocks': 1,
                'num_head_channels': 256,
            },
            'epoch': 50,
        }

        result = load_diffusion_model_with_metadata(
            checkpoint_path='/path/to/model.pt',
            device=torch.device('cpu'),
            in_channels=2,
            out_channels=1,
        )

        assert result.wrapper_type == 'raw'
        assert result.epoch == 50
        assert mock_unet_cls.called
        # Wrapper not called for raw model
        mock_wrapper.assert_not_called()

    @patch('medgen.diffusion.loading.torch.load')
    @patch('medgen.diffusion.loading.DiffusionModelUNet')
    @patch('medgen.diffusion.loading.create_conditioning_wrapper')
    def test_load_combined_wrapper(self, mock_wrapper, mock_unet_cls, mock_torch_load):
        """Load model with combined wrapper."""
        mock_base_model = MagicMock()
        mock_wrapped_model = MagicMock()
        mock_unet_cls.return_value = mock_base_model
        mock_wrapper.return_value = (mock_wrapped_model, 'combined')

        # Setup wrapped model methods
        mock_wrapped_model.load_state_dict = MagicMock()
        mock_wrapped_model.to = MagicMock(return_value=mock_wrapped_model)
        mock_wrapped_model.eval = MagicMock(return_value=mock_wrapped_model)

        mock_torch_load.return_value = {
            'model_state_dict': {
                'model.time_embed.0.weight': torch.randn(512, 128),
                'combined_time_embed.omega_mlp.0.weight': torch.randn(512, 16),
                'combined_time_embed.mode_mlp.0.weight': torch.randn(512, 4),
            },
            'config': {},
            'model_config': {},
            'epoch': 100,
        }

        result = load_diffusion_model_with_metadata(
            checkpoint_path='/path/to/model.pt',
            device=torch.device('cpu'),
            in_channels=2,
            out_channels=1,
        )

        assert result.wrapper_type == 'combined'
        mock_wrapper.assert_called_once()
        call_kwargs = mock_wrapper.call_args[1]
        assert call_kwargs['use_omega'] is True
        assert call_kwargs['use_mode'] is True

    @patch('medgen.diffusion.loading.torch.load')
    @patch('medgen.diffusion.loading.DiffusionModelUNet')
    def test_uses_checkpoint_arch_params(self, mock_unet_cls, mock_torch_load):
        """Load uses architecture params from checkpoint."""
        mock_model = MagicMock()
        mock_unet_cls.return_value = mock_model
        mock_model.load_state_dict = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        mock_torch_load.return_value = {
            'model_state_dict': {
                'time_embed.0.weight': torch.randn(512, 128),
            },
            'model_config': {
                'in_channels': 3,
                'out_channels': 2,
                'channels': [64, 128, 256],
                'attention_levels': [False, False, True],
                'num_res_blocks': 2,
                'num_head_channels': 128,
                'spatial_dims': 2,
            },
        }

        # Don't pass explicit channels - should use checkpoint
        load_diffusion_model(
            checkpoint_path='/path/to/model.pt',
            device=torch.device('cpu'),
        )

        # Verify UNet was created with checkpoint params
        call_kwargs = mock_unet_cls.call_args[1]
        assert call_kwargs['channels'] == (64, 128, 256)
        assert call_kwargs['attention_levels'] == (False, False, True)
        assert call_kwargs['num_res_blocks'] == 2
        assert call_kwargs['in_channels'] == 3
        assert call_kwargs['out_channels'] == 2

    @patch('medgen.diffusion.loading.torch.load')
    @patch('medgen.diffusion.loading.DiffusionModelUNet')
    def test_explicit_channels_override_checkpoint(self, mock_unet_cls, mock_torch_load):
        """Explicit channel args override checkpoint values."""
        mock_model = MagicMock()
        mock_unet_cls.return_value = mock_model
        mock_model.load_state_dict = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)

        mock_torch_load.return_value = {
            'model_state_dict': {
                'time_embed.0.weight': torch.randn(512, 128),
            },
            'model_config': {
                'in_channels': 3,
                'out_channels': 2,
            },
        }

        # Pass explicit channels
        load_diffusion_model(
            checkpoint_path='/path/to/model.pt',
            device=torch.device('cpu'),
            in_channels=5,
            out_channels=4,
        )

        call_kwargs = mock_unet_cls.call_args[1]
        assert call_kwargs['in_channels'] == 5
        assert call_kwargs['out_channels'] == 4

    @patch('medgen.diffusion.loading.torch.load')
    @patch('medgen.diffusion.loading.DiffusionModelUNet')
    def test_raises_without_channels(self, mock_unet_cls, mock_torch_load):
        """Raises ValueError when channels not provided and not in checkpoint."""
        mock_torch_load.return_value = {
            'model_state_dict': {},
            'model_config': {},  # No channels in checkpoint
        }

        with pytest.raises(ValueError, match="in_channels not provided"):
            load_diffusion_model(
                checkpoint_path='/path/to/model.pt',
                device=torch.device('cpu'),
                # No channels provided
            )

    @patch('medgen.diffusion.loading.torch.load')
    @patch('medgen.diffusion.loading.DiffusionModelUNet')
    @patch('medgen.diffusion.loading.torch.compile')
    def test_compilation(self, mock_compile, mock_unet_cls, mock_torch_load):
        """Model is compiled when compile_model=True."""
        mock_model = MagicMock()
        mock_compiled = MagicMock()
        mock_unet_cls.return_value = mock_model
        mock_model.load_state_dict = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_compile.return_value = mock_compiled

        mock_torch_load.return_value = {
            'model_state_dict': {
                'time_embed.0.weight': torch.randn(512, 128),
            },
            'model_config': {},
        }

        result = load_diffusion_model(
            checkpoint_path='/path/to/model.pt',
            device=torch.device('cpu'),
            in_channels=2,
            out_channels=1,
            compile_model=True,
        )

        mock_compile.assert_called_once()
        assert result is mock_compiled


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
