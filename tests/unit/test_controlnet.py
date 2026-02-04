"""Tests for ControlNet wrapper for conditional latent diffusion.

Tests cover ControlNet creation, ControlNetConditionedUNet wrapper,
gradient flow for frozen/unfrozen UNet, checkpoint I/O, and the
generation wrapper.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from omegaconf import OmegaConf

from medgen.models.controlnet import (
    create_controlnet_for_unet,
    freeze_unet_for_controlnet,
    unfreeze_unet,
    ControlNetConditionedUNet,
    load_controlnet_checkpoint,
    save_controlnet_checkpoint,
    ControlNetGenerationWrapper,
)


@pytest.fixture
def mock_cfg():
    """Create minimal config for ControlNet creation."""
    return OmegaConf.create({
        'model': {
            'channels': [128, 256, 512],
            'attention_levels': [False, True, True],
            'num_res_blocks': 2,
            'num_head_channels': 64,
        },
        'controlnet': {
            'conditioning_embedding_in_channels': 1,
            'conditioning_embedding_num_channels': [16, 32, 96, 256],
            'gradient_checkpointing': False,
        },
    })


@pytest.fixture
def mock_unet():
    """Create mock UNet module."""
    unet = Mock(spec=nn.Module)
    unet.parameters.return_value = iter([torch.randn(10, 10, requires_grad=True)])

    # Make it callable to return expected shape
    def forward_fn(x, timesteps, context=None, class_labels=None,
                   down_block_additional_residuals=None, mid_block_additional_residual=None):
        return torch.randn_like(x)

    unet.side_effect = forward_fn
    unet.return_value = torch.randn(2, 4, 32, 32)
    unet.training = True
    return unet


@pytest.fixture
def mock_controlnet():
    """Create mock ControlNet module."""
    controlnet = Mock(spec=nn.Module)

    # Return down_residuals and mid_residual
    def forward_fn(x, timesteps, controlnet_cond, conditioning_scale=1.0,
                   context=None, class_labels=None):
        # Return tuple of (down_residuals, mid_residual)
        down_residuals = [torch.randn(x.shape[0], 128, x.shape[2], x.shape[3]) for _ in range(3)]
        mid_residual = torch.randn(x.shape[0], 512, x.shape[2] // 4, x.shape[3] // 4)
        return down_residuals, mid_residual

    controlnet.side_effect = forward_fn
    controlnet.parameters.return_value = iter([torch.randn(10, 10, requires_grad=True)])
    controlnet.state_dict.return_value = {'test_key': torch.randn(10)}
    controlnet.training = True
    return controlnet


class TestCreateControlNet:
    """Tests for ControlNet factory function."""

    def test_controlnet_creation_calls_monai(self, mock_cfg):
        """ControlNet creation uses MONAI ControlNet."""
        # Patch at the import location (inside the function, it imports from monai)
        with patch('monai.networks.nets.ControlNet') as MockControlNet:
            mock_instance = Mock()
            mock_instance.to.return_value = mock_instance
            mock_instance.parameters.return_value = iter([torch.randn(10)])
            MockControlNet.return_value = mock_instance

            mock_unet = Mock()
            controlnet = create_controlnet_for_unet(
                mock_unet, mock_cfg, device=torch.device('cpu'), spatial_dims=2
            )

            MockControlNet.assert_called_once()
            call_kwargs = MockControlNet.call_args[1]
            assert call_kwargs['spatial_dims'] == 2
            assert call_kwargs['in_channels'] == 4  # Default latent channels
            assert call_kwargs['channels'] == (128, 256, 512)

    def test_controlnet_uses_config_channels(self, mock_cfg):
        """ControlNet uses channels from config."""
        with patch('monai.networks.nets.ControlNet') as MockControlNet:
            mock_instance = Mock()
            mock_instance.to.return_value = mock_instance
            mock_instance.parameters.return_value = iter([torch.randn(10)])
            MockControlNet.return_value = mock_instance

            mock_unet = Mock()
            create_controlnet_for_unet(
                mock_unet, mock_cfg, device=torch.device('cpu'), spatial_dims=2
            )

            call_kwargs = MockControlNet.call_args[1]
            assert call_kwargs['channels'] == (128, 256, 512)

    def test_controlnet_conditioning_embedding_config(self, mock_cfg):
        """Conditioning embedding has correct channels."""
        with patch('monai.networks.nets.ControlNet') as MockControlNet:
            mock_instance = Mock()
            mock_instance.to.return_value = mock_instance
            mock_instance.parameters.return_value = iter([torch.randn(10)])
            MockControlNet.return_value = mock_instance

            mock_unet = Mock()
            create_controlnet_for_unet(
                mock_unet, mock_cfg, device=torch.device('cpu'), spatial_dims=2
            )

            call_kwargs = MockControlNet.call_args[1]
            assert call_kwargs['conditioning_embedding_in_channels'] == 1
            assert call_kwargs['conditioning_embedding_num_channels'] == (16, 32, 96, 256)

    def test_controlnet_3d_spatial_dims(self, mock_cfg):
        """ControlNet respects spatial_dims=3."""
        with patch('monai.networks.nets.ControlNet') as MockControlNet:
            mock_instance = Mock()
            mock_instance.to.return_value = mock_instance
            mock_instance.parameters.return_value = iter([torch.randn(10)])
            MockControlNet.return_value = mock_instance

            mock_unet = Mock()
            create_controlnet_for_unet(
                mock_unet, mock_cfg, device=torch.device('cpu'), spatial_dims=3
            )

            call_kwargs = MockControlNet.call_args[1]
            assert call_kwargs['spatial_dims'] == 3


class TestControlNetConditionedUNet:
    """Tests for the combined UNet+ControlNet wrapper."""

    def test_forward_calls_both_models(self, mock_unet, mock_controlnet):
        """Forward pass calls both ControlNet and UNet."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        x = torch.randn(2, 4, 32, 32)
        timesteps = torch.tensor([100, 200])
        cond = torch.randn(2, 1, 256, 256)

        conditioned_unet(x, timesteps, controlnet_cond=cond)

        # Both should be called
        mock_controlnet.assert_called_once()
        mock_unet.assert_called_once()

    def test_conditioning_scale_passed_to_controlnet(self, mock_unet, mock_controlnet):
        """Conditioning scale passed to ControlNet."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=0.5
        )

        x = torch.randn(2, 4, 32, 32)
        timesteps = torch.tensor([100, 200])
        cond = torch.randn(2, 1, 256, 256)

        conditioned_unet(x, timesteps, controlnet_cond=cond)

        # Check ControlNet was called with scale=0.5
        call_kwargs = mock_controlnet.call_args[1]
        assert call_kwargs['conditioning_scale'] == 0.5

    def test_conditioning_scale_override(self, mock_unet, mock_controlnet):
        """conditioning_scale parameter overrides default."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        x = torch.randn(2, 4, 32, 32)
        timesteps = torch.tensor([100, 200])
        cond = torch.randn(2, 1, 256, 256)

        # Override with 0.7
        conditioned_unet(x, timesteps, controlnet_cond=cond, conditioning_scale=0.7)

        call_kwargs = mock_controlnet.call_args[1]
        assert call_kwargs['conditioning_scale'] == 0.7

    def test_set_conditioning_scale(self, mock_unet, mock_controlnet):
        """set_conditioning_scale updates default scale."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        conditioned_unet.set_conditioning_scale(0.3)
        assert conditioned_unet.conditioning_scale == 0.3

    def test_residuals_passed_to_unet(self, mock_unet, mock_controlnet):
        """ControlNet residuals passed to UNet."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        x = torch.randn(2, 4, 32, 32)
        timesteps = torch.tensor([100, 200])
        cond = torch.randn(2, 1, 256, 256)

        conditioned_unet(x, timesteps, controlnet_cond=cond)

        # Check UNet was called with residuals
        call_kwargs = mock_unet.call_args[1]
        assert 'down_block_additional_residuals' in call_kwargs
        assert 'mid_block_additional_residual' in call_kwargs


class TestFreezeUnfreeze:
    """Tests for freeze/unfreeze UNet functions."""

    def test_freeze_unet_sets_requires_grad_false(self):
        """freeze_unet_for_controlnet sets requires_grad=False."""
        unet = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        freeze_unet_for_controlnet(unet)

        for param in unet.parameters():
            assert param.requires_grad is False

    def test_unfreeze_unet_sets_requires_grad_true(self):
        """unfreeze_unet sets requires_grad=True."""
        unet = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        freeze_unet_for_controlnet(unet)
        unfreeze_unet(unet)

        for param in unet.parameters():
            assert param.requires_grad is True

    def test_frozen_unet_no_gradients_during_backward(self):
        """Frozen UNet parameters have no gradients after backward."""
        unet = nn.Linear(10, 10)
        freeze_unet_for_controlnet(unet)

        x = torch.randn(4, 10, requires_grad=True)
        output = unet(x)
        loss = output.mean()
        loss.backward()

        # Input should have gradients
        assert x.grad is not None

        # UNet should have no gradients (frozen)
        assert unet.weight.grad is None


class TestCheckpointIO:
    """Tests for checkpoint save/load."""

    def test_save_creates_file(self, tmp_path):
        """save_controlnet_checkpoint creates file."""
        # Use real modules for checkpoint test
        controlnet = nn.Linear(10, 10)
        unet = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(controlnet.parameters())

        save_path = tmp_path / "controlnet.pt"
        save_controlnet_checkpoint(
            controlnet=controlnet,
            unet=unet,
            optimizer=optimizer,
            epoch=5,
            save_path=str(save_path)
        )

        assert save_path.exists()

    def test_save_includes_required_keys(self, tmp_path):
        """Checkpoint includes all required keys."""
        controlnet = nn.Linear(10, 10)
        unet = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(controlnet.parameters())

        save_path = tmp_path / "controlnet.pt"
        save_controlnet_checkpoint(
            controlnet=controlnet,
            unet=unet,
            optimizer=optimizer,
            epoch=5,
            save_path=str(save_path)
        )

        checkpoint = torch.load(save_path, weights_only=False)
        assert 'epoch' in checkpoint
        assert 'controlnet_state_dict' in checkpoint
        assert 'unet_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert checkpoint['epoch'] == 5

    def test_save_includes_config_when_provided(self, tmp_path, mock_cfg):
        """Config saved when provided."""
        controlnet = nn.Linear(10, 10)
        unet = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(controlnet.parameters())

        save_path = tmp_path / "controlnet.pt"
        save_controlnet_checkpoint(
            controlnet=controlnet,
            unet=unet,
            optimizer=optimizer,
            epoch=5,
            save_path=str(save_path),
            cfg=mock_cfg
        )

        checkpoint = torch.load(save_path, weights_only=False)
        assert 'config' in checkpoint

    def test_load_with_controlnet_state_dict_key(self, tmp_path):
        """Load handles controlnet_state_dict key."""
        controlnet = nn.Linear(10, 10)
        save_path = tmp_path / "controlnet.pt"

        # Save with controlnet_state_dict key
        torch.save({
            'controlnet_state_dict': controlnet.state_dict()
        }, save_path)

        # Create new controlnet and load
        new_controlnet = nn.Linear(10, 10)
        load_controlnet_checkpoint(new_controlnet, str(save_path), device=torch.device('cpu'))

        # Weights should match
        assert torch.equal(controlnet.weight, new_controlnet.weight)

    def test_load_with_state_dict_key(self, tmp_path):
        """Load handles state_dict key (legacy format)."""
        controlnet = nn.Linear(10, 10)
        save_path = tmp_path / "controlnet.pt"

        # Save with state_dict key (legacy)
        torch.save({
            'state_dict': controlnet.state_dict()
        }, save_path)

        # Create new controlnet and load
        new_controlnet = nn.Linear(10, 10)
        load_controlnet_checkpoint(new_controlnet, str(save_path), device=torch.device('cpu'))

        # Weights should match
        assert torch.equal(controlnet.weight, new_controlnet.weight)

    def test_load_raw_state_dict(self, tmp_path):
        """Load handles raw state_dict (no wrapper)."""
        controlnet = nn.Linear(10, 10)
        save_path = tmp_path / "controlnet.pt"

        # Save raw state_dict
        torch.save(controlnet.state_dict(), save_path)

        # Create new controlnet and load
        new_controlnet = nn.Linear(10, 10)
        load_controlnet_checkpoint(new_controlnet, str(save_path), device=torch.device('cpu'))

        # Weights should match
        assert torch.equal(controlnet.weight, new_controlnet.weight)


class TestControlNetGenerationWrapper:
    """Tests for ControlNetGenerationWrapper."""

    def test_wrapper_callable(self, mock_unet, mock_controlnet):
        """Wrapper is callable like a model."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        conditioning = torch.randn(2, 1, 256, 256)
        wrapper = ControlNetGenerationWrapper(conditioned_unet, conditioning)

        x = torch.randn(2, 4, 32, 32)
        timesteps = torch.tensor([100, 200])

        # Should be callable
        output = wrapper(x, timesteps)
        assert output is not None

    def test_wrapper_passes_conditioning(self, mock_unet, mock_controlnet):
        """Wrapper passes bound conditioning to model."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        conditioning = torch.randn(2, 1, 256, 256)
        wrapper = ControlNetGenerationWrapper(conditioned_unet, conditioning)

        x = torch.randn(2, 4, 32, 32)
        timesteps = torch.tensor([100, 200])

        wrapper(x, timesteps)

        # ControlNet should receive the conditioning
        call_kwargs = mock_controlnet.call_args[1]
        assert 'controlnet_cond' in call_kwargs
        assert torch.equal(call_kwargs['controlnet_cond'], conditioning)

    def test_wrapper_eval_propagates(self, mock_unet, mock_controlnet):
        """eval() propagates to underlying model."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        conditioning = torch.randn(2, 1, 256, 256)
        wrapper = ControlNetGenerationWrapper(conditioned_unet, conditioning)

        result = wrapper.eval()

        # Should return self
        assert result is wrapper
        # Model eval should be called
        conditioned_unet.eval()

    def test_wrapper_train_propagates(self, mock_unet, mock_controlnet):
        """train() propagates to underlying model."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        conditioning = torch.randn(2, 1, 256, 256)
        wrapper = ControlNetGenerationWrapper(conditioned_unet, conditioning)

        result = wrapper.train()

        # Should return self
        assert result is wrapper

    def test_wrapper_parameters(self, mock_unet, mock_controlnet):
        """parameters() returns model parameters."""
        conditioned_unet = ControlNetConditionedUNet(
            unet=mock_unet,
            controlnet=mock_controlnet,
            conditioning_scale=1.0
        )

        conditioning = torch.randn(2, 1, 256, 256)
        wrapper = ControlNetGenerationWrapper(conditioned_unet, conditioning)

        # Should return parameters from the model
        params = wrapper.parameters()
        assert params is not None


class TestGradientFlow:
    """Tests for gradient flow through ControlNet."""

    def test_controlnet_receives_gradients(self):
        """ControlNet parameters receive gradients during training."""
        # Use real small modules for gradient testing
        unet = nn.Linear(4, 4)
        controlnet = nn.Linear(4, 4)

        # Simple combined forward
        class SimpleConditionedUNet(nn.Module):
            def __init__(self, unet, controlnet):
                super().__init__()
                self.unet = unet
                self.controlnet = controlnet

            def forward(self, x, cond):
                # ControlNet modifies input
                control_out = self.controlnet(cond.mean(dim=(2, 3)))
                # UNet processes modified input
                return self.unet(x.mean(dim=(2, 3))) + control_out

        combined = SimpleConditionedUNet(unet, controlnet)

        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        cond = torch.randn(2, 4, 8, 8)

        output = combined(x, cond)
        loss = output.mean()
        loss.backward()

        # ControlNet should have gradients
        assert controlnet.weight.grad is not None
        assert controlnet.weight.grad.abs().sum() > 0

    def test_frozen_unet_controlnet_still_gets_gradients(self):
        """ControlNet receives gradients even with frozen UNet."""
        unet = nn.Linear(4, 4)
        controlnet = nn.Linear(4, 4)

        # Freeze UNet
        for param in unet.parameters():
            param.requires_grad = False

        class SimpleConditionedUNet(nn.Module):
            def __init__(self, unet, controlnet):
                super().__init__()
                self.unet = unet
                self.controlnet = controlnet

            def forward(self, x, cond):
                control_out = self.controlnet(cond.mean(dim=(2, 3)))
                return self.unet(x.mean(dim=(2, 3))) + control_out

        combined = SimpleConditionedUNet(unet, controlnet)

        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        cond = torch.randn(2, 4, 8, 8)

        output = combined(x, cond)
        loss = output.mean()
        loss.backward()

        # UNet should have no gradients (frozen)
        assert unet.weight.grad is None

        # ControlNet should still have gradients
        assert controlnet.weight.grad is not None
        assert controlnet.weight.grad.abs().sum() > 0
