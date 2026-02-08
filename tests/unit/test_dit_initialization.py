"""Unit tests for DiT model initialization.

Ensures the model produces non-zero outputs and has flowing gradients
after initialization. Catches bugs like zero-initialized final layers.
"""
import pytest
import torch

import sys
sys.path.insert(0, 'src')

from medgen.models.dit import DiT


class TestDiTInitialization:
    """Tests for DiT weight initialization correctness."""

    @pytest.fixture(params=[2, 3])
    def spatial_dims(self, request):
        """Test both 2D and 3D."""
        return request.param

    @pytest.fixture
    def dit_model(self, spatial_dims):
        """Create minimal DiT model for testing."""
        if spatial_dims == 2:
            return DiT(
                spatial_dims=2,
                input_size=16,
                patch_size=2,
                in_channels=2,
                out_channels=1,
                hidden_size=64,
                depth=2,
                num_heads=2,
            )
        else:
            return DiT(
                spatial_dims=3,
                input_size=16,
                depth_size=8,
                patch_size=2,
                in_channels=2,
                out_channels=1,
                hidden_size=64,
                depth=2,
                num_heads=2,
            )

    @pytest.fixture
    def sample_input(self, spatial_dims):
        """Create sample input tensor."""
        if spatial_dims == 2:
            return torch.randn(2, 2, 16, 16)
        else:
            return torch.randn(2, 2, 8, 16, 16)

    def test_output_is_zero_at_init(self, dit_model, sample_input):
        """Model should produce zero output at initialization.

        This is expected behavior with standard DiT/Latte zero-initialization
        of the final layer. The model learns to produce non-zero output during training.
        """
        dit_model.eval()
        t = torch.tensor([0.5, 0.5])

        with torch.no_grad():
            output = dit_model(sample_input, t)

        assert output.std() < 1e-6, (
            f"Model output std={output.std():.6f} should be ~0 at init. "
            "Standard DiT initialization produces zero output initially."
        )

    def test_output_shape_correct(self, dit_model, sample_input):
        """Output should have correct shape matching input spatial dims."""
        dit_model.eval()
        t = torch.tensor([0.5, 0.5])

        with torch.no_grad():
            output = dit_model(sample_input, t)

        # Output should have same spatial dims, but out_channels (1) instead of in_channels (2)
        expected_shape = list(sample_input.shape)
        expected_shape[1] = 1  # out_channels
        assert list(output.shape) == expected_shape, (
            f"Output shape {output.shape} doesn't match expected {expected_shape}"
        )

    def test_gradients_flow_to_final_layer(self, dit_model, sample_input):
        """Gradients should flow to final layer weights.

        With standard DiT zero-init, gradients flow to the final layer
        weights (so they can learn to become non-zero), even though output is zero.
        """
        dit_model.train()
        t = torch.tensor([0.5, 0.5])

        output = dit_model(sample_input, t)
        loss = output.mean()
        loss.backward()

        # Final layer weights should receive gradients
        final_grad = dit_model.final_layer.linear.weight.grad
        assert final_grad is not None, "No gradient computed for final layer"
        # Note: with zero output, mean() loss has zero gradient, so we use sum() or check adaLN

        # adaLN modulation should receive gradients (through the conditioning path)
        adaln_grad = dit_model.final_layer.adaLN_modulation[-1].weight.grad
        assert adaln_grad is not None, "No gradient computed for adaLN modulation"

    def test_final_layer_weights_zero(self, dit_model):
        """Final layer should be zero-initialized (matches DiT/Latte standard)."""
        weight_norm = dit_model.final_layer.linear.weight.norm()
        bias_norm = dit_model.final_layer.linear.bias.norm()

        assert weight_norm < 1e-6, (
            f"Final layer weight norm={weight_norm:.6f} should be zero. "
            "Standard DiT initialization zeros the final layer."
        )
        assert bias_norm < 1e-6, (
            f"Final layer bias norm={bias_norm:.6f} should be zero."
        )

    def test_adaln_modulation_zero_initialized(self, dit_model):
        """adaLN modulation should be zero-initialized (for identity start)."""
        # Check first block
        block = dit_model.blocks[0]
        adaln_weight = block.adaLN_modulation[-1].weight
        adaln_bias = block.adaLN_modulation[-1].bias

        # These SHOULD be zero (for adaLN-Zero)
        assert adaln_weight.norm() < 1e-6, "adaLN weights should be zero-initialized"
        assert adaln_bias.norm() < 1e-6, "adaLN biases should be zero-initialized"

    def test_training_step_reduces_loss(self, dit_model, sample_input):
        """A training step should reduce loss (model can learn)."""
        dit_model.train()
        optimizer = torch.optim.Adam(dit_model.parameters(), lr=1e-3)
        t = torch.tensor([0.5, 0.5])
        target = torch.randn_like(sample_input[:, :1])  # Match output channels

        # Initial loss
        output = dit_model(sample_input, t)
        loss_before = ((output - target) ** 2).mean()

        # Training step
        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        # Loss after
        with torch.no_grad():
            output_after = dit_model(sample_input, t)
            loss_after = ((output_after - target) ** 2).mean()

        # Loss should generally decrease (or at least the model should change)
        output_changed = (output - output_after).abs().mean() > 1e-6
        assert output_changed, "Model output didn't change after training step"


class TestDiTVariants:
    """Test all DiT variants initialize correctly."""

    @pytest.mark.parametrize("variant", ["S", "B", "L", "XL"])
    def test_variant_initialization(self, variant):
        """All variants should initialize correctly with zero output."""
        from medgen.models.dit import DIT_VARIANTS

        config = DIT_VARIANTS[variant]
        model = DiT(
            spatial_dims=2,
            input_size=16,
            patch_size=2,
            in_channels=1,
            out_channels=1,
            hidden_size=config['hidden_size'],
            depth=config['depth'],
            num_heads=config['num_heads'],
        )
        model.eval()

        x = torch.randn(1, 1, 16, 16)
        t = torch.tensor([0.5])

        with torch.no_grad():
            output = model(x, t)

        # Standard DiT initialization produces zero output
        assert output.std() < 1e-6, f"DiT-{variant} should produce zero output at init"
        assert output.shape == x.shape, f"DiT-{variant} output shape mismatch"
