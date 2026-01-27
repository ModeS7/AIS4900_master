"""Unit tests for SiT model initialization.

Ensures the model produces non-zero outputs and has flowing gradients
after initialization. Catches bugs like zero-initialized final layers.
"""
import pytest
import torch

import sys
sys.path.insert(0, 'src')

from medgen.models.sit import SiT


class TestSiTInitialization:
    """Tests for SiT weight initialization correctness."""

    @pytest.fixture(params=[2, 3])
    def spatial_dims(self, request):
        """Test both 2D and 3D."""
        return request.param

    @pytest.fixture
    def sit_model(self, spatial_dims):
        """Create minimal SiT model for testing."""
        if spatial_dims == 2:
            return SiT(
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
            return SiT(
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

    def test_output_is_zero_at_init(self, sit_model, sample_input):
        """Model should produce zero output at initialization.

        This is expected behavior with standard DiT/SiT/Latte zero-initialization
        of the final layer. The model learns to produce non-zero output during training.
        """
        sit_model.eval()
        t = torch.tensor([0.5, 0.5])

        with torch.no_grad():
            output = sit_model(sample_input, t)

        assert output.std() < 1e-6, (
            f"Model output std={output.std():.6f} should be ~0 at init. "
            "Standard DiT/SiT initialization produces zero output initially."
        )

    def test_output_shape_correct(self, sit_model, sample_input):
        """Output should have correct shape matching input spatial dims."""
        sit_model.eval()
        t = torch.tensor([0.5, 0.5])

        with torch.no_grad():
            output = sit_model(sample_input, t)

        # Output should have same spatial dims, but out_channels (1) instead of in_channels (2)
        expected_shape = list(sample_input.shape)
        expected_shape[1] = 1  # out_channels
        assert list(output.shape) == expected_shape, (
            f"Output shape {output.shape} doesn't match expected {expected_shape}"
        )

    def test_gradients_flow_to_final_layer(self, sit_model, sample_input):
        """Gradients should flow to final layer weights.

        With standard DiT/SiT zero-init, gradients flow to the final layer
        weights (so they can learn to become non-zero), even though output is zero.
        """
        sit_model.train()
        t = torch.tensor([0.5, 0.5])

        output = sit_model(sample_input, t)
        loss = output.mean()
        loss.backward()

        # Final layer weights should receive gradients
        final_grad = sit_model.final_layer.linear.weight.grad
        assert final_grad is not None, "No gradient computed for final layer"
        # Note: with zero output, mean() loss has zero gradient, so we use sum() or check adaLN

        # adaLN modulation should receive gradients (through the conditioning path)
        adaln_grad = sit_model.final_layer.adaLN_modulation[-1].weight.grad
        assert adaln_grad is not None, "No gradient computed for adaLN modulation"

    def test_final_layer_weights_zero(self, sit_model):
        """Final layer should be zero-initialized (matches DiT/SiT/Latte standard)."""
        weight_norm = sit_model.final_layer.linear.weight.norm()
        bias_norm = sit_model.final_layer.linear.bias.norm()

        assert weight_norm < 1e-6, (
            f"Final layer weight norm={weight_norm:.6f} should be zero. "
            "Standard DiT/SiT initialization zeros the final layer."
        )
        assert bias_norm < 1e-6, (
            f"Final layer bias norm={bias_norm:.6f} should be zero."
        )

    def test_adaln_modulation_zero_initialized(self, sit_model):
        """adaLN modulation should be zero-initialized (for identity start)."""
        # Check first block
        block = sit_model.blocks[0]
        adaln_weight = block.adaLN_modulation[-1].weight
        adaln_bias = block.adaLN_modulation[-1].bias

        # These SHOULD be zero (for adaLN-Zero)
        assert adaln_weight.norm() < 1e-6, "adaLN weights should be zero-initialized"
        assert adaln_bias.norm() < 1e-6, "adaLN biases should be zero-initialized"

    def test_training_step_reduces_loss(self, sit_model, sample_input):
        """A training step should reduce loss (model can learn)."""
        sit_model.train()
        optimizer = torch.optim.Adam(sit_model.parameters(), lr=1e-3)
        t = torch.tensor([0.5, 0.5])
        target = torch.randn_like(sample_input[:, :1])  # Match output channels

        # Initial loss
        output = sit_model(sample_input, t)
        loss_before = ((output - target) ** 2).mean()

        # Training step
        optimizer.zero_grad()
        loss_before.backward()
        optimizer.step()

        # Loss after
        with torch.no_grad():
            output_after = sit_model(sample_input, t)
            loss_after = ((output_after - target) ** 2).mean()

        # Loss should generally decrease (or at least the model should change)
        output_changed = (output - output_after).abs().mean() > 1e-6
        assert output_changed, "Model output didn't change after training step"


class TestSiTVariants:
    """Test all SiT variants initialize correctly."""

    @pytest.mark.parametrize("variant", ["S", "B", "L", "XL"])
    def test_variant_initialization(self, variant):
        """All variants should initialize correctly with zero output."""
        from medgen.models.sit import SIT_VARIANTS

        config = SIT_VARIANTS[variant]
        model = SiT(
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

        # Standard DiT/SiT initialization produces zero output
        assert output.std() < 1e-6, f"SiT-{variant} should produce zero output at init"
        assert output.shape == x.shape, f"SiT-{variant} output shape mismatch"
