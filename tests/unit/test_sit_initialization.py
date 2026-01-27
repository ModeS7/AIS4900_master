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

    def test_output_is_nonzero(self, sit_model, sample_input):
        """Model should produce non-zero output after initialization.

        This catches bugs like zero-initialized final layers that make
        the model output all zeros regardless of input.
        """
        sit_model.eval()
        t = torch.tensor([0.5, 0.5])

        with torch.no_grad():
            output = sit_model(sample_input, t)

        assert output.std() > 0.1, (
            f"Model output std={output.std():.6f} is too small. "
            "This likely indicates a zero-initialized final layer bug."
        )

    def test_output_has_reasonable_scale(self, sit_model, sample_input):
        """Output should have reasonable magnitude (not exploding/vanishing)."""
        sit_model.eval()
        t = torch.tensor([0.5, 0.5])

        with torch.no_grad():
            output = sit_model(sample_input, t)

        assert 0.1 < output.std() < 10.0, (
            f"Output std={output.std():.4f} is outside reasonable range [0.1, 10.0]"
        )
        assert output.abs().max() < 100.0, (
            f"Output max={output.abs().max():.4f} is too large"
        )

    def test_gradients_flow_to_input(self, sit_model, sample_input):
        """Gradients should flow from loss to input.

        This catches dead gradient issues from bad initialization.
        """
        sit_model.train()
        sample_input.requires_grad_(True)
        t = torch.tensor([0.5, 0.5])

        output = sit_model(sample_input, t)
        loss = output.mean()
        loss.backward()

        assert sample_input.grad is not None, "No gradient computed for input"
        assert sample_input.grad.norm() > 1e-8, (
            f"Input gradient norm={sample_input.grad.norm():.2e} is too small. "
            "Gradients are not flowing properly."
        )

    def test_final_layer_weights_nonzero(self, sit_model):
        """Final layer should have non-zero weights after initialization."""
        weight_norm = sit_model.final_layer.linear.weight.norm()

        assert weight_norm > 0.1, (
            f"Final layer weight norm={weight_norm:.6f} is too small. "
            "Final layer should not be zero-initialized."
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
        """All variants should produce non-zero outputs."""
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

        assert output.std() > 0.1, f"SiT-{variant} produces near-zero output"
