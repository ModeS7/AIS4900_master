"""Edge case tests for robustness verification.

Tests behavior at boundaries: empty inputs, NaN values, device mismatches,
and non-standard tensor configurations.
"""
import math
import pytest
import torch


class TestEmptyBatchHandling:
    """Verify graceful handling of B=0 tensors."""

    def test_psnr_empty_batch_raises_or_handles(self):
        """PSNR with empty batch should raise or return valid value."""
        from medgen.metrics.quality import compute_psnr

        images = torch.rand(0, 1, 64, 64)
        # Either raises ValueError/RuntimeError or returns valid float
        try:
            result = compute_psnr(images, images)
            assert isinstance(result, float)
        except (ValueError, RuntimeError):
            pass  # Acceptable behavior

    def test_msssim_empty_batch_raises_or_handles(self):
        """MS-SSIM with empty batch should raise or return valid value."""
        from medgen.metrics.quality import compute_msssim

        images = torch.rand(0, 1, 64, 64)
        try:
            result = compute_msssim(images, images, spatial_dims=2)
            assert isinstance(result, float)
        except (ValueError, RuntimeError):
            pass


class TestNaNInputHandling:
    """Verify NaN inputs don't cause silent failures."""

    def test_psnr_nan_input_propagates(self):
        """PSNR with NaN should return NaN or 100 (self-comparison)."""
        from medgen.metrics.quality import compute_psnr

        images = torch.tensor([[[[float('nan')]]]])
        result = compute_psnr(images, images)
        # Self-comparison may clamp to 100, or NaN propagates
        assert result == 100.0 or math.isnan(result)

    def test_dice_nan_input_handled(self):
        """Dice with NaN should not crash."""
        from medgen.metrics.quality import compute_dice

        pred = torch.tensor([[[[float('nan')]]]])
        target = torch.ones(1, 1, 1, 1)
        try:
            result = compute_dice(pred, target, apply_sigmoid=False)
            assert isinstance(result, float)
        except (ValueError, RuntimeError):
            pass


class TestDeviceMismatch:
    """Verify device mismatches are caught."""

    @pytest.mark.gpu
    def test_psnr_device_mismatch_raises(self):
        """PSNR with CPU vs CUDA should raise RuntimeError."""
        from medgen.metrics.quality import compute_psnr

        cpu_tensor = torch.rand(2, 1, 64, 64)
        cuda_tensor = torch.rand(2, 1, 64, 64).cuda()

        with pytest.raises(RuntimeError):
            compute_psnr(cpu_tensor, cuda_tensor)

    @pytest.mark.gpu
    def test_msssim_device_mismatch_handles_gracefully(self):
        """MS-SSIM with CPU vs CUDA handles error gracefully.

        The implementation catches device mismatches and returns 0.0 with a warning.
        """
        from medgen.metrics.quality import compute_msssim

        cpu_tensor = torch.rand(2, 1, 64, 64)
        cuda_tensor = torch.rand(2, 1, 64, 64).cuda()

        # Should either raise or return 0.0 (fallback value on error)
        try:
            result = compute_msssim(cpu_tensor, cuda_tensor, spatial_dims=2)
            # Implementation returns 0.0 on error (logged as warning)
            assert result == 0.0 or math.isnan(result), "Device mismatch should return 0.0 or NaN"
        except RuntimeError:
            pass  # Also acceptable behavior


class TestNonContiguousTensors:
    """Verify non-contiguous tensors work correctly."""

    def test_psnr_non_contiguous_works(self):
        """PSNR should handle non-contiguous tensors."""
        from medgen.metrics.quality import compute_psnr

        images = torch.rand(4, 1, 64, 64)[:, :, ::2, ::2]  # Non-contiguous
        assert not images.is_contiguous()

        result = compute_psnr(images, images)
        assert result == 100.0

    def test_msssim_non_contiguous_works(self):
        """MS-SSIM should handle non-contiguous tensors."""
        from medgen.metrics.quality import compute_msssim

        images = torch.rand(4, 1, 64, 64)[:, :, ::2, ::2]
        assert not images.is_contiguous()

        result = compute_msssim(images, images, spatial_dims=2)
        assert result > 0.99

    def test_dice_non_contiguous_works(self):
        """Dice should handle non-contiguous tensors."""
        from medgen.metrics.quality import compute_dice

        masks = (torch.rand(4, 1, 64, 64) > 0.5).float()[:, :, ::2, ::2]
        assert not masks.is_contiguous()

        result = compute_dice(masks, masks, apply_sigmoid=False)
        assert result > 0.99


class TestSingleElementTensors:
    """Verify single-element tensor edge cases."""

    def test_psnr_single_pixel(self):
        """PSNR with 1x1 images should work."""
        from medgen.metrics.quality import compute_psnr

        images = torch.rand(2, 1, 1, 1)
        result = compute_psnr(images, images)
        assert result == 100.0

    def test_dice_single_pixel(self):
        """Dice with 1x1 masks should work."""
        from medgen.metrics.quality import compute_dice

        masks = torch.ones(2, 1, 1, 1)
        result = compute_dice(masks, masks, apply_sigmoid=False)
        assert result > 0.99


class TestInfInputHandling:
    """Verify Inf inputs are handled appropriately."""

    def test_psnr_inf_input(self):
        """PSNR with Inf should not crash."""
        from medgen.metrics.quality import compute_psnr

        images = torch.tensor([[[[float('inf')]]]])
        reference = torch.ones(1, 1, 1, 1)
        try:
            result = compute_psnr(images, reference)
            # After clamping inf->1, should be identical to reference
            assert isinstance(result, float)
        except (ValueError, RuntimeError):
            pass  # Also acceptable

    def test_dice_inf_input(self):
        """Dice with Inf should not crash."""
        from medgen.metrics.quality import compute_dice

        pred = torch.tensor([[[[float('inf')]]]])
        target = torch.ones(1, 1, 1, 1)
        try:
            result = compute_dice(pred, target, apply_sigmoid=False)
            assert isinstance(result, float)
        except (ValueError, RuntimeError):
            pass
