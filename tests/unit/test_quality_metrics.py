"""Tests for quality metrics: PSNR, MS-SSIM (2D/3D), LPIPS (2D/3D), Dice, IoU."""

import pytest
import torch

from medgen.metrics.quality import (
    compute_psnr,
    compute_msssim,
    compute_msssim_2d_slicewise,
    compute_lpips,
    compute_lpips_3d,
    compute_dice,
    compute_iou,
    clear_metric_caches,
    reset_msssim_nan_warning,
    reset_lpips_nan_warning,
)


class TestPSNR:
    """Test compute_psnr function."""

    def test_identical_images_returns_100(self):
        """Perfect match should return 100 dB."""
        images = torch.rand(4, 1, 64, 64)
        psnr = compute_psnr(images, images)
        assert psnr == 100.0

    def test_completely_different_returns_low_psnr(self):
        """Very different images should have low PSNR."""
        zeros = torch.zeros(4, 1, 64, 64)
        ones = torch.ones(4, 1, 64, 64)
        psnr = compute_psnr(zeros, ones)
        assert psnr < 10.0  # Very low PSNR expected

    @pytest.mark.parametrize("shape,desc", [
        ((2, 3, 32, 32), "2D [B, C, H, W]"),
        ((2, 1, 16, 32, 32), "3D [B, C, D, H, W]"),
    ])
    def test_tensor_shape(self, shape, desc):
        """Works with both 2D and 3D tensors."""
        images = torch.rand(*shape)
        reference = torch.rand(*shape)
        psnr = compute_psnr(images, reference)
        assert isinstance(psnr, float)
        assert not torch.isnan(torch.tensor(psnr)), f"PSNR is NaN for {desc}"
        assert not torch.isinf(torch.tensor(psnr)), f"PSNR is inf for {desc}"
        assert 0.0 < psnr < 100.0

    def test_clamps_input_to_data_range(self):
        """Values outside [0, data_range] are clamped."""
        images = torch.tensor([[[[-0.5, 1.5]]]])
        reference = torch.tensor([[[[0.0, 1.0]]]])
        # After clamping: [-0.5, 1.5] -> [0, 1], comparing to [0, 1]
        psnr = compute_psnr(images, reference)
        assert psnr == 100.0  # After clamping they should be identical

    def test_custom_data_range(self):
        """Custom data_range parameter works."""
        # Images in [0, 255] range
        images = torch.randint(0, 256, (2, 1, 32, 32)).float()
        reference = images.clone()
        psnr = compute_psnr(images, reference, data_range=255.0)
        assert psnr == 100.0

    def test_single_batch(self):
        """Works with batch_size=1."""
        images = torch.rand(1, 1, 64, 64)
        reference = torch.rand(1, 1, 64, 64)
        psnr = compute_psnr(images, reference)
        assert isinstance(psnr, float)
        assert not torch.isnan(torch.tensor(psnr))
        assert 0.0 < psnr <= 100.0

    def test_psnr_returns_float(self):
        """Returns Python float, not tensor."""
        images = torch.rand(2, 1, 32, 32)
        psnr = compute_psnr(images, images)
        assert isinstance(psnr, float)


class TestMSSSIM:
    """Test compute_msssim function."""

    def test_identical_images_returns_near_1(self):
        """Perfect match should return ~1.0."""
        images = torch.rand(4, 1, 64, 64)
        msssim = compute_msssim(images, images, spatial_dims=2)
        assert msssim > 0.99

    def test_completely_different_returns_low(self):
        """Very different images have low MS-SSIM."""
        zeros = torch.zeros(4, 1, 64, 64)
        ones = torch.ones(4, 1, 64, 64)
        msssim = compute_msssim(zeros, ones, spatial_dims=2)
        assert msssim < 0.5

    @pytest.mark.parametrize("shape,spatial_dims,desc", [
        ((2, 1, 64, 64), 2, "2D [B, C, H, W]"),
        ((2, 1, 24, 64, 64), 3, "3D [B, C, D, H, W]"),
    ])
    def test_spatial_dims(self, shape, spatial_dims, desc):
        """Works with both 2D and 3D tensors."""
        images = torch.rand(*shape)
        reference = torch.rand(*shape)
        msssim = compute_msssim(images, reference, spatial_dims=spatial_dims)
        assert isinstance(msssim, float)
        assert not torch.isnan(torch.tensor(msssim)), f"MS-SSIM is NaN for {desc}"
        assert not torch.isinf(torch.tensor(msssim)), f"MS-SSIM is inf for {desc}"
        assert 0.0 <= msssim <= 1.0

    def test_small_image_uses_fewer_scales(self):
        """Images < 177px use fewer MS-SSIM scales (no crash)."""
        images = torch.rand(2, 1, 48, 48)  # < 89 pixels, should use 3 scales
        reference = torch.rand(2, 1, 48, 48)
        msssim = compute_msssim(images, reference, spatial_dims=2)
        assert isinstance(msssim, float)
        assert not torch.isnan(torch.tensor(msssim))
        assert 0.0 <= msssim <= 1.0

    def test_very_small_image_uses_2_scales(self):
        """Images < 45px use only 2 scales."""
        images = torch.rand(2, 1, 32, 32)  # < 45 pixels, should use 2 scales
        reference = torch.rand(2, 1, 32, 32)
        msssim = compute_msssim(images, reference, spatial_dims=2)
        assert isinstance(msssim, float)
        assert not torch.isnan(torch.tensor(msssim))
        assert 0.0 <= msssim <= 1.0

    def test_msssim_multichannel_input(self):
        """Works with C > 1 channels."""
        images = torch.rand(2, 3, 64, 64)
        reference = torch.rand(2, 3, 64, 64)
        msssim = compute_msssim(images, reference, spatial_dims=2)
        assert isinstance(msssim, float)
        assert not torch.isnan(torch.tensor(msssim))
        assert 0.0 <= msssim <= 1.0

    def test_msssim_returns_float(self):
        """Returns Python float, not tensor."""
        images = torch.rand(2, 1, 64, 64)
        msssim = compute_msssim(images, images, spatial_dims=2)
        assert isinstance(msssim, float)
        assert not torch.isnan(torch.tensor(msssim))


class TestMSSSIM2DSlicewise:
    """Test compute_msssim_2d_slicewise for 3D volumes."""

    def test_identical_volumes_returns_near_1(self):
        """Perfect match across all slices."""
        volumes = torch.rand(2, 1, 16, 64, 64)
        msssim = compute_msssim_2d_slicewise(volumes, volumes)
        assert msssim > 0.99

    def test_different_volumes(self):
        """Different volumes have lower MS-SSIM."""
        vol1 = torch.rand(2, 1, 16, 64, 64)
        vol2 = torch.rand(2, 1, 16, 64, 64)
        msssim = compute_msssim_2d_slicewise(vol1, vol2)
        assert 0.0 <= msssim < 1.0

    def test_output_is_scalar(self):
        """Returns single float, not tensor."""
        volumes = torch.rand(2, 1, 8, 64, 64)
        msssim = compute_msssim_2d_slicewise(volumes, volumes)
        assert isinstance(msssim, float)

    def test_handles_single_batch(self):
        """Works with B=1."""
        volumes = torch.rand(1, 1, 16, 64, 64)
        msssim = compute_msssim_2d_slicewise(volumes, volumes)
        assert msssim > 0.99


@pytest.mark.usefixtures("lpips_available")
class TestLPIPS:
    """Test compute_lpips function (2D only)."""

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_identical_images_returns_near_zero(self):
        """Perfect match should return ~0."""
        images = torch.rand(4, 1, 64, 64)
        lpips = compute_lpips(images, images)
        assert lpips < 0.01

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_different_images_returns_positive(self):
        """Different images have positive LPIPS."""
        img1 = torch.rand(4, 1, 64, 64)
        img2 = torch.rand(4, 1, 64, 64)
        lpips = compute_lpips(img1, img2)
        assert lpips > 0.0

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_single_channel_input(self):
        """Grayscale (1 channel) is repeated to 3 channels."""
        images = torch.rand(2, 1, 64, 64)
        reference = torch.rand(2, 1, 64, 64)
        lpips = compute_lpips(images, reference)
        assert isinstance(lpips, float)
        assert not torch.isnan(torch.tensor(lpips))
        assert lpips >= 0.0  # LPIPS is non-negative distance

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_dual_channel_input(self):
        """2-channel input computes per-channel average."""
        images = torch.rand(2, 2, 64, 64)
        reference = torch.rand(2, 2, 64, 64)
        lpips = compute_lpips(images, reference)
        assert isinstance(lpips, float)
        assert not torch.isnan(torch.tensor(lpips))
        assert lpips >= 0.0

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_three_channel_input(self):
        """3-channel input used directly."""
        images = torch.rand(2, 3, 64, 64)
        reference = torch.rand(2, 3, 64, 64)
        lpips = compute_lpips(images, reference)
        assert isinstance(lpips, float)
        assert not torch.isnan(torch.tensor(lpips))
        assert lpips >= 0.0

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_lpips_multichannel_input(self):
        """C > 3 channels averaged per-channel."""
        images = torch.rand(2, 4, 64, 64)
        reference = torch.rand(2, 4, 64, 64)
        lpips = compute_lpips(images, reference)
        assert isinstance(lpips, float)
        assert not torch.isnan(torch.tensor(lpips))
        assert lpips >= 0.0

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_lpips_returns_float(self):
        """Returns Python float, not tensor."""
        images = torch.rand(2, 1, 64, 64)
        lpips = compute_lpips(images, images)
        assert isinstance(lpips, float)
        assert not torch.isnan(torch.tensor(lpips))


@pytest.mark.usefixtures("lpips_available")
class TestLPIPS3D:
    """Test compute_lpips_3d for 3D volumes."""

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_identical_volumes_returns_near_zero(self):
        """Perfect match should return ~0."""
        volumes = torch.rand(2, 1, 8, 64, 64)
        lpips = compute_lpips_3d(volumes, volumes)
        assert lpips < 0.01

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_different_volumes_returns_positive(self):
        """Different volumes have positive LPIPS."""
        vol1 = torch.rand(2, 1, 8, 64, 64)
        vol2 = torch.rand(2, 1, 8, 64, 64)
        lpips = compute_lpips_3d(vol1, vol2)
        assert lpips > 0.0

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_chunk_size_parameter(self):
        """chunk_size controls batch size."""
        volumes = torch.rand(2, 1, 16, 64, 64)
        # Should work with different chunk sizes
        lpips1 = compute_lpips_3d(volumes, volumes, chunk_size=4)
        lpips2 = compute_lpips_3d(volumes, volumes, chunk_size=8)
        # Both should be near zero for identical inputs
        assert lpips1 < 0.01
        assert lpips2 < 0.01

    @pytest.mark.timeout(30)
    @pytest.mark.slow
    def test_lpips_3d_returns_scalar(self):
        """Returns single float."""
        volumes = torch.rand(2, 1, 8, 64, 64)
        lpips = compute_lpips_3d(volumes, volumes)
        assert isinstance(lpips, float)
        assert not torch.isnan(torch.tensor(lpips))
        assert lpips >= 0.0


class TestDiceScore:
    """Test compute_dice function."""

    def test_perfect_overlap_returns_1(self):
        """Identical masks return 1.0."""
        mask = torch.ones(4, 1, 64, 64)
        dice = compute_dice(mask, mask, apply_sigmoid=False)
        assert abs(dice - 1.0) < 0.01

    def test_no_overlap_returns_near_zero(self):
        """Non-overlapping masks return ~0 (smooth prevents exact 0)."""
        pred = torch.zeros(4, 1, 64, 64)
        target = torch.ones(4, 1, 64, 64)
        dice = compute_dice(pred, target, apply_sigmoid=False)
        assert dice < 0.1

    def test_empty_masks_returns_1(self):
        """Empty pred and target: numerator and denominator both small."""
        pred = torch.zeros(4, 1, 64, 64)
        target = torch.zeros(4, 1, 64, 64)
        dice = compute_dice(pred, target, apply_sigmoid=False)
        # With smooth=1.0: (0 + 1) / (0 + 0 + 1) = 1.0
        assert dice > 0.9

    def test_sigmoid_applied_to_logits(self):
        """apply_sigmoid=True converts logits to probs."""
        # Logits: large positive = near 1 after sigmoid
        logits = torch.full((4, 1, 64, 64), 10.0)
        target = torch.ones(4, 1, 64, 64)
        dice = compute_dice(logits, target, apply_sigmoid=True)
        assert dice > 0.9

    def test_sigmoid_skipped_for_probs(self):
        """apply_sigmoid=False for [0,1] inputs."""
        probs = torch.ones(4, 1, 64, 64) * 0.8
        target = torch.ones(4, 1, 64, 64)
        dice = compute_dice(probs, target, apply_sigmoid=False)
        assert dice > 0.8

    def test_threshold_parameter(self):
        """Custom threshold for binarization."""
        probs = torch.ones(4, 1, 64, 64) * 0.3  # Below default 0.5
        target = torch.ones(4, 1, 64, 64)

        dice_default = compute_dice(probs, target, threshold=0.5, apply_sigmoid=False)
        dice_low_thresh = compute_dice(probs, target, threshold=0.2, apply_sigmoid=False)

        # With threshold=0.2, 0.3 > 0.2 so all are 1s -> high dice
        # With threshold=0.5, 0.3 < 0.5 so all are 0s -> low dice
        assert dice_low_thresh > dice_default

    @pytest.mark.parametrize("shape,desc", [
        ((4, 1, 64, 64), "2D [B, C, H, W]"),
        ((2, 1, 16, 32, 32), "3D [B, C, D, H, W]"),
    ])
    def test_volumes_shapes(self, shape, desc):
        """Works with both 2D and 3D tensors."""
        pred = torch.ones(*shape)
        target = torch.ones(*shape)
        dice = compute_dice(pred, target, apply_sigmoid=False)
        assert isinstance(dice, float)
        assert not torch.isnan(torch.tensor(dice)), f"Dice is NaN for {desc}"
        assert dice > 0.99

    def test_dice_returns_float(self):
        """Returns Python float, not tensor."""
        mask = torch.ones(2, 1, 32, 32)
        dice = compute_dice(mask, mask, apply_sigmoid=False)
        assert isinstance(dice, float)
        assert not torch.isnan(torch.tensor(dice))
        assert 0.0 <= dice <= 1.0


class TestIoU:
    """Test compute_iou function."""

    def test_perfect_overlap_returns_1(self):
        """Identical masks return 1.0."""
        mask = torch.ones(4, 1, 64, 64)
        iou = compute_iou(mask, mask, apply_sigmoid=False)
        assert abs(iou - 1.0) < 0.01

    def test_no_overlap_returns_near_zero(self):
        """Non-overlapping masks return ~0."""
        pred = torch.zeros(4, 1, 64, 64)
        target = torch.ones(4, 1, 64, 64)
        iou = compute_iou(pred, target, apply_sigmoid=False)
        assert iou < 0.1

    def test_formula_differs_from_dice(self):
        """IoU <= Dice for partial overlap."""
        # Create 50% overlap
        pred = torch.zeros(1, 1, 64, 64)
        pred[:, :, :32, :] = 1.0
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 16:48, :] = 1.0

        dice = compute_dice(pred, target, apply_sigmoid=False)
        iou = compute_iou(pred, target, apply_sigmoid=False)

        # For partial overlap: IoU < Dice
        assert iou <= dice

    @pytest.mark.parametrize("shape,desc", [
        ((4, 1, 64, 64), "2D [B, C, H, W]"),
        ((2, 1, 16, 32, 32), "3D [B, C, D, H, W]"),
    ])
    def test_volumes_shapes(self, shape, desc):
        """Works with both 2D and 3D tensors."""
        pred = torch.ones(*shape)
        target = torch.ones(*shape)
        iou = compute_iou(pred, target, apply_sigmoid=False)
        assert isinstance(iou, float)
        assert not torch.isnan(torch.tensor(iou)), f"IoU is NaN for {desc}"
        assert iou > 0.99

    def test_iou_returns_float(self):
        """Returns Python float, not tensor."""
        mask = torch.ones(2, 1, 32, 32)
        iou = compute_iou(mask, mask, apply_sigmoid=False)
        assert isinstance(iou, float)
        assert not torch.isnan(torch.tensor(iou))
        assert 0.0 <= iou <= 1.0


class TestCacheClear:
    """Test clear_metric_caches and reset functions."""

    def test_clear_metric_caches_runs(self):
        """clear_metric_caches executes without error."""
        # Just verify it doesn't crash
        clear_metric_caches()

    def test_reset_nan_warnings_run(self):
        """Reset warning flags execute without error."""
        reset_msssim_nan_warning()
        reset_lpips_nan_warning()
