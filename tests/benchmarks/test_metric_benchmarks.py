"""Performance benchmarks for metrics using pytest-benchmark.

Run with:
    pytest tests/benchmarks/test_metric_perf.py -v --benchmark-only
    pytest tests/benchmarks/test_metric_perf.py --benchmark-compare

Compare across runs:
    pytest tests/benchmarks/test_metric_perf.py --benchmark-autosave
    pytest tests/benchmarks/test_metric_perf.py --benchmark-compare
"""
import pytest

# Skip if pytest-benchmark not installed
pytest.importorskip("pytest_benchmark")


@pytest.mark.benchmark(group="quality-metrics-2d")
class TestQualityMetricPerformance2D:
    """Benchmark quality metric computation times for 2D images."""

    def test_psnr_2d(self, benchmark, benchmark_images_2d):
        """Benchmark PSNR computation on 2D images."""
        from medgen.metrics.quality import compute_psnr

        images = benchmark_images_2d
        result = benchmark(compute_psnr, images, images)
        assert result == 100.0

    def test_msssim_2d(self, benchmark, benchmark_images_2d):
        """Benchmark MS-SSIM 2D computation."""
        from medgen.metrics.quality import compute_msssim

        images = benchmark_images_2d
        result = benchmark(compute_msssim, images, images, spatial_dims=2)
        assert result > 0.99

    def test_dice_2d(self, benchmark, benchmark_masks_2d):
        """Benchmark Dice score computation."""
        from medgen.metrics.quality import compute_dice

        masks = benchmark_masks_2d
        result = benchmark(compute_dice, masks, masks, apply_sigmoid=False)
        assert result > 0.99

    def test_iou_2d(self, benchmark, benchmark_masks_2d):
        """Benchmark IoU score computation."""
        from medgen.metrics.quality import compute_iou

        masks = benchmark_masks_2d
        result = benchmark(compute_iou, masks, masks, apply_sigmoid=False)
        assert result > 0.99


@pytest.mark.benchmark(group="generation-metrics")
class TestGenerationMetricPerformance:
    """Benchmark generation metric computation times."""

    def test_kid(self, benchmark, benchmark_features):
        """Benchmark KID computation."""
        from medgen.metrics.generation import compute_kid

        features = benchmark_features
        result = benchmark(compute_kid, features, features)
        # Returns (mean, std) tuple
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.timeout(300)  # FID uses O(n³) matrix sqrtm; ~22s/iter × 4+ iters
    def test_fid(self, benchmark, benchmark_features):
        """Benchmark FID computation."""
        from medgen.metrics.generation import compute_fid

        features = benchmark_features
        result = benchmark(compute_fid, features, features)
        assert isinstance(result, float)
        assert result < 1.0  # Identical distributions

    def test_cmmd(self, benchmark, benchmark_clip_features):
        """Benchmark CMMD computation."""
        from medgen.metrics.generation import compute_cmmd

        features = benchmark_clip_features
        result = benchmark(compute_cmmd, features, features)
        assert isinstance(result, float)
        assert result < 0.1  # Identical distributions


@pytest.mark.benchmark(group="diversity-metrics")
class TestDiversityMetricPerformance:
    """Benchmark diversity metric computation times."""

    def test_msssim_diversity(self, benchmark, benchmark_images_2d):
        """Benchmark MS-SSIM diversity computation."""
        from medgen.metrics.quality import compute_msssim_diversity

        images = benchmark_images_2d
        result = benchmark(compute_msssim_diversity, images)
        assert isinstance(result, float)
        assert result >= 0.0
