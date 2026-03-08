"""Tests for tri-planar generation metrics (Generation_3d/ TensorBoard section).

Tests cover:
- extract_features_3d_triplanar: shape, padding removal, content correctness
- _extract_chunks: shared helper for chunked feature extraction
- StreamingFeatures: NamedTuple fields and construction
- ReferenceFeatureCache: tri-planar fields, original_depth, triplanar extraction
- TensorBoard routing: 3d_ prefix → Generation_3d/ section
- _compute_triplanar_metrics: integration with metric computation
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest
import torch

from medgen.metrics.generation_3d import (
    _extract_chunks,
    extract_features_3d,
    extract_features_3d_triplanar,
    volumes_to_slices,
)
from medgen.metrics.generation_sampling import StreamingFeatures
from medgen.metrics.generation import (
    GenerationMetrics,
    GenerationMetricsConfig,
    ReferenceFeatureCache,
)


# =============================================================================
# Mock extractor fixture
# =============================================================================


def _make_mock_extractor(feat_dim: int = 2048):
    """Create a mock feature extractor that returns [N, feat_dim] for [N, C, H, W] input."""
    mock = Mock()
    mock.extract_features = Mock(
        side_effect=lambda x: torch.randn(x.shape[0], feat_dim)
    )
    return mock


# =============================================================================
# _extract_chunks
# =============================================================================


class TestExtractChunks:
    """Test the shared _extract_chunks helper."""

    def test_output_shape(self):
        """Returns [N, feat_dim] for [N, C, H, W] input."""
        extractor = _make_mock_extractor(2048)
        slices = torch.randn(20, 1, 64, 64)
        features = _extract_chunks(slices, extractor, chunk_size=8)
        assert features.shape == (20, 2048)

    def test_single_chunk(self):
        """All slices fit in one chunk."""
        extractor = _make_mock_extractor(512)
        slices = torch.randn(5, 1, 32, 32)
        features = _extract_chunks(slices, extractor, chunk_size=64)
        assert features.shape == (5, 512)
        # Called once since 5 < 64
        assert extractor.extract_features.call_count == 1

    def test_multiple_chunks(self):
        """Slices split across multiple chunks."""
        extractor = _make_mock_extractor(2048)
        slices = torch.randn(30, 1, 64, 64)
        features = _extract_chunks(slices, extractor, chunk_size=8)
        assert features.shape == (30, 2048)
        # 30 / 8 = 4 chunks (8+8+8+6)
        assert extractor.extract_features.call_count == 4

    def test_features_on_cpu(self):
        """Output features are on CPU."""
        extractor = _make_mock_extractor(2048)
        slices = torch.randn(10, 1, 32, 32)
        features = _extract_chunks(slices, extractor, chunk_size=5)
        assert features.device == torch.device('cpu')


# =============================================================================
# extract_features_3d_triplanar
# =============================================================================


class TestExtractFeatures3DTriplanar:
    """Test tri-planar feature extraction from 3D volumes."""

    def test_output_shape_single_volume(self):
        """Single volume: [1, C, D, H, W] → [D+H+W, feat_dim] features."""
        extractor = _make_mock_extractor(2048)
        volumes = torch.randn(1, 1, 16, 32, 32)
        features = extract_features_3d_triplanar(volumes, extractor, chunk_size=64)
        # D=16, H=32, W=32 → 16+32+32 = 80
        assert features.shape == (80, 2048)

    def test_output_shape_multiple_volumes(self):
        """Multiple volumes: [B, C, D, H, W] → [B*(D+H+W), feat_dim]."""
        extractor = _make_mock_extractor(512)
        volumes = torch.randn(2, 1, 8, 16, 16)
        features = extract_features_3d_triplanar(volumes, extractor, chunk_size=64)
        # B=2, D=8, H=16, W=16 → 2*(8+16+16) = 80
        assert features.shape == (80, 512)

    def test_padding_removal(self):
        """Padded slices removed from axial dimension when original_depth specified."""
        extractor = _make_mock_extractor(2048)
        # Volume padded to D=20, original D=12
        volumes = torch.randn(1, 1, 20, 32, 32)
        features = extract_features_3d_triplanar(
            volumes, extractor, chunk_size=64, original_depth=12
        )
        # D=12 (after trimming), H=32, W=32 → 12+32+32 = 76
        assert features.shape == (76, 2048)

    def test_no_padding_removal_when_depth_matches(self):
        """No trimming when current depth equals original_depth."""
        extractor = _make_mock_extractor(2048)
        volumes = torch.randn(1, 1, 16, 32, 32)
        features = extract_features_3d_triplanar(
            volumes, extractor, chunk_size=64, original_depth=16
        )
        # D=16, H=32, W=32 → 16+32+32 = 80
        assert features.shape == (80, 2048)

    def test_no_padding_removal_when_none(self):
        """No trimming when original_depth is None."""
        extractor = _make_mock_extractor(2048)
        volumes = torch.randn(1, 1, 20, 32, 32)
        features = extract_features_3d_triplanar(
            volumes, extractor, chunk_size=64, original_depth=None
        )
        # D=20, H=32, W=32 → 20+32+32 = 84
        assert features.shape == (84, 2048)

    def test_rejects_4d_input(self):
        """Raises ValueError for 4D input (2D images)."""
        extractor = _make_mock_extractor(2048)
        images = torch.randn(8, 1, 64, 64)
        with pytest.raises(ValueError, match="Expected 5D"):
            extract_features_3d_triplanar(images, extractor)

    def test_rejects_3d_input(self):
        """Raises ValueError for 3D input."""
        extractor = _make_mock_extractor(2048)
        with pytest.raises(ValueError, match="Expected 5D"):
            extract_features_3d_triplanar(torch.randn(1, 64, 64), extractor)

    def test_more_features_than_axial_only(self):
        """Tri-planar produces more features than axial-only."""
        extractor = _make_mock_extractor(2048)
        volumes = torch.randn(1, 1, 16, 32, 32)

        axial_features = extract_features_3d(volumes, extractor, chunk_size=64)
        triplanar_features = extract_features_3d_triplanar(
            volumes, extractor, chunk_size=64
        )

        # Axial: D=16, Triplanar: D+H+W = 16+32+32 = 80
        assert axial_features.shape[0] == 16
        assert triplanar_features.shape[0] == 80
        assert triplanar_features.shape[0] > axial_features.shape[0]

    def test_slice_shapes_passed_to_extractor(self):
        """Extractor receives correct slice shapes for each plane."""
        call_shapes = []
        mock = Mock()
        mock.extract_features = Mock(
            side_effect=lambda x: (call_shapes.append(x.shape), torch.randn(x.shape[0], 128))[1]
        )

        volumes = torch.randn(1, 1, 8, 16, 32)
        extract_features_3d_triplanar(volumes, mock, chunk_size=256)

        # With chunk_size=256, each plane is a single chunk
        # Axial: [8, 1, 16, 32], Coronal: [16, 1, 8, 32], Sagittal: [32, 1, 8, 16]
        assert len(call_shapes) == 3
        assert call_shapes[0] == (8, 1, 16, 32)   # Axial: D slices, each [C, H, W]
        assert call_shapes[1] == (16, 1, 8, 32)    # Coronal: H slices, each [C, D, W]
        assert call_shapes[2] == (32, 1, 8, 16)    # Sagittal: W slices, each [C, D, H]

    def test_features_on_cpu(self):
        """Output features are on CPU regardless of input device."""
        extractor = _make_mock_extractor(2048)
        volumes = torch.randn(1, 1, 8, 16, 16)
        features = extract_features_3d_triplanar(volumes, extractor, chunk_size=64)
        assert features.device == torch.device('cpu')


# =============================================================================
# StreamingFeatures
# =============================================================================


class TestStreamingFeatures:
    """Test StreamingFeatures NamedTuple."""

    def test_field_names(self):
        """All expected fields present."""
        assert StreamingFeatures._fields == (
            'resnet', 'biomed', 'resnet_rin',
            'resnet_3d', 'biomed_3d', 'resnet_rin_3d',
            'diversity_samples',
        )

    def test_construction_all_populated(self):
        """Can construct with all fields populated."""
        sf = StreamingFeatures(
            resnet=torch.randn(10, 2048),
            biomed=torch.randn(10, 512),
            resnet_rin=torch.randn(10, 2048),
            resnet_3d=torch.randn(80, 2048),
            biomed_3d=torch.randn(80, 512),
            resnet_rin_3d=torch.randn(80, 2048),
            diversity_samples=torch.randn(2, 1, 16, 64, 64),
        )
        assert sf.resnet.shape == (10, 2048)
        assert sf.resnet_3d.shape == (80, 2048)
        assert sf.diversity_samples.shape[0] == 2

    def test_construction_with_nones(self):
        """Can construct with optional fields as None."""
        sf = StreamingFeatures(
            resnet=torch.randn(10, 2048),
            biomed=torch.randn(10, 512),
            resnet_rin=None,
            resnet_3d=None,
            biomed_3d=None,
            resnet_rin_3d=None,
            diversity_samples=None,
        )
        assert sf.resnet_rin is None
        assert sf.resnet_3d is None
        assert sf.diversity_samples is None

    def test_is_namedtuple(self):
        """StreamingFeatures is a proper NamedTuple (iterable, indexable)."""
        sf = StreamingFeatures(
            resnet=torch.randn(5, 2048),
            biomed=torch.randn(5, 512),
            resnet_rin=None,
            resnet_3d=None,
            biomed_3d=None,
            resnet_rin_3d=None,
            diversity_samples=None,
        )
        # Indexable
        assert sf[0] is sf.resnet
        assert sf[1] is sf.biomed
        # Length
        assert len(sf) == 7


# =============================================================================
# ReferenceFeatureCache — tri-planar fields
# =============================================================================


class TestReferenceFeatureCacheTriplanar:
    """Test ReferenceFeatureCache tri-planar support."""

    def _make_cache(self, original_depth=None):
        """Create a ReferenceFeatureCache with mock extractors."""
        resnet = _make_mock_extractor(2048)
        biomed = _make_mock_extractor(512)
        resnet_rin = _make_mock_extractor(2048)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ReferenceFeatureCache(
                resnet, biomed, Path(tmpdir), torch.device('cpu'),
                batch_size=32, resnet_rin_extractor=resnet_rin,
                original_depth=original_depth,
            )
            yield cache

    def test_original_depth_stored(self):
        """original_depth parameter stored on cache."""
        for cache in self._make_cache(original_depth=150):
            assert cache.original_depth == 150

    def test_original_depth_default_none(self):
        """original_depth defaults to None."""
        for cache in self._make_cache():
            assert cache.original_depth is None

    def test_triplanar_fields_initialized_none(self):
        """All _3d fields initialized to None."""
        for cache in self._make_cache():
            assert cache.train_resnet_3d is None
            assert cache.val_resnet_3d is None
            assert cache.train_biomed_3d is None
            assert cache.val_biomed_3d is None
            assert cache.train_resnet_rin_3d is None
            assert cache.val_resnet_rin_3d is None

    def test_extract_features_triplanar_param(self):
        """_extract_features_from_loader accepts triplanar parameter."""
        for cache in self._make_cache(original_depth=8):
            # Create mock dataloader with 3D data
            batch = {
                'image': torch.randn(2, 1, 10, 16, 16),
                'seg': (torch.rand(2, 1, 10, 16, 16) > 0.3).float(),
            }
            loader = [batch]

            # Axial-only: should produce B*D features
            axial_feats = cache._extract_features_from_loader(
                loader, cache.resnet, "test", triplanar=False
            )

            # Tri-planar: should produce B*(D'+H+W) features where D'=original_depth
            triplanar_feats = cache._extract_features_from_loader(
                loader, cache.resnet, "test", triplanar=True
            )

            # Tri-planar should produce more features than axial-only
            assert triplanar_feats.shape[0] > axial_feats.shape[0]


# =============================================================================
# GenerationMetrics — original_depth passed to cache
# =============================================================================


class TestGenerationMetricsTriplanar:
    """Test that GenerationMetrics passes original_depth to cache."""

    def test_original_depth_forwarded_to_cache(self):
        """Config's original_depth is forwarded to ReferenceFeatureCache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig(original_depth=150)
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )
            assert metrics.cache.original_depth == 150

    def test_no_original_depth_for_2d(self):
        """2D configs have original_depth=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )
            assert metrics.cache.original_depth is None

    def test_3d_extract_features_batched_triplanar(self):
        """_extract_features_batched (axial) unchanged by tri-planar changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig(original_depth=12)
            metrics = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )

            mock_extractor = _make_mock_extractor(2048)

            # 3D volume [1, 1, 16, 32, 32] with original_depth=12
            samples = torch.randn(1, 1, 16, 32, 32)
            features = metrics._extract_features_batched(samples, mock_extractor)

            # Existing behavior: axial only, padded slices removed → 12 features
            assert features.shape == (12, 2048)


# =============================================================================
# TensorBoard routing — log_generation and log_test_generation
# =============================================================================


class TestTensorBoardRouting:
    """Test that 3d_ prefix routes to Generation_3d/ section."""

    def _make_mock_metrics(self):
        """Create mock UnifiedMetrics with mock writer."""
        mock = Mock()
        mock.writer = MagicMock()
        mock.modality = None
        return mock

    def test_log_generation_3d_prefix(self):
        """Keys starting with '3d_' route to Generation_3d/."""
        from medgen.metrics.unified_logging import log_generation

        metrics = self._make_mock_metrics()
        results = {
            'KID_mean_train': 0.05,         # → Generation/
            '3d_KID_mean_train': 0.06,       # → Generation_3d/
            'Diversity/LPIPS': 0.3,          # → Generation_Diversity/
        }

        log_generation(metrics, epoch=10, results=results)

        calls = {c.args[0]: c.args[1:] for c in metrics.writer.add_scalar.call_args_list}

        # Standard metric → Generation/
        assert 'Generation/KID_mean_train' in calls
        # Tri-planar → Generation_3d/
        assert 'Generation_3d/KID_mean_train' in calls
        # Diversity → Generation_Diversity/
        assert 'Generation_Diversity/LPIPS' in calls

    def test_log_generation_3d_value_correct(self):
        """3d_ prefix stripped correctly, value passed through."""
        from medgen.metrics.unified_logging import log_generation

        metrics = self._make_mock_metrics()
        results = {'3d_CMMD_val': 0.42}

        log_generation(metrics, epoch=5, results=results)

        metrics.writer.add_scalar.assert_called_once_with(
            'Generation_3d/CMMD_val', 0.42, 5
        )

    def test_log_test_generation_3d_prefix(self):
        """Test generation logs 3d_ metrics to separate section."""
        from medgen.metrics.unified_logging import log_test_generation

        metrics = self._make_mock_metrics()
        results = {
            'KID_mean': 0.05,
            '3d_KID_mean': 0.06,
        }

        exported = log_test_generation(metrics, results, prefix='test_best')

        calls = {c.args[0] for c in metrics.writer.add_scalar.call_args_list}

        assert 'test_best_generation/KID_mean' in calls
        assert 'test_best_generation_3d/KID_mean' in calls

        # Exported dict uses gen_3d_ prefix
        assert 'gen_3d_kid_mean' in exported
        assert 'gen_kid_mean' in exported

    def test_log_generation_extended_3d_prefix(self):
        """Extended tri-planar metrics route correctly."""
        from medgen.metrics.unified_logging import log_generation

        metrics = self._make_mock_metrics()
        results = {
            '3d_extended_KID_mean_train': 0.04,
            '3d_extended_CMMD_val': 0.3,
        }

        log_generation(metrics, epoch=10, results=results)

        calls = {c.args[0] for c in metrics.writer.add_scalar.call_args_list}

        assert 'Generation_3d/extended_KID_mean_train' in calls
        assert 'Generation_3d/extended_CMMD_val' in calls


# =============================================================================
# _compute_triplanar_metrics
# =============================================================================


class TestComputeTriplanarMetrics:
    """Test _compute_triplanar_metrics helper."""

    def test_skips_when_no_3d_features(self):
        """No metrics computed when streaming has no 3d features."""
        from medgen.metrics.generation_computation import _compute_triplanar_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            gm = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )

            streaming = StreamingFeatures(
                resnet=torch.randn(10, 2048),
                biomed=torch.randn(10, 512),
                resnet_rin=None,
                resnet_3d=None,  # No tri-planar
                biomed_3d=None,
                resnet_rin_3d=None,
                diversity_samples=None,
            )

            results = {}
            _compute_triplanar_metrics(gm, streaming, results)
            # No 3d_ keys should be added
            assert not any(k.startswith('3d_') for k in results)

    def test_skips_when_no_cache_3d_features(self):
        """No metrics computed when cache has no 3d reference features."""
        from medgen.metrics.generation_computation import _compute_triplanar_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            gm = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )
            # Cache has no _3d features (default)
            assert gm.cache.train_resnet_3d is None

            streaming = StreamingFeatures(
                resnet=torch.randn(10, 2048),
                biomed=torch.randn(10, 512),
                resnet_rin=None,
                resnet_3d=torch.randn(80, 2048),  # Has generated tri-planar
                biomed_3d=torch.randn(80, 512),
                resnet_rin_3d=None,
                diversity_samples=None,
            )

            results = {}
            _compute_triplanar_metrics(gm, streaming, results)
            assert not any(k.startswith('3d_') for k in results)

    def test_computes_when_both_available(self):
        """Computes 3d_ metrics when both streaming and cache have features."""
        from medgen.metrics.generation_computation import _compute_triplanar_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            gm = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )
            # Populate cache with mock 3d features
            gm.cache.train_resnet_3d = torch.randn(200, 2048)
            gm.cache.train_biomed_3d = torch.randn(200, 512)
            gm.cache.val_resnet_3d = torch.randn(100, 2048)
            gm.cache.val_biomed_3d = torch.randn(100, 512)

            streaming = StreamingFeatures(
                resnet=torch.randn(10, 2048),
                biomed=torch.randn(10, 512),
                resnet_rin=None,
                resnet_3d=torch.randn(80, 2048),
                biomed_3d=torch.randn(80, 512),
                resnet_rin_3d=None,
                diversity_samples=None,
            )

            results = {}
            _compute_triplanar_metrics(gm, streaming, results)

            # Should have 3d_ prefixed keys with _train and _val suffixes
            assert '3d_KID_mean_train' in results
            assert '3d_KID_std_train' in results
            assert '3d_CMMD_train' in results
            assert '3d_KID_mean_val' in results
            assert '3d_KID_std_val' in results
            assert '3d_CMMD_val' in results

    def test_extended_prefix(self):
        """Extended prefix combined with 3d_ prefix."""
        from medgen.metrics.generation_computation import _compute_triplanar_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            gm = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )
            gm.cache.train_resnet_3d = torch.randn(200, 2048)
            gm.cache.train_biomed_3d = torch.randn(200, 512)
            gm.cache.val_resnet_3d = torch.randn(100, 2048)
            gm.cache.val_biomed_3d = torch.randn(100, 512)

            streaming = StreamingFeatures(
                resnet=torch.randn(10, 2048),
                biomed=torch.randn(10, 512),
                resnet_rin=None,
                resnet_3d=torch.randn(80, 2048),
                biomed_3d=torch.randn(80, 512),
                resnet_rin_3d=None,
                diversity_samples=None,
            )

            results = {}
            _compute_triplanar_metrics(gm, streaming, results, prefix="extended_")

            assert '3d_extended_KID_mean_train' in results
            assert '3d_extended_CMMD_val' in results

    def test_rin_features_included(self):
        """KID_RIN computed when RadImageNet features available."""
        from medgen.metrics.generation_computation import _compute_triplanar_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            gm = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )
            gm.cache.train_resnet_3d = torch.randn(200, 2048)
            gm.cache.train_biomed_3d = torch.randn(200, 512)
            gm.cache.val_resnet_3d = torch.randn(100, 2048)
            gm.cache.val_biomed_3d = torch.randn(100, 512)
            gm.cache.train_resnet_rin_3d = torch.randn(200, 2048)
            gm.cache.val_resnet_rin_3d = torch.randn(100, 2048)

            streaming = StreamingFeatures(
                resnet=torch.randn(10, 2048),
                biomed=torch.randn(10, 512),
                resnet_rin=torch.randn(10, 2048),
                resnet_3d=torch.randn(80, 2048),
                biomed_3d=torch.randn(80, 512),
                resnet_rin_3d=torch.randn(80, 2048),
                diversity_samples=None,
            )

            results = {}
            _compute_triplanar_metrics(gm, streaming, results)

            assert '3d_KID_RIN_mean_train' in results
            assert '3d_KID_RIN_std_train' in results
            assert '3d_KID_RIN_mean_val' in results


# =============================================================================
# Cache backward compatibility
# =============================================================================


class TestCacheBackwardCompatibility:
    """Test that old caches without tri-planar features still load."""

    def test_old_cache_loads_without_3d_keys(self):
        """Cache files from before tri-planar support load without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "test_exp_reference_features.pt"

            # Simulate old cache (no _3d keys)
            old_cache = {
                'train_resnet': torch.randn(100, 2048),
                'train_biomed': torch.randn(100, 512),
                'val_resnet': torch.randn(50, 2048),
                'val_biomed': torch.randn(50, 512),
                'train_resnet_rin': torch.randn(100, 2048),
                'val_resnet_rin': torch.randn(50, 2048),
            }
            torch.save(old_cache, cache_file)

            # Create cache without original_depth (2D experiment)
            resnet = _make_mock_extractor(2048)
            biomed = _make_mock_extractor(512)
            cache = ReferenceFeatureCache(
                resnet, biomed, cache_dir, torch.device('cpu'),
                batch_size=32,
            )

            # Mock loaders (shouldn't be used for 2D without original_depth)
            mock_loader = []

            cache.extract_and_cache(mock_loader, mock_loader, "test_exp")

            # Should load axial features
            assert cache.train_resnet is not None
            assert cache.val_resnet is not None
            # 3d features should be None (no backfill for 2D)
            assert cache.train_resnet_3d is None
            assert cache.val_resnet_3d is None


# =============================================================================
# Integration: metric key naming consistency
# =============================================================================


class TestMetricKeyNaming:
    """Test that metric keys follow consistent naming patterns."""

    def test_3d_prefix_before_metric_name(self):
        """3d_ prefix comes before metric name, suffix (_train/_val) at end."""
        # This is the pattern: 3d_{prefix}{METRIC}_{split}
        # e.g., 3d_KID_mean_train, 3d_extended_CMMD_val
        from medgen.metrics.generation_computation import compute_metrics_against_reference

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            gm = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )

            gen_resnet = torch.randn(50, 2048)
            gen_biomed = torch.randn(50, 512)
            ref_resnet = torch.randn(100, 2048)
            ref_biomed = torch.randn(100, 512)

            # With "3d_" prefix
            results = compute_metrics_against_reference(
                gm, gen_resnet, gen_biomed, ref_resnet, ref_biomed, prefix="3d_"
            )

            assert '3d_KID_mean' in results
            assert '3d_KID_std' in results
            assert '3d_CMMD' in results

    def test_3d_extended_prefix(self):
        """3d_extended_ prefix produces correct keys."""
        from medgen.metrics.generation_computation import compute_metrics_against_reference

        with tempfile.TemporaryDirectory() as tmpdir:
            config = GenerationMetricsConfig()
            gm = GenerationMetrics(
                config, torch.device('cpu'), Path(tmpdir), mode_name='bravo',
            )

            results = compute_metrics_against_reference(
                gm,
                torch.randn(50, 2048), torch.randn(50, 512),
                torch.randn(100, 2048), torch.randn(100, 512),
                prefix="3d_extended_",
            )

            assert '3d_extended_KID_mean' in results
            assert '3d_extended_CMMD' in results
