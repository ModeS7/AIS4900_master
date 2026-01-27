"""Integration tests for trainer equivalence.

These tests verify that trainers produce deterministic, reproducible results
by comparing against captured baselines.

NOTE: Baseline comparison tests are marked with @pytest.mark.baseline.
These tests may fail after code changes that affect training behavior.
To regenerate baselines:
    python misc/experiments/capture_baseline.py --spatial_dims 2 --mode bravo --output tests/integration/baselines/2d_bravo_baseline.json
    python misc/experiments/capture_baseline.py --spatial_dims 2 --mode seg --output tests/integration/baselines/2d_seg_baseline.json
    python misc/experiments/capture_baseline.py --spatial_dims 3 --mode bravo --output tests/integration/baselines/3d_bravo_baseline.json
    python misc/experiments/capture_baseline.py --spatial_dims 3 --mode seg --output tests/integration/baselines/3d_seg_baseline.json

Usage:
    # Run all equivalence tests
    pytest tests/integration/test_trainer_equivalence.py -v

    # Skip baseline comparison tests (run only determinism tests)
    pytest tests/integration/test_trainer_equivalence.py -v -m "not baseline"

    # Run specific dimension
    pytest tests/integration/test_trainer_equivalence.py -v -k "2d"

    # Skip slow 3D tests
    pytest tests/integration/test_trainer_equivalence.py -v -k "not 3d"
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

# Import baseline utilities
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "misc" / "experiments"))

from capture_baseline import (
    capture_baseline,
    compare_baselines,
    set_deterministic,
)

logger = logging.getLogger(__name__)

# Baseline directory (co-located with integration tests)
BASELINES_DIR = Path(__file__).parent / "baselines"

# Test tolerance for floating point comparison
TOLERANCE = 1e-5


def load_baseline(name: str) -> Dict[str, Any]:
    """Load a baseline JSON file."""
    path = BASELINES_DIR / f"{name}_baseline.json"
    if not path.exists():
        pytest.skip(f"Baseline not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)


class TestTrainer2DEquivalence:
    """Test 2D trainer produces equivalent results to baseline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up deterministic state."""
        set_deterministic(42)

    @pytest.mark.baseline
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for deterministic comparison"
    )
    def test_2d_bravo_matches_baseline(self):
        """2D bravo mode produces same results as baseline."""
        baseline = load_baseline("2d_bravo")

        current = capture_baseline(
            spatial_dims=2,
            num_steps=10,
            seed=42,
            mode='bravo',
        )

        assert compare_baselines(baseline, current, tolerance=TOLERANCE), \
            "2D bravo trainer results differ from baseline"

    @pytest.mark.baseline
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for deterministic comparison"
    )
    def test_2d_seg_matches_baseline(self):
        """2D seg mode produces same results as baseline."""
        baseline = load_baseline("2d_seg")

        current = capture_baseline(
            spatial_dims=2,
            num_steps=10,
            seed=42,
            mode='seg',
        )

        assert compare_baselines(baseline, current, tolerance=TOLERANCE), \
            "2D seg trainer results differ from baseline"


class TestTrainer3DEquivalence:
    """Test 3D trainer produces equivalent results to baseline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up deterministic state."""
        set_deterministic(42)

    @pytest.mark.timeout(60)
    @pytest.mark.baseline
    @pytest.mark.slow
    @pytest.mark.timeout(180)  # 3 minutes for 3D operations
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for deterministic comparison"
    )
    def test_3d_bravo_matches_baseline(self):
        """3D bravo mode produces same results as baseline."""
        baseline = load_baseline("3d_bravo")

        current = capture_baseline(
            spatial_dims=3,
            num_steps=10,
            seed=42,
            mode='bravo',
        )

        assert compare_baselines(baseline, current, tolerance=TOLERANCE), \
            "3D bravo trainer results differ from baseline"

    @pytest.mark.timeout(60)
    @pytest.mark.baseline
    @pytest.mark.slow
    @pytest.mark.timeout(180)  # 3 minutes for 3D operations
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for deterministic comparison"
    )
    def test_3d_seg_matches_baseline(self):
        """3D seg mode produces same results as baseline."""
        baseline = load_baseline("3d_seg")

        current = capture_baseline(
            spatial_dims=3,
            num_steps=10,
            seed=42,
            mode='seg',
        )

        assert compare_baselines(baseline, current, tolerance=TOLERANCE), \
            "3D seg trainer results differ from baseline"


class TestTrainerDeterminism:
    """Test trainers are deterministic across runs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up deterministic state."""
        set_deterministic(42)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for deterministic comparison"
    )
    def test_2d_is_deterministic(self):
        """Two runs with same seed produce identical results."""
        run1 = capture_baseline(spatial_dims=2, num_steps=5, seed=42, mode='bravo')
        run2 = capture_baseline(spatial_dims=2, num_steps=5, seed=42, mode='bravo')

        assert run1['final_state_hash'] == run2['final_state_hash'], \
            "2D trainer is not deterministic"

        for i, (s1, s2) in enumerate(zip(run1['step_metrics'], run2['step_metrics'])):
            assert abs(s1['total_loss'] - s2['total_loss']) < 1e-6, \
                f"Step {i} loss differs between runs"

    @pytest.mark.timeout(60)
    @pytest.mark.slow
    @pytest.mark.timeout(180)  # 3 minutes for 3D operations
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required for deterministic comparison"
    )
    def test_3d_is_deterministic(self):
        """Two runs with same seed produce identical results."""
        run1 = capture_baseline(spatial_dims=3, num_steps=3, seed=42, mode='bravo')
        run2 = capture_baseline(spatial_dims=3, num_steps=3, seed=42, mode='bravo')

        assert run1['final_state_hash'] == run2['final_state_hash'], \
            "3D trainer is not deterministic"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
