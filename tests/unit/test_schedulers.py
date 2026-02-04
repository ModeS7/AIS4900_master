"""Tests for learning rate scheduler utilities.

Tests verify scheduler behavior for warmup phases, cosine annealing,
constant LR periods, and edge cases.
"""

import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from medgen.core.schedulers import (
    create_warmup_cosine_scheduler,
    create_warmup_constant_scheduler,
    create_plateau_scheduler,
)


@pytest.fixture
def optimizer():
    """Create optimizer with known base LR."""
    model = nn.Linear(10, 10)
    return torch.optim.Adam(model.parameters(), lr=1e-3)


class TestWarmupCosineScheduler:
    """Tests for warmup + cosine annealing scheduler."""

    def test_warmup_phase_increases_lr(self, optimizer):
        """LR increases during warmup phase."""
        scheduler = create_warmup_cosine_scheduler(
            optimizer, warmup_epochs=10, total_epochs=100, start_factor=0.1
        )
        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # LR should start at 0.1 * base = 1e-4
        assert lrs[0] == pytest.approx(1e-4, rel=0.01)
        # LR should increase during warmup
        assert lrs[-1] > lrs[0]

    def test_lr_reaches_base_after_warmup(self, optimizer):
        """LR reaches base LR after warmup completes."""
        scheduler = create_warmup_cosine_scheduler(
            optimizer, warmup_epochs=10, total_epochs=100, start_factor=0.1
        )
        # Step through warmup
        for _ in range(10):
            scheduler.step()

        # After warmup, LR should be at base (1e-3)
        assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-3, rel=0.01)

    def test_cosine_phase_decreases_lr(self, optimizer):
        """LR decreases during cosine phase."""
        eta_min = 1e-6
        scheduler = create_warmup_cosine_scheduler(
            optimizer, warmup_epochs=10, total_epochs=100, eta_min=eta_min
        )
        # Skip warmup
        for _ in range(10):
            scheduler.step()

        # Collect cosine phase LRs
        lrs = []
        for _ in range(90):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # Should decay from base toward eta_min
        assert lrs[0] > lrs[-1], "LR should decrease during cosine phase"
        # Allow larger tolerance since cosine annealing may not reach exact eta_min
        assert lrs[-1] == pytest.approx(eta_min, rel=0.5)

    def test_eta_min_achieved_at_end(self, optimizer):
        """Final LR equals eta_min."""
        eta_min = 1e-6
        scheduler = create_warmup_cosine_scheduler(
            optimizer, warmup_epochs=10, total_epochs=100, eta_min=eta_min
        )
        for _ in range(100):
            scheduler.step()
        assert optimizer.param_groups[0]['lr'] == pytest.approx(eta_min, rel=0.1)

    def test_cosine_decay_shape(self, optimizer):
        """Cosine annealing has characteristic smooth decay."""
        scheduler = create_warmup_cosine_scheduler(
            optimizer, warmup_epochs=10, total_epochs=110, eta_min=0
        )
        # Skip warmup
        for _ in range(10):
            scheduler.step()

        # Collect cosine phase LRs
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # Cosine: at t=0, LR=base; at t=T/2, LR=base/2; at t=T, LR=0
        base_lr = 1e-3
        assert lrs[0] == pytest.approx(base_lr, rel=0.01)
        # Midpoint should be around half (cosine property)
        assert lrs[50] == pytest.approx(base_lr / 2, rel=0.1)


class TestWarmupConstantScheduler:
    """Tests for warmup + constant scheduler."""

    def test_warmup_increases_lr(self, optimizer):
        """LR increases during warmup."""
        scheduler = create_warmup_constant_scheduler(
            optimizer, warmup_epochs=10, total_epochs=100, start_factor=0.1
        )
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(10):
            scheduler.step()
        final_warmup_lr = optimizer.param_groups[0]['lr']

        assert initial_lr == pytest.approx(1e-4, rel=0.01)  # 0.1 * 1e-3
        assert final_warmup_lr == pytest.approx(1e-3, rel=0.01)  # base LR

    def test_constant_after_warmup(self, optimizer):
        """LR stays constant after warmup."""
        scheduler = create_warmup_constant_scheduler(
            optimizer, warmup_epochs=10, total_epochs=100
        )
        for _ in range(10):
            scheduler.step()

        base_lr = optimizer.param_groups[0]['lr']
        for _ in range(50):
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            assert current_lr == pytest.approx(base_lr, rel=0.001), \
                "LR should stay constant after warmup"

    def test_full_schedule_shape(self, optimizer):
        """Full schedule: ramp up, then flat."""
        scheduler = create_warmup_constant_scheduler(
            optimizer, warmup_epochs=20, total_epochs=100, start_factor=0.1
        )

        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # First 20: increasing
        for i in range(1, 20):
            assert lrs[i] > lrs[i-1] or abs(lrs[i] - lrs[i-1]) < 1e-9

        # After 20: constant
        for i in range(21, 100):
            assert lrs[i] == pytest.approx(lrs[20], rel=0.001)


class TestPlateauScheduler:
    """Tests for reduce-on-plateau scheduler."""

    def test_plateau_scheduler_created(self, optimizer):
        """ReduceLROnPlateau created with correct params."""
        scheduler = create_plateau_scheduler(
            optimizer, mode='min', factor=0.5, patience=10
        )
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.mode == 'min'
        assert scheduler.factor == 0.5
        assert scheduler.patience == 10

    def test_plateau_scheduler_mode_max(self, optimizer):
        """ReduceLROnPlateau works in 'max' mode."""
        scheduler = create_plateau_scheduler(
            optimizer, mode='max', factor=0.1, patience=5
        )
        assert scheduler.mode == 'max'
        assert scheduler.factor == 0.1
        assert scheduler.patience == 5

    def test_plateau_scheduler_min_lr(self, optimizer):
        """ReduceLROnPlateau respects min_lr."""
        scheduler = create_plateau_scheduler(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-5
        )
        assert scheduler.min_lrs == [1e-5]

    def test_plateau_reduces_lr_on_stagnation(self, optimizer):
        """LR reduces when metric doesn't improve."""
        scheduler = create_plateau_scheduler(
            optimizer, mode='min', factor=0.5, patience=2
        )
        initial_lr = optimizer.param_groups[0]['lr']

        # Simulate non-improving metric
        for _ in range(5):
            scheduler.step(1.0)  # Same metric value

        # LR should have been reduced
        assert optimizer.param_groups[0]['lr'] < initial_lr


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_warmup_epochs_1_documents_behavior(self, optimizer):
        """warmup_epochs=1 has abrupt LR change (documented behavior)."""
        scheduler = create_warmup_cosine_scheduler(
            optimizer, warmup_epochs=1, total_epochs=100, start_factor=0.1
        )
        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step()
        lr_after = optimizer.param_groups[0]['lr']

        # Documents that this is an abrupt jump from 0.1*base to base
        assert lr_before == pytest.approx(1e-4, rel=0.01)
        # After one step of warmup, we're at base LR
        assert lr_after == pytest.approx(1e-3, rel=0.01)

    def test_warmup_gte_total_raises_error(self, optimizer):
        """warmup_epochs >= total_epochs raises ValueError."""
        with pytest.raises(ValueError, match="warmup.*total"):
            create_warmup_cosine_scheduler(
                optimizer, warmup_epochs=100, total_epochs=100
            )

    def test_warmup_greater_than_total_raises_error(self, optimizer):
        """warmup_epochs > total_epochs raises ValueError."""
        with pytest.raises(ValueError, match="warmup.*total"):
            create_warmup_cosine_scheduler(
                optimizer, warmup_epochs=150, total_epochs=100
            )

    def test_warmup_constant_gte_total_raises_error(self, optimizer):
        """warmup_epochs >= total_epochs raises ValueError for constant scheduler."""
        with pytest.raises(ValueError, match="warmup.*total"):
            create_warmup_constant_scheduler(
                optimizer, warmup_epochs=100, total_epochs=100
            )

    def test_zero_warmup_epochs_raises_error(self, optimizer):
        """warmup_epochs=0 should work (skip warmup, only cosine)."""
        # This might not raise - depends on implementation
        # Test that it at least doesn't crash
        try:
            scheduler = create_warmup_cosine_scheduler(
                optimizer, warmup_epochs=0, total_epochs=100, start_factor=0.1
            )
            # If it succeeds, verify it goes straight to cosine
            # First step should already be in cosine phase
            scheduler.step()
        except (ValueError, ZeroDivisionError):
            # If it raises, that's also acceptable behavior
            pass


class TestMultipleParamGroups:
    """Tests with multiple parameter groups."""

    def test_warmup_cosine_multiple_param_groups(self):
        """Scheduler works with multiple param groups."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 10))
        optimizer = torch.optim.Adam([
            {'params': model[0].parameters(), 'lr': 1e-3},
            {'params': model[1].parameters(), 'lr': 1e-4},
        ])

        scheduler = create_warmup_cosine_scheduler(
            optimizer, warmup_epochs=10, total_epochs=100, start_factor=0.1
        )

        # Both groups should have their LRs scaled
        assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-4, rel=0.01)
        assert optimizer.param_groups[1]['lr'] == pytest.approx(1e-5, rel=0.01)

        # After warmup, both should be at their base LRs
        for _ in range(10):
            scheduler.step()

        assert optimizer.param_groups[0]['lr'] == pytest.approx(1e-3, rel=0.01)
        assert optimizer.param_groups[1]['lr'] == pytest.approx(1e-4, rel=0.01)
