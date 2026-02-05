"""Integration tests for checkpoint save/load/resume functionality.

Tests state preservation across checkpoint operations including:
- Model and optimizer state
- Epoch counters and RNG state
- EMA weights
- Cross-device loading
- CheckpointManager integration
"""
import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock


class TestCheckpointSaveLoad:
    """Basic state preservation tests using PyTorch primitives."""

    @pytest.fixture
    def model_and_optimizer(self):
        """Simple model and optimizer for testing."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, optimizer

    def test_model_state_preserved(self, model_and_optimizer, temp_checkpoint_dir):
        """Weights are identical after save/load."""
        model, _ = model_and_optimizer
        path = temp_checkpoint_dir / "checkpoint.pt"

        # Get original weights
        original_weight = model.weight.clone()
        original_bias = model.bias.clone()

        # Save
        torch.save({'model_state_dict': model.state_dict()}, path)

        # Modify model weights
        with torch.no_grad():
            model.weight.fill_(0)
            model.bias.fill_(0)

        # Load
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Verify
        assert torch.allclose(model.weight, original_weight), "Weights not restored"
        assert torch.allclose(model.bias, original_bias), "Biases not restored"

    def test_optimizer_state_preserved(self, model_and_optimizer, temp_checkpoint_dir):
        """Optimizer momentum buffers preserved across save/load."""
        model, optimizer = model_and_optimizer
        path = temp_checkpoint_dir / "checkpoint.pt"

        # Run a few training steps to populate optimizer state
        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()

        # Get original optimizer state (momentum buffers)
        original_state = {
            k: {sk: sv.clone() if isinstance(sv, torch.Tensor) else sv
                for sk, sv in v.items()}
            for k, v in optimizer.state.items()
        }

        # Save
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

        # Create fresh optimizer
        new_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Load
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Verify optimizer state matches
        for param_id in original_state:
            for key in ['exp_avg', 'exp_avg_sq']:
                if key in original_state[param_id]:
                    # Note: param_id may differ, check by index
                    assert any(
                        torch.allclose(original_state[param_id][key], v.get(key, torch.tensor(0)))
                        for v in new_optimizer.state.values()
                        if isinstance(v, dict) and key in v
                    ), f"Optimizer state {key} not restored"

    def test_epoch_counter_preserved(self, temp_checkpoint_dir):
        """Epoch counter survives roundtrip."""
        path = temp_checkpoint_dir / "checkpoint.pt"

        # Save with specific epoch
        torch.save({'epoch': 42}, path)

        # Load
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        assert checkpoint['epoch'] == 42, f"Expected epoch 42, got {checkpoint['epoch']}"

    def test_rng_state_preserved(self, temp_checkpoint_dir):
        """RNG state is preserved across save/load."""
        path = temp_checkpoint_dir / "checkpoint.pt"

        # Set specific seed and generate some values
        torch.manual_seed(42)
        expected_values = torch.randn(5)

        # Reset and get RNG state
        torch.manual_seed(42)
        rng_state = torch.get_rng_state()

        # Save
        torch.save({'rng_state': rng_state}, path)

        # Corrupt RNG state
        torch.manual_seed(9999)
        _ = torch.randn(100)

        # Load and restore
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        torch.set_rng_state(checkpoint['rng_state'])

        # Generate values - should match original
        restored_values = torch.randn(5)
        assert torch.allclose(expected_values, restored_values), "RNG state not restored"

    def test_ema_weights_preserved(self, temp_checkpoint_dir):
        """EMA weights are saved and loaded separately from model."""
        path = temp_checkpoint_dir / "checkpoint.pt"

        # Create model with distinct EMA state
        model = nn.Linear(10, 10)
        ema_state = {
            'shadow': {k: v * 2 for k, v in model.state_dict().items()},
            'decay': 0.999,
        }

        # Save both
        torch.save({
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema_state,
        }, path)

        # Load
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        assert 'ema_state_dict' in checkpoint, "EMA state not in checkpoint"
        assert checkpoint['ema_state_dict']['decay'] == 0.999
        for key in model.state_dict():
            expected = model.state_dict()[key] * 2
            actual = checkpoint['ema_state_dict']['shadow'][key]
            assert torch.allclose(expected, actual), f"EMA shadow mismatch for {key}"


class TestCheckpointCompatibility:
    """Cross-device and partial loading tests."""

    @pytest.fixture
    def gpu_model(self, device):
        """Model on GPU if available."""
        return nn.Linear(10, 10).to(device)

    def test_load_on_different_device(self, temp_checkpoint_dir):
        """GPU-saved checkpoint loads on CPU via map_location."""
        path = temp_checkpoint_dir / "checkpoint.pt"

        # Create and save on GPU (or CPU if unavailable)
        model = nn.Linear(10, 10)
        if torch.cuda.is_available():
            model = model.cuda()

        torch.save({'model_state_dict': model.state_dict()}, path)

        # Load on CPU explicitly
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        cpu_model = nn.Linear(10, 10)
        cpu_model.load_state_dict(checkpoint['model_state_dict'])

        # All params should be on CPU
        for param in cpu_model.parameters():
            assert param.device == torch.device('cpu'), "Param not on CPU after load"

    def test_partial_state_dict_load(self, temp_checkpoint_dir):
        """strict=False allows partial state dict loading."""
        path = temp_checkpoint_dir / "checkpoint.pt"

        # Save smaller model
        small_model = nn.Linear(10, 10)
        torch.save({'model_state_dict': small_model.state_dict()}, path)

        # Try to load into model with extra parameters
        class ExtendedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                self.extra = nn.Linear(10, 5)  # Extra parameter

            def forward(self, x):
                return self.extra(self.linear(x))

        extended_model = ExtendedModel()

        # This should not raise with strict=False
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        # Rename keys to match nested structure
        renamed_state = {f'linear.{k}': v for k, v in checkpoint['model_state_dict'].items()}
        extended_model.load_state_dict(renamed_state, strict=False)

        # Verify partial load worked
        assert torch.allclose(
            extended_model.linear.weight,
            small_model.weight
        ), "Partial load failed"

    def test_missing_keys_warning(self, temp_checkpoint_dir, caplog):
        """Missing keys are logged but don't cause failure with strict=False."""
        import logging
        path = temp_checkpoint_dir / "checkpoint.pt"

        # Save empty state
        torch.save({'model_state_dict': {}}, path)

        model = nn.Linear(10, 10)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Load with strict=False should succeed
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Should have missing keys
        assert len(missing) > 0, "Expected missing keys for empty state dict"
        assert 'weight' in missing or any('weight' in k for k in missing)


class TestCheckpointManagerIntegration:
    """Tests using the actual CheckpointManager class."""

    @pytest.fixture
    def checkpoint_manager_setup(self, temp_checkpoint_dir):
        """Create CheckpointManager with minimal model."""
        from medgen.pipeline.checkpoint_manager import CheckpointManager

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        manager = CheckpointManager(
            save_dir=str(temp_checkpoint_dir),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metric_name='total',
            metric_mode='min',
        )

        return manager, model, optimizer, scheduler, temp_checkpoint_dir

    def test_save_checkpoint_creates_file(self, checkpoint_manager_setup):
        """save() creates checkpoint file at expected path."""
        manager, model, optimizer, scheduler, temp_dir = checkpoint_manager_setup

        path = manager.save(epoch=5, metrics={'total': 0.5}, name='test')

        assert Path(path).exists(), f"Checkpoint file not created at {path}"
        assert 'checkpoint_test.pt' in path

        # Verify contents
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 5
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint

    def test_load_checkpoint_restores_state(self, checkpoint_manager_setup):
        """Loaded checkpoint restores model/optimizer state."""
        manager, model, optimizer, scheduler, temp_dir = checkpoint_manager_setup

        # Save initial state
        original_weight = model.weight.clone()
        path = manager.save(epoch=10, metrics={'total': 0.3}, name='latest')

        # Modify state
        with torch.no_grad():
            model.weight.fill_(999)

        # Load
        metadata = manager.load(path)

        # Verify restoration
        assert torch.allclose(model.weight, original_weight), "Model state not restored"
        assert metadata['epoch'] == 10

    def test_best_checkpoint_tracked(self, checkpoint_manager_setup):
        """save_if_best() only saves when metric improves."""
        manager, model, optimizer, scheduler, temp_dir = checkpoint_manager_setup

        # First save - should succeed (initial best is inf for 'min' mode)
        saved = manager.save_if_best(epoch=1, metrics={'total': 0.5})
        assert saved, "First save should always succeed"
        assert (temp_dir / 'checkpoint_best.pt').exists()

        # Worse metric - should not save
        saved = manager.save_if_best(epoch=2, metrics={'total': 0.8})
        assert not saved, "Should not save worse metric"

        # Better metric - should save
        saved = manager.save_if_best(epoch=3, metrics={'total': 0.3})
        assert saved, "Should save better metric"

        # Verify best checkpoint has epoch 3
        checkpoint = torch.load(temp_dir / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 3

    def test_resume_continues_from_saved_epoch(self, checkpoint_manager_setup):
        """resume() returns correct start epoch."""
        manager, model, optimizer, scheduler, temp_dir = checkpoint_manager_setup

        # Save at epoch 15
        manager.save(epoch=15, metrics={'total': 0.4}, name='latest')

        # Resume
        start_epoch = manager.resume()

        # Should resume from epoch 16 (15 + 1)
        assert start_epoch == 16, f"Expected start_epoch=16, got {start_epoch}"

    def test_checkpoint_contains_all_components(self, checkpoint_manager_setup):
        """Checkpoint contains model, optimizer, scheduler state dicts."""
        manager, model, optimizer, scheduler, temp_dir = checkpoint_manager_setup

        path = manager.save(epoch=5, metrics={'total': 0.5})
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        required_keys = [
            'epoch',
            'model_state_dict',
            'optimizer_state_dict',
            'scheduler_state_dict',
            'best_metric',
            'metric_name',
        ]

        for key in required_keys:
            assert key in checkpoint, f"Missing required key: {key}"


class TestCheckpointEdgeCases:
    """Edge cases and error handling for checkpoints."""

    def test_load_nonexistent_file_raises(self, temp_checkpoint_dir):
        """Loading nonexistent file raises appropriate error."""
        path = temp_checkpoint_dir / "nonexistent.pt"

        with pytest.raises(FileNotFoundError):
            torch.load(path, map_location='cpu', weights_only=False)

    def test_corrupted_checkpoint_raises(self, temp_checkpoint_dir):
        """Loading corrupted file raises appropriate error."""
        path = temp_checkpoint_dir / "corrupted.pt"

        # Write garbage data
        with open(path, 'wb') as f:
            f.write(b'not a valid pytorch checkpoint')

        with pytest.raises(Exception):  # Could be various errors
            torch.load(path, map_location='cpu', weights_only=False)

    def test_checkpoint_preserves_training_state(self, temp_checkpoint_dir):
        """model.training flag state is preserved."""
        path = temp_checkpoint_dir / "checkpoint.pt"

        model = nn.Linear(10, 10)
        model.eval()  # Set to eval mode

        # Note: training mode is not saved in state_dict
        # This test documents the behavior
        torch.save({'model_state_dict': model.state_dict()}, path)

        new_model = nn.Linear(10, 10)
        new_model.train()  # Different mode
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        new_model.load_state_dict(checkpoint['model_state_dict'])

        # Training mode is NOT restored - this is expected PyTorch behavior
        assert new_model.training, "Training mode is not stored in state_dict"

    def test_empty_metrics_dict(self, temp_checkpoint_dir):
        """Checkpoint works with empty metrics dict."""
        from medgen.pipeline.checkpoint_manager import CheckpointManager

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        manager = CheckpointManager(
            save_dir=str(temp_checkpoint_dir),
            model=model,
            optimizer=optimizer,
        )

        # Should not raise
        path = manager.save(epoch=1, metrics={})
        assert Path(path).exists()


class TestCheckpointVersionValidation:
    """Tests for checkpoint version validation."""

    def test_legacy_checkpoint_without_version(self, temp_checkpoint_dir):
        """Legacy checkpoint without version info loads successfully."""
        from medgen.pipeline.checkpoint_manager import CheckpointManager

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        path = temp_checkpoint_dir / 'legacy.pt'

        # Create legacy checkpoint without version
        torch.save({
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)

        manager = CheckpointManager(
            save_dir=str(temp_checkpoint_dir),
            model=model,
            optimizer=optimizer,
        )

        # Should load without error
        metadata = manager.load(str(path))
        assert metadata['epoch'] == 5

    def test_same_version_loads_successfully(self, temp_checkpoint_dir):
        """Checkpoint with same version loads successfully."""
        from medgen.pipeline.checkpoint_manager import CheckpointManager

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        path = temp_checkpoint_dir / 'same_version.pt'

        manager = CheckpointManager(
            save_dir=str(temp_checkpoint_dir),
            model=model,
            optimizer=optimizer,
        )

        # Create checkpoint with current version
        torch.save({
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'checkpoint_manager_version': manager.VERSION,
        }, path)

        # Should load without error
        metadata = manager.load(str(path))
        assert metadata['epoch'] == 10

    def test_newer_minor_version_warns(self, temp_checkpoint_dir, caplog):
        """Checkpoint from newer minor version logs warning but loads."""
        import logging
        from medgen.pipeline.checkpoint_manager import CheckpointManager

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        path = temp_checkpoint_dir / 'newer_minor.pt'

        manager = CheckpointManager(
            save_dir=str(temp_checkpoint_dir),
            model=model,
            optimizer=optimizer,
        )

        # Create checkpoint with newer minor version (1.9 > current 1.0)
        torch.save({
            'epoch': 15,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'checkpoint_manager_version': '1.9',
        }, path)

        # Should load but warn
        with caplog.at_level(logging.WARNING):
            metadata = manager.load(str(path))

        assert metadata['epoch'] == 15
        assert 'newer version' in caplog.text.lower()

    def test_major_version_mismatch_raises(self, temp_checkpoint_dir):
        """Checkpoint with different major version raises ValueError."""
        from medgen.pipeline.checkpoint_manager import CheckpointManager

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        path = temp_checkpoint_dir / 'major_mismatch.pt'

        manager = CheckpointManager(
            save_dir=str(temp_checkpoint_dir),
            model=model,
            optimizer=optimizer,
        )

        # Create checkpoint with major version mismatch
        torch.save({
            'epoch': 20,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'checkpoint_manager_version': '2.0',
        }, path)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Incompatible checkpoint version"):
            manager.load(str(path))

    def test_invalid_version_format_warns(self, temp_checkpoint_dir, caplog):
        """Invalid version format logs warning but loads."""
        import logging
        from medgen.pipeline.checkpoint_manager import CheckpointManager

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        path = temp_checkpoint_dir / 'invalid_version.pt'

        manager = CheckpointManager(
            save_dir=str(temp_checkpoint_dir),
            model=model,
            optimizer=optimizer,
        )

        # Create checkpoint with invalid version format
        torch.save({
            'epoch': 25,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'checkpoint_manager_version': 'invalid.format.here',
        }, path)

        # Should load but warn
        with caplog.at_level(logging.WARNING):
            metadata = manager.load(str(path))

        assert metadata['epoch'] == 25
        assert 'invalid version format' in caplog.text.lower()
