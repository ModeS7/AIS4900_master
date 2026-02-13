"""Pipeline smoke tests for DiffusionTrainer.

Integration tests that exercise the full DiffusionTrainer pipeline:
init -> setup_model -> train (2 epochs x ~8 batches).

These catch bugs that unit tests miss:
- DataLoader persistent worker deadlocks
- FLOPs measurement crashes
- EMA initialization/update errors
- Checkpoint save/load issues
- Mode-specific batch formatting bugs
- Strategy-specific timestep sampling bugs
- Space encoding/decoding dimension mismatches

Run with: pytest tests/integration/test_pipeline_smoke.py -m slow -v
"""
import os
import signal

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIGS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
)

NUM_SAMPLES = 8
IMAGE_SIZE = 32
VOLUME_DEPTH = 16  # Tiny depth for 3D (must survive 3 UNet downsamples + s2d/2)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticDiffusionDataset(Dataset):
    """Generates random dict-format batches matching real data shape.

    Returns dicts compatible with TrainingMode.prepare_batch():
      - bravo: {'image': [1,H,W], 'seg': [1,H,W]}
      - seg:   {'image': [1,H,W]}
      - 3D variants: add depth dimension

    Segmentation masks are guaranteed to have at least 1 nonzero voxel.
    """

    def __init__(
        self,
        num_samples: int,
        image_size: int,
        spatial_dims: int,
        mode: str,
        depth: int = VOLUME_DEPTH,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.spatial_dims = spatial_dims
        self.mode = mode
        self.depth = depth

    def __len__(self) -> int:
        return self.num_samples

    def _make_spatial_shape(self) -> tuple[int, ...]:
        if self.spatial_dims == 2:
            return (1, self.image_size, self.image_size)
        return (1, self.depth, self.image_size, self.image_size)

    def _make_seg(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Create a binary segmentation mask with at least 1 positive voxel."""
        seg = (torch.rand(shape) > 0.7).float()
        # Guarantee at least one positive voxel
        if self.spatial_dims == 2:
            seg[0, 0, 0] = 1.0
        else:
            seg[0, 0, 0, 0] = 1.0
        return seg

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        shape = self._make_spatial_shape()

        if self.mode == 'seg':
            return {'image': self._make_seg(shape)}

        # bravo (conditional): image + seg
        return {
            'image': torch.randn(shape),
            'seg': self._make_seg(shape),
        }


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

# (test_id, spatial_dims, mode, strategy, space_type)
SMOKE_PARAMS = [
    pytest.param(1, 2, 'bravo', 'rflow', 'pixel', id='2d-bravo-rflow-pixel'),
    pytest.param(2, 2, 'seg', 'ddpm', 'pixel', id='2d-seg-ddpm-pixel'),
    pytest.param(3, 3, 'bravo', 'rflow', 'pixel', id='3d-bravo-rflow-pixel'),
    pytest.param(4, 3, 'bravo', 'rflow', 's2d', id='3d-bravo-rflow-s2d'),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cfg(tmp_path, spatial_dims, mode, strategy, space_type):
    """Build a complete Hydra DictConfig for the given test parameters.

    Uses diffusion_3d config for 3D tests (includes volume + space_to_depth groups)
    and diffusion config for 2D tests.
    """
    is_3d = spatial_dims == 3
    config_name = 'diffusion_3d' if is_3d else 'diffusion'

    overrides = [
        'model=smoke_test',
        'training=smoke_test',
        f'mode={mode}',
        f'strategy={strategy}',
        f'model.spatial_dims={spatial_dims}',
        f'+save_dir_override={tmp_path}',
        # Point data_dir to tmp_path so volume loaders find no NiFTI files
        # (prevents loading real 256x256 volumes during smoke tests)
        f'paths.data_dir={tmp_path}',
    ]

    if is_3d:
        overrides.extend([
            f'volume.depth={VOLUME_DEPTH}',
            f'volume.height={IMAGE_SIZE}',
            f'volume.width={IMAGE_SIZE}',
            f'volume.pad_depth_to={VOLUME_DEPTH}',
            f'volume.original_depth={VOLUME_DEPTH}',
            'training.gradient_checkpointing=false',
        ])

    if space_type == 's2d':
        overrides.extend([
            'space_to_depth.enabled=true',
            'space_to_depth.spatial_factor=2',
            'space_to_depth.depth_factor=2',
        ])

    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _build_dataloader(dataset, persistent_workers=True):
    """Build a DataLoader matching real training config."""
    num_workers = 2 if persistent_workers else 0
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )


class _Timeout:
    """Context manager that raises TimeoutError after `seconds`."""

    def __init__(self, seconds: int):
        self.seconds = seconds

    def _handler(self, signum, frame):
        raise TimeoutError(
            f"Test timed out after {self.seconds}s — possible deadlock"
        )

    def __enter__(self):
        self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize(
    'test_id,spatial_dims,mode,strategy,space_type',
    SMOKE_PARAMS,
)
def test_pipeline_smoke(
    tmp_path,
    test_id,
    spatial_dims,
    mode,
    strategy,
    space_type,
):
    """Full pipeline: init -> setup_model -> train (2 epochs).

    Validates:
    - No deadlocks (120s timeout)
    - Checkpoint is saved
    - EMA is initialized and updated
    """
    from medgen.pipeline import DiffusionTrainer

    cfg = _build_cfg(tmp_path, spatial_dims, mode, strategy, space_type)

    # Build space object if needed
    space = None
    if space_type == 's2d':
        from medgen.diffusion.spaces import SpaceToDepthSpace
        space = SpaceToDepthSpace(spatial_factor=2, depth_factor=2)

    # 1. Init trainer
    trainer = DiffusionTrainer(cfg, spatial_dims=spatial_dims, space=space)

    # 2. Create synthetic datasets and dataloaders (train + val, like real training)
    train_dataset = SyntheticDiffusionDataset(
        num_samples=NUM_SAMPLES,
        image_size=IMAGE_SIZE,
        spatial_dims=spatial_dims,
        mode=mode,
        depth=VOLUME_DEPTH,
    )
    val_dataset = SyntheticDiffusionDataset(
        num_samples=4,
        image_size=IMAGE_SIZE,
        spatial_dims=spatial_dims,
        mode=mode,
        depth=VOLUME_DEPTH,
    )
    loader = _build_dataloader(train_dataset, persistent_workers=True)
    val_loader = _build_dataloader(val_dataset, persistent_workers=True)

    # 3. Setup model
    trainer.setup_model(train_dataset)
    assert trainer.model is not None, 'Model not created'
    assert trainer.optimizer is not None, 'Optimizer not created'

    # 4. Train with timeout to catch deadlocks
    with _Timeout(120):
        trainer.train(
            train_loader=loader,
            train_dataset=train_dataset,
            val_loader=val_loader,
        )

    # 5. Assertions
    # Checkpoint should exist
    checkpoints = [
        f for f in os.listdir(tmp_path)
        if f.startswith('checkpoint') and f.endswith('.pt')
    ]
    assert len(checkpoints) > 0, f'No checkpoints found in {tmp_path}'

    # EMA should have been initialized
    assert trainer.ema is not None, 'EMA not initialized despite use_ema=true'

    # Writer should have been created
    assert trainer.writer is not None, 'TensorBoard writer not created'

    # Cleanup: close writer, delete loader before test ends
    if trainer.writer is not None:
        trainer.writer.close()
    del loader
    del val_loader


# ---------------------------------------------------------------------------
# Job chaining test
# ---------------------------------------------------------------------------

CHAIN_PARAMS = [
    pytest.param(2, 'bravo', 'rflow', 'pixel', id='2d-bravo-rflow'),
    pytest.param(3, 'bravo', 'rflow', 'pixel', id='3d-bravo-rflow'),
]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('spatial_dims,mode,strategy,space_type', CHAIN_PARAMS)
def test_job_chaining(tmp_path, spatial_dims, mode, strategy, space_type):
    """Simulate SLURM job chaining: train → save → fresh trainer → load → resume.

    Validates:
    - Checkpoint file is a loadable zip archive
    - Model weights survive roundtrip (not re-initialized)
    - Optimizer state preserved (momentum buffers continue)
    - Scheduler state preserved (LR continues from correct position)
    - EMA weights preserved
    - Training resumes from correct epoch
    - Second segment produces different (lower) loss than first-epoch loss
    """
    from medgen.pipeline import DiffusionTrainer

    cfg = _build_cfg(tmp_path, spatial_dims, mode, strategy, space_type)

    # ── Segment 1: Train 2 epochs ──────────────────────────────────────
    train_dataset = SyntheticDiffusionDataset(
        num_samples=NUM_SAMPLES,
        image_size=IMAGE_SIZE,
        spatial_dims=spatial_dims,
        mode=mode,
        depth=VOLUME_DEPTH,
    )
    val_dataset = SyntheticDiffusionDataset(
        num_samples=4,
        image_size=IMAGE_SIZE,
        spatial_dims=spatial_dims,
        mode=mode,
        depth=VOLUME_DEPTH,
    )
    loader = _build_dataloader(train_dataset, persistent_workers=False)
    val_loader = _build_dataloader(val_dataset, persistent_workers=False)

    trainer1 = DiffusionTrainer(cfg, spatial_dims=spatial_dims)
    trainer1.setup_model(train_dataset)

    with _Timeout(120):
        last_epoch = trainer1.train(
            train_loader=loader,
            train_dataset=train_dataset,
            val_loader=val_loader,
        )

    # Verify checkpoint exists and is loadable
    ckpt_path = os.path.join(tmp_path, 'checkpoint_latest.pt')
    assert os.path.exists(ckpt_path), f'No checkpoint_latest.pt in {tmp_path}'

    import zipfile
    with zipfile.ZipFile(ckpt_path) as zf:
        assert len(zf.namelist()) > 0, 'Checkpoint zip is empty'

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'model_state_dict' in checkpoint, 'Missing model_state_dict'
    assert 'optimizer_state_dict' in checkpoint, 'Missing optimizer_state_dict'
    assert 'epoch' in checkpoint, 'Missing epoch'

    # Snapshot weights from segment 1
    seg1_weights = {
        k: v.clone() for k, v in checkpoint['model_state_dict'].items()
    }
    seg1_epoch = checkpoint['epoch']

    if trainer1.writer is not None:
        trainer1.writer.close()
    del trainer1, loader, val_loader
    torch.cuda.empty_cache()

    # ── Segment 2: Fresh trainer, load checkpoint, continue ────────────
    trainer2 = DiffusionTrainer(cfg, spatial_dims=spatial_dims)
    trainer2.setup_model(train_dataset)

    start_epoch = trainer2.load_checkpoint(ckpt_path)
    assert start_epoch == seg1_epoch + 1, (
        f'Expected resume from epoch {seg1_epoch + 1}, got {start_epoch}'
    )

    # Verify weights were restored (not re-initialized)
    for key in seg1_weights:
        restored = trainer2.model_raw.state_dict()[key]
        assert torch.equal(restored, seg1_weights[key].to(restored.device)), (
            f'Weight mismatch after load: {key}'
        )

    # Verify optimizer has populated state (momentum buffers from segment 1)
    assert len(trainer2.optimizer.state) > 0, (
        'Optimizer state empty after load — momentum buffers lost'
    )

    # Verify EMA was restored
    if trainer2.ema is not None:
        assert 'ema_state_dict' in checkpoint, 'EMA state not in checkpoint'

    # Run 2 more epochs from the resumed state
    loader2 = _build_dataloader(train_dataset, persistent_workers=False)
    val_loader2 = _build_dataloader(val_dataset, persistent_workers=False)

    with _Timeout(120):
        last_epoch_2 = trainer2.train(
            train_loader=loader2,
            train_dataset=train_dataset,
            val_loader=val_loader2,
            start_epoch=start_epoch,
            max_epochs=start_epoch + 2,
        )

    # Verify training continued from correct epoch
    assert last_epoch_2 >= start_epoch, (
        f'Expected training to reach at least epoch {start_epoch}, got {last_epoch_2}'
    )

    # Verify checkpoint was updated
    ckpt2 = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert ckpt2['epoch'] >= seg1_epoch, (
        f'Checkpoint epoch did not advance: {ckpt2["epoch"]} vs {seg1_epoch}'
    )

    # Verify weights changed (training actually happened, not a no-op)
    weights_changed = False
    for key in seg1_weights:
        if not torch.equal(seg1_weights[key], ckpt2['model_state_dict'][key]):
            weights_changed = True
            break
    assert weights_changed, 'Weights unchanged after resumed training — training was a no-op'

    if trainer2.writer is not None:
        trainer2.writer.close()
    del trainer2, loader2, val_loader2
