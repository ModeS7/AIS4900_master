# Test Fixtures

This directory contains fixtures for end-to-end testing.

## Golden Checkpoint

The `golden_checkpoint/` directory contains pre-trained model checkpoints used for E2E testing.
These checkpoints are **NOT stored in git** (too large) - they must be generated locally or
downloaded from a shared location.

### Creating Golden Checkpoints

Run the helper script to create all necessary checkpoints:

```bash
./tests/fixtures/create_golden_checkpoint.sh
```

Or create them manually:

```bash
# Bravo mode checkpoint (diffusion model for brain MRI)
python -m medgen.scripts.train \
    mode=bravo \
    strategy=rflow \
    training.epochs=1 \
    training.batch_size=2 \
    training.limit_batches=10 \
    model.channels="[32,64]" \
    model.image_size=64 \
    hydra.run.dir=tests/fixtures/golden_checkpoint/bravo

# Seg mode checkpoint (segmentation generation)
python -m medgen.scripts.train \
    mode=seg \
    strategy=rflow \
    training.epochs=1 \
    training.batch_size=2 \
    training.limit_batches=10 \
    model.channels="[32,64]" \
    model.image_size=64 \
    hydra.run.dir=tests/fixtures/golden_checkpoint/seg

# VAE checkpoint (compression model)
python -m medgen.scripts.train_compression \
    --config-name=vae \
    mode=multi_modality \
    training.epochs=1 \
    training.batch_size=2 \
    training.limit_batches=10 \
    model.channels="[32,64]" \
    hydra.run.dir=tests/fixtures/golden_checkpoint/vae
```

### Expected Structure

After running the script, you should have:

```
tests/fixtures/golden_checkpoint/
├── bravo/
│   ├── checkpoint_best.pt
│   ├── checkpoint_latest.pt
│   └── events.out.tfevents.*
├── seg/
│   ├── checkpoint_best.pt
│   ├── checkpoint_latest.pt
│   └── events.out.tfevents.*
└── vae/
    ├── checkpoint_best.pt
    ├── checkpoint_latest.pt
    └── events.out.tfevents.*
```

### Test Behavior Without Checkpoints

E2E tests that require golden checkpoints will be **skipped** if the checkpoints
are not available. This allows:

- Running unit tests without the full setup
- Faster CI on PRs (E2E tests run nightly with GPU)
- Local development without training checkpoints

### Requirements

Creating golden checkpoints requires:
- GPU with at least 4GB VRAM (or CPU with patience)
- Dataset available at configured paths
- ~10-15 minutes per checkpoint

### Regenerating Checkpoints

If models or training code change significantly, regenerate checkpoints:

```bash
rm -rf tests/fixtures/golden_checkpoint/
./tests/fixtures/create_golden_checkpoint.sh
```

Note: Checkpoint format changes may require updating E2E tests as well.
