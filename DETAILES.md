# MedGen - Technical Details

Detailed technical documentation for the synthetic medical image generation framework.

## Project Structure

```
AIS4900_master/
├── pyproject.toml           # Package definition, install with pip install -e .
├── README.md                # User-facing documentation
├── DETAILES.md              # This file - detailed technical docs
├── FUTURE_WORK.md           # Planned improvements and experiments
├── requirements.txt         # Python dependencies
│
├── configs/                 # Hydra configuration files
│   ├── config.yaml          # Main config (composes others via defaults)
│   ├── model/
│   │   └── default.yaml     # Model architecture (override image_size via CLI)
│   ├── strategy/
│   │   ├── ddpm.yaml
│   │   └── rflow.yaml
│   ├── mode/
│   │   ├── seg.yaml         # Unconditional segmentation mask generation
│   │   ├── bravo.yaml       # Conditional single image (mask -> MRI)
│   │   └── dual.yaml        # Conditional dual image (mask -> T1 pre + post)
│   ├── training/
│   │   └── default.yaml
│   └── paths/
│       ├── local.yaml       # Local Linux development
│       └── cluster.yaml     # NTNU IDUN cluster
│
├── src/medgen/              # Main Python package
│   ├── core/
│   │   ├── constants.py     # Magic numbers, thresholds
│   │   └── paths.py         # Path utilities
│   ├── data/
│   │   ├── nifti_dataset.py # NiFTI volume loading, slice extraction
│   │   └── transforms.py    # MONAI transform pipelines
│   ├── diffusion/
│   │   ├── trainer.py       # DiffusionTrainer class
│   │   ├── strategies.py    # DDPM, RFlow strategies
│   │   ├── modes.py         # Seg, Bravo, Dual modes
│   │   ├── utils.py         # Checkpointing, logging
│   │   └── lr_finder.py     # Learning rate finder
│   └── scripts/
│       ├── train.py         # Training script (Hydra)
│       └── generate.py      # Inference script
│
└── IDUN/
    └── train_template.slurm # SLURM job template
```

## Architecture

### Strategy + Mode Pattern

The framework uses a **Strategy + Mode** design pattern:

- **Strategy** (HOW to diffuse): DDPM or Rectified Flow
- **Mode** (WHAT to generate): Segmentation, Bravo, or Dual

This gives 6 combinations (2 strategies x 3 modes).

### Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `DiffusionTrainer` | `src/medgen/diffusion/trainer.py` | Main training orchestrator |
| `DDPMStrategy` | `src/medgen/diffusion/strategies.py` | DDPM noise prediction |
| `RFlowStrategy` | `src/medgen/diffusion/strategies.py` | Rectified flow velocity |
| `SegmentationMode` | `src/medgen/diffusion/modes.py` | Unconditional mask gen |
| `ConditionalSingleMode` | `src/medgen/diffusion/modes.py` | Mask -> single image |
| `ConditionalDualMode` | `src/medgen/diffusion/modes.py` | Mask -> dual images |
| `NiFTIDataset` | `src/medgen/data/nifti_dataset.py` | NIfTI data loading |

## Hydra Configuration

Config files are in `configs/`. The main entry point is `configs/config.yaml`.

### How Hydra Works

1. `defaults:` list specifies which sub-configs to compose
2. CLI args override any value: `python -m medgen.scripts.train training.batch_size=32`
3. Hydra creates timestamped output dirs with config snapshots

### CLI Override Examples

```bash
# See resolved config without running
python -m medgen.scripts.train --cfg job

# Quick debug run
python -m medgen.scripts.train training.epochs=2 training.batch_size=4

# Change image size
python -m medgen.scripts.train model.image_size=256

# Disable training features
python -m medgen.scripts.train training.use_ema=false training.use_min_snr=false

# Full cluster training
python -m medgen.scripts.train paths=cluster strategy=rflow training.epochs=1000
```

## Data

- **Dataset**: BrainMetShare (brain MRI with metastasis)
- **Format**: NIfTI 3D volumes (.nii.gz)
- **Modalities**: bravo, t1_pre, t1_gd, flair, seg (segmentation mask)
- **Processing**: 3D volumes -> 2D slice extraction

## Training Features

- Distributed training (DDP) for multi-GPU
- Mixed precision (bfloat16)
- EMA (Exponential Moving Average)
- Min-SNR loss weighting
- Perceptual loss (RadImageNet)
- torch.compile optimization
- TensorBoard logging

## Important Paths

| Environment | Data | Models |
|------------|------|--------|
| Local | `/home/mode/NTNU/MedicalDataSets/brainmetshare-3` | `./runs/` |
| Cluster | `/cluster/work/modestas/MedicalDataSets/brainmetshare-3` | `/cluster/work/modestas/AIS4900_master/runs` |

## Extending the Framework

### Adding a new strategy
1. Create class extending `DiffusionStrategy` in `strategies.py`
2. Add config file in `configs/strategy/`
3. Register in strategy factory

### Adding a new mode
1. Create class extending `TrainingMode` in `modes.py`
2. Add config file in `configs/mode/`
3. Register in mode factory

### Running on IDUN cluster
```bash
sbatch IDUN/train_template.slurm
```
