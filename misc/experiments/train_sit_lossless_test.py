#!/usr/bin/env python
"""Test DiT training with lossless mask conditioning on 8x8x32 latent space.

Uses REAL segmentation masks from BrainMetShare dataset.

Usage:
    python scripts/train_dit_lossless_test.py
    python scripts/train_dit_lossless_test.py --epochs 100
"""

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from medgen.models.dit import create_dit
from medgen.data.lossless_mask_codec import encode_mask_lossless, decode_mask_lossless


class LosslessMaskEncoder(nn.Module):
    """Encode lossless mask latent to valid float representation for NN."""

    def __init__(self, latent_channels: int = 32, hidden_channels: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
        )

    def forward(self, mask_latent: torch.Tensor) -> torch.Tensor:
        int_latent = mask_latent.view(torch.int32)
        normalized = int_latent.float() / 2147483648.0
        return self.proj(normalized)


class RealMaskDataset(Dataset):
    """Dataset of real segmentation masks from BrainMetShare."""

    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir) / split

        print(f'Loading real masks from {self.data_dir}...')

        # Collect all positive mask slices
        self.masks = []
        self.mask_latents = []
        self.latents = []  # Synthetic image latents (placeholder)

        patients = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        print(f'Found {len(patients)} patients')

        for patient_dir in tqdm(patients, desc='Loading patients'):
            seg_path = patient_dir / 'seg.nii.gz'
            if not seg_path.exists():
                continue

            # Load segmentation volume
            seg_vol = nib.load(seg_path).get_fdata()

            # Extract positive slices (slices with tumor)
            for z in range(seg_vol.shape[2]):
                slice_2d = seg_vol[:, :, z]

                # Skip empty slices
                if slice_2d.sum() == 0:
                    continue

                # Resize to 256x256 if needed
                if slice_2d.shape != (256, 256):
                    slice_2d = self._resize(slice_2d, (256, 256))

                # Binarize
                mask = (slice_2d > 0).astype(np.float32)
                mask_tensor = torch.from_numpy(mask)

                # Encode losslessly
                mask_latent = encode_mask_lossless(mask_tensor, 'f32')

                # Create synthetic image latent (placeholder - in real use: DC-AE encoded)
                latent = torch.randn(32, 8, 8)

                self.masks.append(mask_tensor)
                self.mask_latents.append(mask_latent)
                self.latents.append(latent)

        # Stack all
        self.masks = torch.stack(self.masks)
        self.mask_latents = torch.stack(self.mask_latents)
        self.latents = torch.stack(self.latents)

        print(f'Loaded {len(self.masks)} positive mask slices')

        # Stats
        mask_sums = self.masks.sum(dim=(1, 2))
        print(f'Mask pixel counts: min={mask_sums.min():.0f}, max={mask_sums.max():.0f}, mean={mask_sums.mean():.0f}')

    def _resize(self, img: np.ndarray, size: tuple) -> np.ndarray:
        """Simple resize using torch interpolation."""
        tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=size, mode='nearest')
        return resized.squeeze().numpy()

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        return {
            'latent': self.latents[idx],
            'mask': self.masks[idx],
            'mask_latent': self.mask_latents[idx],
        }


def main():
    parser = argparse.ArgumentParser(description='Test DiT with lossless mask conditioning')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--variant', type=str, default='S', choices=['S', 'B', 'L'], help='DiT variant')
    parser.add_argument('--data_dir', type=str,
                        default='/home/mode/NTNU/MedicalDataSets/brainmetshare-3',
                        help='Path to BrainMetShare dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print()

    # Create dataset with real masks
    dataset = RealMaskDataset(args.data_dir, split='train')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print(f'Batches per epoch: {len(dataloader)}')
    print()

    # Create models
    mask_encoder = LosslessMaskEncoder(latent_channels=32, hidden_channels=32).to(device)

    model = create_dit(
        variant=args.variant,
        spatial_dims=2,
        input_size=8,
        patch_size=2,
        in_channels=64,  # 32 image + 32 mask
        conditioning='concat',
    ).to(device)

    sit_params = sum(p.numel() for p in model.parameters())
    enc_params = sum(p.numel() for p in mask_encoder.parameters())
    print(f'=== DiT-{args.variant} with Lossless Mask Conditioning ===')
    print(f'DiT parameters: {sit_params/1e6:.2f}M')
    print(f'Mask encoder parameters: {enc_params/1e3:.1f}K')
    print(f'Tokens: {model.num_patches}')
    print()

    # Optimizer
    all_params = list(model.parameters()) + list(mask_encoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f'Training for {args.epochs} epochs...')
    print()

    model.train()
    mask_encoder.train()

    best_loss = float('inf')
    epoch_losses = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False)
        for batch in pbar:
            x0 = batch['latent'].to(device)
            mask_latent_raw = batch['mask_latent'].to(device)

            # Encode mask to valid features
            mask_features = mask_encoder(mask_latent_raw)

            # Sample noise and timesteps
            noise = torch.randn_like(x0)
            t = torch.rand(x0.shape[0], device=device)

            # Flow matching interpolation
            t_expand = t.view(-1, 1, 1, 1)
            xt = (1 - t_expand) * x0 + t_expand * noise

            # Target velocity
            v_target = noise - x0

            # Concatenate and predict
            model_input = torch.cat([xt, mask_features], dim=1)
            v_pred_full = model(model_input, t)
            v_pred = v_pred_full[:, :32]

            # Loss
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        # End of epoch
        avg_loss = epoch_loss / num_batches
        epoch_losses.append(avg_loss)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1:3d}: loss={avg_loss:.4f}, best={best_loss:.4f}, lr={lr:.2e}')

    print()
    print('=== Training Complete ===')
    print(f'Final loss: {epoch_losses[-1]:.4f}')
    print(f'Best loss: {best_loss:.4f}')
    print(f'Loss reduction: {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}')

    # Inference test
    print()
    print('=== Inference Test ===')
    model.eval()
    mask_encoder.eval()

    with torch.no_grad():
        # Use a mask from the dataset
        test_mask = dataset.masks[0].to(device)
        test_mask_latent = dataset.mask_latents[0].unsqueeze(0).to(device)
        test_mask_features = mask_encoder(test_mask_latent)

        # Verify lossless
        decoded = decode_mask_lossless(test_mask_latent[0].cpu(), 'f32')
        is_lossless = torch.equal(test_mask.cpu(), decoded)
        print(f'Mask encoding is lossless: {is_lossless}')
        print(f'Test mask tumor pixels: {test_mask.sum():.0f}')

        # Sample from noise
        x = torch.randn(1, 32, 8, 8, device=device)

        # Euler sampling (20 steps)
        steps = 20
        for i in range(steps):
            t_val = 1.0 - i / steps
            t = torch.tensor([t_val], device=device)

            model_input = torch.cat([x, test_mask_features], dim=1)
            v = model(model_input, t)[:, :32]
            x = x - v / steps

        print(f'Generated latent shape: {x.shape}')
        print(f'Generated latent range: [{x.min():.3f}, {x.max():.3f}]')

    print()
    print('Done!')


if __name__ == '__main__':
    main()
