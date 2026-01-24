"""
Step count comparison study for diffusion schedulers.

Compares DDPM and RFlow schedulers across different step counts
for both mask and bravo image generation.
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, RFlowScheduler
from torch.amp import autocast

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import PathConfig

torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch._dynamo.config.cache_size_limit = 32

device = torch.device("cuda")

# Configuration
STEP_COUNTS = [1000, 500, 250, 100, 50, 30, 10, 5]
IMAGE_SIZE = 128
COMPUTE = "local"  # Change to "cluster" if needed
OUTPUT_DIR = "step_comparison_study"

# Model paths
RFLOW_SEG_MODEL = "RFlow_seg_128_20250919-132722/Epoch499_of_500"
RFLOW_BRAVO_MODEL = "RFlow_bravo_128_20250919-013949/Epoch499_of_500"
DDPM_SEG_MODEL = "Diffusion_seg_128_20250919-223333/Epoch499_of_500"
DDPM_BRAVO_MODEL = "Diffusion_bravo_128_20250919-101308/Epoch499_of_500"


def get_vram_usage() -> str:
    """Get current VRAM usage statistics.

    Returns:
        Formatted string with VRAM allocation statistics.
    """
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
    return f"VRAM: total:{total:.1f}GB, allocated:{allocated:.1f}GB, reserved:{reserved:.1f}GB)"


def load_model(model_path: str, model_input: int) -> torch.nn.Module:
    """Load trained diffusion model.

    Args:
        model_path: Path to saved model weights.
        model_input: Number of input channels.

    Returns:
        Compiled diffusion model in evaluation mode.
    """
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=model_input,
        out_channels=1,
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256,
    )

    pre_trained_model = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(pre_trained_model, strict=False)
    model.to(device)
    model.eval()
    opt_model = torch.compile(model, mode="max-autotune")
    return opt_model


def generate_rflow_mask(model: torch.nn.Module, noise: torch.Tensor, steps: int) -> torch.Tensor:
    """Generate mask using RFlow scheduler.

    Args:
        model: Trained diffusion model.
        noise: Initial noise tensor.
        steps: Number of inference steps.

    Returns:
        Generated mask tensor.
    """
    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=True,
        sample_method='logit-normal',
        use_timestep_transform=True,
        base_img_size_numel=IMAGE_SIZE * IMAGE_SIZE,
        spatial_dim=2
    )

    input_img_size_numel = IMAGE_SIZE * IMAGE_SIZE
    scheduler.set_timesteps(
        num_inference_steps=steps,
        device=device,
        input_img_size_numel=input_img_size_numel
    )

    all_next_timesteps = torch.cat((
        scheduler.timesteps[1:],
        torch.tensor([0], dtype=scheduler.timesteps.dtype, device=device)
    ))

    current_image = noise.clone()

    with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            for t, next_t in zip(scheduler.timesteps, all_next_timesteps):
                timesteps_batch = t.unsqueeze(0).to(device)
                velocity_pred = model(current_image, timesteps=timesteps_batch)
                next_timestep = next_t.to(device) if isinstance(next_t, torch.Tensor) else torch.tensor(next_t,
                                                                                                        device=device)
                current_image, _ = scheduler.step(velocity_pred, t, current_image, next_timestep)

    return current_image


def generate_ddpm_mask(model: torch.nn.Module, noise: torch.Tensor, steps: int) -> torch.Tensor:
    """Generate mask using DDPM scheduler.

    Args:
        model: Trained diffusion model.
        noise: Initial noise tensor.
        steps: Number of inference steps.

    Returns:
        Generated mask tensor.
    """
    scheduler = DDPMScheduler(num_train_timesteps=steps, schedule='cosine')
    inferer = DiffusionInferer(scheduler)

    with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            image = inferer.sample(input_noise=noise.clone(), diffusion_model=model, scheduler=scheduler, verbose=False)

    return image


def generate_rflow_bravo(
    model: torch.nn.Module, mask_tensor: torch.Tensor, noise: torch.Tensor, steps: int
) -> torch.Tensor:
    """Generate bravo image using RFlow scheduler.

    Args:
        model: Trained diffusion model.
        mask_tensor: Conditioning mask tensor.
        noise: Initial noise tensor.
        steps: Number of inference steps.

    Returns:
        Generated bravo image tensor.
    """
    scheduler = RFlowScheduler(
        num_train_timesteps=1000,
        use_discrete_timesteps=True,
        sample_method='logit-normal',
        use_timestep_transform=True,
        base_img_size_numel=IMAGE_SIZE * IMAGE_SIZE,
        spatial_dim=2
    )

    input_img_size_numel = IMAGE_SIZE * IMAGE_SIZE
    scheduler.set_timesteps(
        num_inference_steps=steps,
        device=device,
        input_img_size_numel=input_img_size_numel
    )

    all_next_timesteps = torch.cat((
        scheduler.timesteps[1:],
        torch.tensor([0], dtype=scheduler.timesteps.dtype, device=device)
    ))

    current_image = noise.clone()

    with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            for t, next_t in zip(scheduler.timesteps, all_next_timesteps):
                timesteps_batch = t.unsqueeze(0).to(device)
                combined = torch.cat((current_image, mask_tensor), dim=1)
                velocity_pred = model(combined, timesteps=timesteps_batch)
                next_timestep = next_t.to(device) if isinstance(next_t, torch.Tensor) else torch.tensor(next_t,
                                                                                                        device=device)
                current_image, _ = scheduler.step(velocity_pred, t, current_image, next_timestep)

    return current_image


def generate_ddpm_bravo(
    model: torch.nn.Module, mask_tensor: torch.Tensor, noise: torch.Tensor, steps: int
) -> torch.Tensor:
    """Generate bravo image using DDPM scheduler.

    Args:
        model: Trained diffusion model.
        mask_tensor: Conditioning mask tensor.
        noise: Initial noise tensor.
        steps: Number of inference steps.

    Returns:
        Generated bravo image tensor.
    """
    scheduler = DDPMScheduler(num_train_timesteps=steps, schedule='cosine')

    current_image = noise.clone()
    scheduler.set_timesteps(num_inference_steps=steps)

    with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        with torch.no_grad():
            for t in scheduler.timesteps:
                combined = torch.cat((current_image, mask_tensor), dim=1)
                timesteps_batch = torch.full((1,), t, device=device, dtype=torch.long)
                prediction_t = model(combined, timesteps=timesteps_batch)
                current_image, _ = scheduler.step(prediction_t, t, current_image)

    return current_image


def normalize_for_display(tensor: torch.Tensor) -> np.ndarray:
    """Normalize tensor to [0, 255] for display.

    Args:
        tensor: Input tensor to normalize.

    Returns:
        Normalized uint8 numpy array.
    """
    img = tensor.squeeze().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)


def create_mask_comparison_grid(
    mask_results: Dict[str, torch.Tensor], mask_noise_seeds: List[torch.Tensor]
) -> plt.Figure:
    """Create mask generation comparison grid at different step counts.

    Args:
        mask_results: Dictionary mapping result keys to mask tensors.
        mask_noise_seeds: List of noise tensors used as seeds.

    Returns:
        Matplotlib figure with comparison grid.
    """
    step_labels = ["Noise"] + [str(s) for s in STEP_COUNTS]

    # Grid dimensions: 6 rows × 9 columns (noise + 8 steps)
    rows = 6  # 3 seeds × 2 schedulers
    cols = 9  # noise + 8 step counts

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    fig.suptitle("Mask Generation at Different Step Counts", fontsize=16, fontweight='bold')

    # Add column headers
    for col, label in enumerate(step_labels):
        axes[0, col].set_title(label, fontsize=12, fontweight='bold')

    row_idx = 0
    for seed_idx in range(3):
        for scheduler_idx, scheduler in enumerate(["DDPM", "RFlow"]):
            # First column: show noise only in middle of each seed pair (rows 0, 2, 4)
            if scheduler_idx == 0:  # DDPM row - show noise
                noise_img = normalize_for_display(mask_noise_seeds[seed_idx])
                axes[row_idx, 0].imshow(noise_img, cmap='gray')
            else:  # RFlow row - leave empty
                axes[row_idx, 0].set_facecolor('white')
            axes[row_idx, 0].axis('off')

            # Subsequent columns: show generated masks
            for step_idx, steps in enumerate(STEP_COUNTS):
                col_idx = step_idx + 1

                if scheduler == "DDPM":
                    mask_img = normalize_for_display(mask_results[f"ddpm_seed{seed_idx}_{steps}"])
                else:  # RFlow
                    mask_img = normalize_for_display(mask_results[f"rflow_seed{seed_idx}_{steps}"])

                axes[row_idx, col_idx].imshow(mask_img, cmap='gray')
                axes[row_idx, col_idx].axis('off')

            row_idx += 1

    # Add scheduler labels on the right side
    for seed_idx in range(3):
        ddpm_row = seed_idx * 2
        rflow_row = seed_idx * 2 + 1

        # Add DDPM label - positioned to the right of the rightmost subplot
        axes[ddpm_row, -1].text(1.05, 0.5, 'DDPM', fontsize=12, fontweight='bold',
                                transform=axes[ddpm_row, -1].transAxes, rotation=270,
                                va='center', ha='left')

        # Add RFlow label
        axes[rflow_row, -1].text(1.05, 0.5, 'RFlow', fontsize=12, fontweight='bold',
                                 transform=axes[rflow_row, -1].transAxes, rotation=270,
                                 va='center', ha='left')

    plt.tight_layout()
    return fig


def create_bravo_comparison_grid(
    bravo_results: Dict[str, torch.Tensor],
    reference_masks: List[torch.Tensor],
    bravo_noise_seeds: List[torch.Tensor]
) -> plt.Figure:
    """Create bravo generation comparison grid at different step counts.

    Args:
        bravo_results: Dictionary mapping result keys to bravo tensors.
        reference_masks: List of reference mask tensors.
        bravo_noise_seeds: List of noise tensors used as seeds.

    Returns:
        Matplotlib figure with comparison grid.
    """
    step_labels = ["Noise+Mask"] + [str(s) for s in STEP_COUNTS]

    # Grid dimensions: 6 rows × 9 columns (noise+mask + 8 steps)
    rows = 6  # 3 pairs × 2 schedulers
    cols = 9  # noise+mask + 8 step counts

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    fig.suptitle("Bravo Generation at Different Step Counts", fontsize=16, fontweight='bold')

    # Add column headers
    for col, label in enumerate(step_labels):
        axes[0, col].set_title(label, fontsize=12, fontweight='bold')

    row_idx = 0
    for pair_idx in range(3):
        for scheduler_idx, scheduler in enumerate(["DDPM", "RFlow"]):
            # First column: show noise in DDPM row, mask in RFlow row
            if scheduler_idx == 0:  # DDPM row - show noise
                noise_img = normalize_for_display(bravo_noise_seeds[pair_idx])
                axes[row_idx, 0].imshow(noise_img, cmap='gray')
            else:  # RFlow row - show mask
                mask_img = normalize_for_display(reference_masks[pair_idx])
                axes[row_idx, 0].imshow(mask_img, cmap='gray')
            axes[row_idx, 0].axis('off')

            # Subsequent columns: show generated bravo images
            for step_idx, steps in enumerate(STEP_COUNTS):
                col_idx = step_idx + 1

                if scheduler == "DDPM":
                    bravo_img = normalize_for_display(bravo_results[f"ddpm_pair{pair_idx}_{steps}"])
                else:  # RFlow
                    bravo_img = normalize_for_display(bravo_results[f"rflow_pair{pair_idx}_{steps}"])

                axes[row_idx, col_idx].imshow(bravo_img, cmap='gray')
                axes[row_idx, col_idx].axis('off')

            row_idx += 1

    # Add scheduler labels on the right side (same approach as mask comparison)
    for pair_idx in range(3):
        ddpm_row = pair_idx * 2
        rflow_row = pair_idx * 2 + 1

        # Add DDPM label - positioned to the right of the rightmost subplot
        axes[ddpm_row, -1].text(1.05, 0.5, 'DDPM', fontsize=12, fontweight='bold',
                                transform=axes[ddpm_row, -1].transAxes, rotation=270,
                                va='center', ha='left')

        # Add RFlow label
        axes[rflow_row, -1].text(1.05, 0.5, 'RFlow', fontsize=12, fontweight='bold',
                                 transform=axes[rflow_row, -1].transAxes, rotation=270,
                                 va='center', ha='left')

    plt.tight_layout()
    return fig


def main() -> None:
    """Main entry point for step count comparison study."""
    print(f"Starting step count comparison study | {get_vram_usage()}")
    print(f"Step counts: {STEP_COUNTS}")
    print(
        f"Mask generation: 3 seeds × {len(STEP_COUNTS)} steps × 2 schedulers = {3 * len(STEP_COUNTS) * 2} images")
    print(
        f"Bravo generation: 3 pairs × {len(STEP_COUNTS)} steps × 2 schedulers = {3 * len(STEP_COUNTS) * 2} images")

    # Set up paths using PathConfig
    path_config = PathConfig(compute=COMPUTE)
    prefix = str(path_config.base_prefix)

    # Create output directory
    save_dir = f"{prefix}/MedicalDataSets/{OUTPUT_DIR}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Load models
    print("\nLoading models...")
    rflow_seg_path = f"{prefix}/AIS4005_IP/trained_model/{RFLOW_SEG_MODEL}"
    rflow_bravo_path = f"{prefix}/AIS4005_IP/trained_model/{RFLOW_BRAVO_MODEL}"
    ddpm_seg_path = f"{prefix}/AIS4005_IP/trained_model/{DDPM_SEG_MODEL}"
    ddpm_bravo_path = f"{prefix}/AIS4005_IP/trained_model/{DDPM_BRAVO_MODEL}"

    rflow_seg_model = load_model(rflow_seg_path, model_input=1)
    rflow_bravo_model = load_model(rflow_bravo_path, model_input=2)
    ddpm_seg_model = load_model(ddpm_seg_path, model_input=1)
    ddpm_bravo_model = load_model(ddpm_bravo_path, model_input=2)
    print("All models loaded successfully")

    # Generate fixed noise seeds
    torch.manual_seed(42)
    mask_noise_seeds = [
        torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
        for _ in range(3)
    ]

    torch.manual_seed(123)
    bravo_noise_seeds = [
        torch.randn((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=device)
        for _ in range(3)
    ]

    print("\n=== MASK GENERATION COMPARISON ===")

    # Storage for all results
    mask_results = {}
    bravo_results = {}
    reference_masks = []

    # Generate masks at different step counts
    for seed_idx, noise in enumerate(mask_noise_seeds):
        print(f"\nProcessing mask seed {seed_idx + 1}/3...")

        for steps in STEP_COUNTS:
            print(f"  Generating masks at {steps} steps...")

            # RFlow mask
            start_time = time.time()
            rflow_mask = generate_rflow_mask(rflow_seg_model, noise, steps)
            rflow_time = time.time() - start_time
            mask_results[f"rflow_seed{seed_idx}_{steps}"] = rflow_mask.clone()

            # DDPM mask
            start_time = time.time()
            ddpm_mask = generate_ddpm_mask(ddpm_seg_model, noise, steps)
            ddpm_time = time.time() - start_time
            mask_results[f"ddpm_seed{seed_idx}_{steps}"] = ddpm_mask.clone()

            print(f"    RFlow: {rflow_time:.2f}s | DDPM: {ddpm_time:.2f}s")

            # Store high-quality masks (1000 steps) for bravo generation
            if steps == 1000:
                if seed_idx < 2:  # Use first 2 seeds
                    reference_masks.append(rflow_mask.clone())
                elif seed_idx == 2:
                    reference_masks.append(ddpm_mask.clone())

            torch.cuda.empty_cache()

    print("\n=== BRAVO GENERATION COMPARISON ===")

    # Generate bravo images
    for pair_idx in range(3):
        reference_mask = reference_masks[pair_idx]
        bravo_noise = bravo_noise_seeds[pair_idx]
        print(f"\nProcessing pair {pair_idx + 1}/3...")

        for steps in STEP_COUNTS:
            print(f"  Generating bravo images at {steps} steps...")

            # RFlow bravo
            start_time = time.time()
            rflow_bravo = generate_rflow_bravo(rflow_bravo_model, reference_mask, bravo_noise, steps)
            rflow_time = time.time() - start_time
            bravo_results[f"rflow_pair{pair_idx}_{steps}"] = rflow_bravo.clone()

            # DDPM bravo
            start_time = time.time()
            ddpm_bravo = generate_ddpm_bravo(ddpm_bravo_model, reference_mask, bravo_noise, steps)
            ddpm_time = time.time() - start_time
            bravo_results[f"ddpm_pair{pair_idx}_{steps}"] = ddpm_bravo.clone()

            print(f"    RFlow: {rflow_time:.2f}s | DDPM: {ddpm_time:.2f}s")

            torch.cuda.empty_cache()

    print("\n=== GENERATING GRID IMAGES ===")

    # Create and save mask comparison grid
    print("Creating mask comparison grid...")
    mask_fig = create_mask_comparison_grid(mask_results, mask_noise_seeds)
    mask_path = f"{save_dir}/mask_comparison.png"
    mask_fig.savefig(mask_path, dpi=2000, bbox_inches='tight')
    plt.close(mask_fig)
    print(f"Mask comparison grid saved: {mask_path}")

    # Create and save bravo comparison grid
    print("Creating bravo comparison grid...")
    bravo_fig = create_bravo_comparison_grid(bravo_results, reference_masks, bravo_noise_seeds)
    bravo_path = f"{save_dir}/bravo_comparison.png"
    bravo_fig.savefig(bravo_path, dpi=2000, bbox_inches='tight')
    plt.close(bravo_fig)
    print(f"Bravo comparison grid saved: {bravo_path}")

    print("\n=== STUDY COMPLETE ===")
    print(f"Grid images saved to: {save_dir}")
    print(f"Final {get_vram_usage()}")


if __name__ == "__main__":
    main()
