"""
Create comparison images between original and binary thresholded outputs.

Utility for visualizing network output quality by showing side-by-side
comparisons of original grayscale images and their binarized versions.
"""
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import PathConfig


def process_single_image(args: Tuple[Path, str, float]) -> str:
    """Process a single image - creates composite with original + fixed threshold.

    Args:
        args: Tuple of (image_path, output_directory, threshold_value).

    Returns:
        Status message indicating success or failure.
    """
    img_path, output_dir, threshold_value = args

    # Read image in grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return f"Failed to load {img_path}"

    # Original image
    original = img.copy()

    # Fixed threshold
    img_float = img.astype(np.float32) / 255.0
    fixed_binary = (img_float > threshold_value).astype(np.uint8) * 255

    # Create composite image (side by side)
    h, w = img.shape
    composite = np.zeros((h, w * 2), dtype=np.uint8)

    # Place images side by side
    composite[:, 0:w] = original        # Original
    composite[:, w:2*w] = fixed_binary  # Fixed threshold

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = 255
    thickness = 1

    # Add labels at the top of each section
    cv2.putText(composite, 'Original', (10, 25), font, font_scale, color, thickness)
    cv2.putText(composite, f'Binary {threshold_value}', (w + 10, 25), font, font_scale, color, thickness)

    # Create output filename and save
    output_filename = f"comparison_{img_path.stem}.png"
    output_filepath = Path(output_dir) / output_filename

    cv2.imwrite(str(output_filepath), composite)

    return f"Processed {img_path.name} -> {output_filename}"


def create_comparison_images(
    input_dir: str,
    output_dir: str,
    threshold_value: float = 0.5,
    num_workers: int = None
) -> None:
    """Create comparison images with original + fixed threshold.

    Args:
        input_dir: Directory containing input PNG images.
        output_dir: Directory to save comparison images.
        threshold_value: Fixed threshold value (0-1 range).
        num_workers: Number of parallel workers (None = auto-detect).
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all PNG files
    png_files = sorted(list(input_path.glob("*.png")))

    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(png_files))

    print(f"Found {len(png_files)} images to process")
    print(f"Creating comparisons: Original | Binary ({threshold_value})")
    print(f"Processing with {num_workers} workers")

    # Prepare arguments for parallel processing
    args_list = [(img_path, output_dir, threshold_value) for img_path in png_files]

    # Process images in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_image, args_list))

    # Print results
    for result in results:
        print(result)


if __name__ == "__main__":
    # Configuration using PathConfig
    path_config = PathConfig()
    input_directory = str(path_config.data_dir / "raw_network_output")
    output_directory = str(path_config.data_dir / "comparison_masks")

    # Create comparison images (PARALLEL)
    print("=== Creating comparison images (Original | Binary) ===")
    create_comparison_images(
        input_dir=input_directory,
        output_dir=output_directory,
        threshold_value=0.1,
        num_workers=4  # Adjust based on your CPU cores
    )

    print("\nComparison creation complete!")
    print(f"Check {output_directory} for comparison images.")