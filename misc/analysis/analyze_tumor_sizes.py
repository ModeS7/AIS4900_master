"""
Analyze tumor size distribution in the dataset for conditioning bin design.

Computes:
- Feret diameter (longest axis) for each tumor
- Tumor count per slice
- Distribution across proposed bin schemes

Usage:
    python scripts/analyze_tumor_sizes.py --data_dir /path/to/brainmetshare-3
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage
from tqdm import tqdm


def compute_feret_diameter(binary_mask: np.ndarray, pixel_spacing_mm: float = 0.9375) -> float:
    """Compute Feret diameter (longest axis) of a binary region.

    Args:
        binary_mask: 2D binary mask of a single connected component.
        pixel_spacing_mm: Size of one pixel in mm (default: 240mm FOV / 256px).

    Returns:
        Feret diameter in mm.
    """
    # Find contour points
    coords = np.argwhere(binary_mask)
    if len(coords) < 2:
        return pixel_spacing_mm  # Single pixel

    # Compute pairwise distances and find maximum
    from scipy.spatial.distance import pdist
    if len(coords) > 1000:
        # Subsample for large regions
        idx = np.random.choice(len(coords), 1000, replace=False)
        coords = coords[idx]

    distances = pdist(coords)
    max_dist_pixels = distances.max() if len(distances) > 0 else 1

    return max_dist_pixels * pixel_spacing_mm


def analyze_slice(seg_mask: np.ndarray, pixel_spacing_mm: float = 0.9375) -> Tuple[int, List[float]]:
    """Analyze a single segmentation slice.

    Args:
        seg_mask: 2D binary segmentation mask.
        pixel_spacing_mm: Pixel spacing in mm.

    Returns:
        Tuple of (tumor_count, list of Feret diameters in mm).
    """
    # Label connected components
    labeled, num_features = ndimage.label(seg_mask > 0.5)

    if num_features == 0:
        return 0, []

    diameters = []
    for i in range(1, num_features + 1):
        component_mask = labeled == i
        diameter = compute_feret_diameter(component_mask, pixel_spacing_mm)
        diameters.append(diameter)

    return num_features, diameters


def load_seg_slices(data_dir: str, split: str = 'train') -> List[np.ndarray]:
    """Load all segmentation slices from the dataset.

    Args:
        data_dir: Path to brainmetshare-3 dataset.
        split: 'train', 'val', or 'test'.

    Returns:
        List of 2D segmentation masks.
    """
    import nibabel as nib

    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        print(f"Warning: {split_dir} does not exist")
        return []

    slices = []
    patients = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])

    for patient in tqdm(patients, desc=f"Loading {split}"):
        patient_dir = os.path.join(split_dir, patient)

        # Try NIfTI format first
        seg_file = os.path.join(patient_dir, 'seg.nii.gz')
        if not os.path.exists(seg_file):
            seg_file = os.path.join(patient_dir, 'seg.nii')
        if not os.path.exists(seg_file):
            seg_file = os.path.join(patient_dir, 'seg.npy')

        if os.path.exists(seg_file):
            if seg_file.endswith('.npy'):
                seg_volume = np.load(seg_file)
            else:
                # NIfTI format
                nii = nib.load(seg_file)
                seg_volume = nii.get_fdata()

            # seg_volume shape: [H, W, D] for NIfTI
            if seg_volume.ndim == 3:
                # Iterate over slices (last dimension for NIfTI)
                for i in range(seg_volume.shape[2]):
                    slice_2d = seg_volume[:, :, i]
                    if slice_2d.sum() > 0:  # Only positive slices
                        slices.append(slice_2d)

    return slices


def bin_sizes(diameters: List[float], bin_edges: List[float]) -> Dict[str, int]:
    """Bin diameters into size categories.

    Args:
        diameters: List of tumor diameters in mm.
        bin_edges: List of bin edges (e.g., [0, 5, 10, 15, ...]).

    Returns:
        Dictionary mapping bin labels to counts.
    """
    counts = defaultdict(int)

    for d in diameters:
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= d < bin_edges[i + 1]:
                label = f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}mm"
                counts[label] += 1
                break
        else:
            # Larger than last bin
            label = f">{bin_edges[-1]:.0f}mm"
            counts[label] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description='Analyze tumor size distribution')
    parser.add_argument('--data_dir', type=str,
                        default='/home/mode/NTNU/MedicalDataSets/brainmetshare-3',
                        help='Path to brainmetshare-3 dataset')
    parser.add_argument('--fov_mm', type=float, default=240.0,
                        help='Field of view in mm')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size in pixels')
    args = parser.parse_args()

    pixel_spacing = args.fov_mm / args.image_size
    print(f"Pixel spacing: {pixel_spacing:.4f} mm/pixel")
    print(f"Data directory: {args.data_dir}")
    print()

    # Collect all tumor data
    all_diameters = []
    tumor_counts = []
    slices_by_count = defaultdict(int)

    for split in ['train', 'val']:
        print(f"=== Processing {split} split ===")
        slices = load_seg_slices(args.data_dir, split)
        print(f"Found {len(slices)} positive slices")

        for seg_mask in tqdm(slices, desc="Analyzing"):
            count, diameters = analyze_slice(seg_mask, pixel_spacing)
            tumor_counts.append(count)
            all_diameters.extend(diameters)
            slices_by_count[count] += 1

    print()
    print("=" * 60)
    print("TUMOR SIZE ANALYSIS")
    print("=" * 60)
    print(f"Total positive slices: {len(tumor_counts)}")
    print(f"Total tumors: {len(all_diameters)}")
    print()

    # Basic statistics
    if all_diameters:
        diameters_arr = np.array(all_diameters)
        print("Diameter Statistics (mm):")
        print(f"  Min:    {diameters_arr.min():.2f}")
        print(f"  Max:    {diameters_arr.max():.2f}")
        print(f"  Mean:   {diameters_arr.mean():.2f}")
        print(f"  Median: {np.median(diameters_arr):.2f}")
        print(f"  Std:    {diameters_arr.std():.2f}")
        print()

        # Percentiles
        print("Percentiles:")
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"  {p}th: {np.percentile(diameters_arr, p):.2f} mm")
        print()

    # Tumor count distribution
    print("Tumors per slice:")
    for count in sorted(slices_by_count.keys()):
        n = slices_by_count[count]
        pct = 100 * n / len(tumor_counts) if tumor_counts else 0
        print(f"  {count} tumor(s): {n} slices ({pct:.1f}%)")
    print()

    # Test different binning schemes
    print("=" * 60)
    print("BINNING SCHEME COMPARISON")
    print("=" * 60)

    bin_schemes = {
        "Linear 5mm": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        "Linear 3mm": [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 35, 40, 50],
        "RANO-inspired": [0, 5, 10, 15, 20, 25, 30, 40, 50],
        "Log-scale": [0, 2, 4, 6, 9, 13, 18, 25, 35, 50],
        "Fine-grained": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50],
    }

    for scheme_name, edges in bin_schemes.items():
        print(f"\n{scheme_name} ({len(edges)-1} bins):")
        print("-" * 40)

        counts = bin_sizes(all_diameters, edges)

        # Create ordered bin labels
        bin_labels = []
        for i in range(len(edges) - 1):
            bin_labels.append(f"{edges[i]:.0f}-{edges[i+1]:.0f}mm")
        bin_labels.append(f">{edges[-1]:.0f}mm")

        for label in bin_labels:
            n = counts.get(label, 0)
            pct = 100 * n / len(all_diameters) if all_diameters else 0
            bar = "█" * int(pct / 2)
            print(f"  {label:>12}: {n:5} ({pct:5.1f}%) {bar}")

    # Per-slice bin vector analysis
    print()
    print("=" * 60)
    print("PER-SLICE BIN VECTOR ANALYSIS (RANO-BM aligned, 6 bins)")
    print("=" * 60)

    log_edges = [0, 3, 6, 10, 15, 20, 30]  # 6 bins, aligned with RANO-BM
    num_bins = len(log_edges) - 1
    bin_labels = [f"{log_edges[i]}-{log_edges[i+1]}" for i in range(num_bins)]

    print(f"\nBins: {bin_labels}")
    print()

    # Re-analyze to get per-slice bin vectors
    slice_bin_vectors = []
    for split in ['train', 'val']:
        slices = load_seg_slices(args.data_dir, split)
        for seg_mask in slices:
            count, diameters = analyze_slice(seg_mask, pixel_spacing)
            # Create bin vector for this slice
            bin_vector = [0] * num_bins
            for d in diameters:
                for i in range(num_bins):
                    if log_edges[i] <= d < log_edges[i + 1]:
                        bin_vector[i] += 1
                        break
                else:
                    # >= 50mm goes in last bin
                    bin_vector[num_bins - 1] += 1
            slice_bin_vectors.append(tuple(bin_vector))

    # Count unique conditioning vectors
    from collections import Counter
    vector_counts = Counter(slice_bin_vectors)

    print(f"Total slices: {len(slice_bin_vectors)}")
    print(f"Unique conditioning vectors: {len(vector_counts)}")
    print()

    # Show most common vectors
    print("Top 30 most common conditioning vectors:")
    print("-" * 70)
    print(f"{'Vector':<45} {'Count':>8} {'%':>8}")
    print("-" * 70)

    for vec, cnt in vector_counts.most_common(30):
        pct = 100 * cnt / len(slice_bin_vectors)
        # Format vector nicely
        vec_str = str(list(vec))
        print(f"{vec_str:<45} {cnt:>8} {pct:>7.1f}%")

    print()

    # Bin occupancy statistics
    print("Bin occupancy (how often each bin has ≥1 tumor):")
    print("-" * 50)
    bin_occupancy = [0] * num_bins
    for vec in slice_bin_vectors:
        for i, v in enumerate(vec):
            if v > 0:
                bin_occupancy[i] += 1

    for i, label in enumerate(bin_labels):
        occ = bin_occupancy[i]
        pct = 100 * occ / len(slice_bin_vectors)
        bar = "█" * int(pct / 2)
        print(f"  {label:>8}mm: {occ:5} slices ({pct:5.1f}%) {bar}")

    print()

    # Max count per bin
    print("Max tumor count observed per bin:")
    print("-" * 50)
    max_per_bin = [0] * num_bins
    for vec in slice_bin_vectors:
        for i, v in enumerate(vec):
            max_per_bin[i] = max(max_per_bin[i], v)

    for i, label in enumerate(bin_labels):
        print(f"  {label:>8}mm: max {max_per_bin[i]} tumors")

    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(f"\nWith {len(vector_counts)} unique vectors, the model needs to learn")
    print("a mapping from 9-dim count vector → segmentation mask.")
    print()
    print("Conditioning approach options:")
    print("  1. Direct embedding: Embed each bin count (0-max) separately")
    print("  2. Sinusoidal: Use sinusoidal encoding for each bin count")
    print("  3. Combined: Concatenate all bin embeddings into one vector")


if __name__ == "__main__":
    main()
