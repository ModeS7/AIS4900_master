#!/usr/bin/env python3
"""
Convert NIfTI (.nii.gz) files to PNG images.

Usage:
    python nii_to_png.py input.nii.gz output.png [--slice SLICE] [--axis AXIS] [--normalize]
    python nii_to_png.py input.nii.gz output_dir/ --all-slices [--axis AXIS] [--normalize]
    python nii_to_png.py --patient patient13 --slice 76 --modalities bravo seg t1_gd -o output.png
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import PathConfig


def normalize_slice(slice_data: np.ndarray, percentile_clip: bool = False) -> np.ndarray:
    """Normalize slice to 0-255 uint8 range (matches sample_dataset.py)."""
    slice_data = slice_data.astype(np.float32)

    if percentile_clip:
        # Clip to 1st-99th percentile to handle outliers
        vmin, vmax = np.percentile(slice_data, [1, 99])
        slice_data = np.clip(slice_data, vmin, vmax)
    else:
        # Use full min-max range (matches NIfTI viewers)
        vmin, vmax = slice_data.min(), slice_data.max()

    # Normalize to 0-255
    if vmax > vmin:
        slice_data = (slice_data - vmin) / (vmax - vmin) * 255
    else:
        slice_data = np.zeros_like(slice_data)

    return slice_data.astype(np.uint8)


def extract_slice(volume: np.ndarray, slice_idx: int, axis: int = 2) -> np.ndarray:
    """Extract a 2D slice from 3D volume along specified axis."""
    if axis == 0:
        return volume[slice_idx, :, :]
    elif axis == 1:
        return volume[:, slice_idx, :]
    elif axis == 2:
        return volume[:, :, slice_idx]
    else:
        raise ValueError(f"Invalid axis {axis}. Must be 0, 1, or 2.")


def save_slice_as_png(slice_data: np.ndarray, output_path: Path, normalize: bool = True):
    """Save 2D numpy array as PNG."""
    if normalize:
        slice_data = normalize_slice(slice_data)
    else:
        # Ensure uint8
        slice_data = slice_data.astype(np.uint8)

    # Rotate 90 degrees clockwise to match anatomical orientation
    slice_data = np.rot90(slice_data, k=3)

    # Create PIL Image and save
    img = Image.fromarray(slice_data, mode='L')  # 'L' for grayscale
    img.save(output_path)


def convert_single_slice(nii_path: Path, output_path: Path, slice_idx: int = None,
                         axis: int = 2, normalize: bool = True, channel: int = None):
    """Convert a single slice from NIfTI to PNG.

    Args:
        nii_path: Path to NIfTI file
        output_path: Path to output PNG
        slice_idx: Slice index (default: middle slice, or channel index for 2-channel data)
        axis: Axis to slice along for 3D volumes
        normalize: Apply min-max normalization
        channel: For 2-channel synthetic data (128,128,2): 0=bravo, 1=seg
    """
    # Load NIfTI file
    nii = nib.load(nii_path)
    volume = nii.get_fdata()

    # Handle 2-channel synthetic format (128, 128, 2)
    if len(volume.shape) == 3 and volume.shape[2] == 2:
        # This is synthetic 2-channel data (bravo + seg)
        if channel is None:
            channel = 0  # Default to bravo
        if channel not in [0, 1]:
            raise ValueError(f"Channel must be 0 (bravo) or 1 (seg), got {channel}")

        slice_data = volume[:, :, channel]
        print(f"  Detected 2-channel synthetic format, extracting channel {channel} ({'bravo' if channel == 0 else 'seg'})")
    else:
        # Standard 3D volume
        # Determine slice index (middle slice if not specified)
        if slice_idx is None:
            slice_idx = volume.shape[axis] // 2

        # Validate slice index
        if slice_idx < 0 or slice_idx >= volume.shape[axis]:
            raise ValueError(f"Slice index {slice_idx} out of range [0, {volume.shape[axis]})")

        slice_data = extract_slice(volume, slice_idx, axis)

    save_slice_as_png(slice_data, output_path, normalize)

    print(f"✓ Saved to {output_path}")
    print(f"  Volume shape: {volume.shape}, Output shape: {slice_data.shape}")


def convert_all_slices(nii_path: Path, output_dir: Path, axis: int = 2, normalize: bool = True):
    """Convert all slices from NIfTI to separate PNG files."""

    # Load NIfTI file
    nii = nib.load(nii_path)
    volume = nii.get_fdata()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get number of slices along axis
    num_slices = volume.shape[axis]

    # Save each slice
    base_name = nii_path.stem.replace('.nii', '')
    for slice_idx in range(num_slices):
        slice_data = extract_slice(volume, slice_idx, axis)
        output_path = output_dir / f"{base_name}_axis{axis}_slice{slice_idx:03d}.png"
        save_slice_as_png(slice_data, output_path, normalize)

    print(f"✓ Saved {num_slices} slices to {output_dir}")
    print(f"  Volume shape: {volume.shape}")


def load_patient_slice(patient_id: str, slice_idx: int, modality: str,
                       base_path: Path = None, axis: int = 2) -> np.ndarray:
    """Load a specific slice from a patient's MRI volume.

    Args:
        patient_id: Patient identifier (e.g., 'patient13' or 'train/Mets_030')
        slice_idx: Slice index (0-indexed)
        modality: Modality name ('bravo', 'seg', 't1_pre', 't1_gd', 'flair')
        base_path: Base directory for dataset
        axis: Axis to slice along (default: 2)

    Returns:
        2D numpy array of the slice
    """
    if base_path is None:
        # Use PathConfig to determine base path
        path_config = PathConfig()
        base_path = path_config.brainmet_train_dir.parent

    # Construct file path - handle both direct paths and patient IDs
    patient_path = Path(patient_id)

    # If patient_id looks like an absolute path or contains /, use it directly
    if patient_path.is_absolute() or '/' in patient_id:
        # Check if it's already a full path to patient directory
        if (patient_path / f"{modality}.nii.gz").exists():
            file_path = patient_path / f"{modality}.nii.gz"
        else:
            # Assume it's relative to base_path
            file_path = base_path / patient_id / f"{modality}.nii.gz"
    else:
        # Simple patient ID
        file_path = base_path / patient_id / f"{modality}.nii.gz"

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load NIfTI file
    nii = nib.load(file_path)
    volume = nii.get_fdata()

    # Extract slice
    slice_data = extract_slice(volume, slice_idx, axis)

    # Special handling for segmentation (binary mask)
    if modality == 'seg':
        slice_data = (slice_data > 0.5).astype(np.float32)

    return slice_data


def save_patient_slices(patient_id: str, slice_idx: int, modalities: list,
                       output_pattern: str, base_path: Path = None, axis: int = 2,
                       percentile_clip: bool = False) -> list:
    """Save patient slices as separate PNG files (matches sample_dataset.py behavior).

    Args:
        patient_id: Patient identifier
        slice_idx: Slice index
        modalities: List of modality names to include
        output_pattern: Output filename pattern (e.g., 'patient13_slice76_{modality}.png')
        base_path: Base directory for dataset
        axis: Axis to slice along
        percentile_clip: Use percentile clipping for normalization

    Returns:
        List of saved file paths
    """
    saved_files = []

    for modality in modalities:
        try:
            # Load slice
            slice_data = load_patient_slice(patient_id, slice_idx, modality, base_path, axis)

            # Normalize (min-max by default, matching sample_dataset.py)
            slice_normalized = normalize_slice(slice_data, percentile_clip)

            # Rotate 90 degrees clockwise to match anatomical orientation
            # This makes the images match how they appear in NIfTI viewers
            slice_normalized = np.rot90(slice_normalized, k=3)

            # Format output filename
            if '{modality}' in output_pattern:
                output_file = output_pattern.format(modality=modality)
            else:
                # If no pattern, append modality to filename
                output_path = Path(output_pattern)
                output_file = output_path.parent / f"{output_path.stem}_{modality}{output_path.suffix}"

            output_path = Path(output_file)

            # Save as grayscale PNG
            img = Image.fromarray(slice_normalized, mode='L')
            img.save(output_path)

            saved_files.append(output_path)
            print(f"  ✓ Saved {modality} to {output_path}")

        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue

    if not saved_files:
        raise ValueError("No valid modalities found")

    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Convert NIfTI (.nii.gz) files to PNG images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode: Convert middle slice (default)
  python nii_to_png.py brain.nii.gz brain.png

  # Convert specific slice
  python nii_to_png.py brain.nii.gz brain_slice50.png --slice 50

  # Convert along different axis (0=sagittal, 1=coronal, 2=axial)
  python nii_to_png.py brain.nii.gz brain_coronal.png --axis 1

  # Convert all slices to directory
  python nii_to_png.py brain.nii.gz output_slices/ --all-slices

  # Synthetic 2-channel data: extract bravo (channel 0)
  python nii_to_png.py synthetic/1234.nii.gz output_bravo.png --channel 0

  # Synthetic 2-channel data: extract seg (channel 1)
  python nii_to_png.py synthetic/1234.nii.gz output_seg.png --channel 1

  # Patient mode: Save separate PNGs for each modality
  python nii_to_png.py --patient train/Mets_030 --slice 86 --modalities bravo seg
  # Creates: train_Mets_030_slice86_bravo.png, train_Mets_030_slice86_seg.png

  # Custom output pattern with {modality} placeholder
  python nii_to_png.py --patient train/Mets_030 --slice 86 --modalities bravo seg -o patient_{modality}.png

  # Include all modalities
  python nii_to_png.py --patient train/Mets_030 --slice 86 --modalities bravo seg t1_pre t1_gd
        """
    )

    parser.add_argument('input', type=str, nargs='?', default=None,
                        help='Input NIfTI file (.nii.gz) - not used in patient mode')
    parser.add_argument('output', type=str, nargs='?', default=None,
                        help='Output PNG file or directory (for --all-slices)')
    parser.add_argument('--slice', type=int, default=None,
                        help='Slice index to extract (default: middle slice)')
    parser.add_argument('--axis', type=int, default=2, choices=[0, 1, 2],
                        help='Axis to slice along: 0=sagittal, 1=coronal, 2=axial (default: 2)')
    parser.add_argument('--channel', type=int, default=None, choices=[0, 1],
                        help='For 2-channel synthetic data: 0=bravo, 1=seg (default: auto-detect)')
    parser.add_argument('--all-slices', action='store_true',
                        help='Export all slices to separate PNG files')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Disable intensity normalization')

    # Patient mode arguments
    parser.add_argument('--patient', type=str, default=None,
                        help='Patient ID for composite mode (e.g., patient13)')
    parser.add_argument('--modalities', type=str, nargs='+',
                        default=['bravo', 'seg'],
                        choices=['bravo', 'seg', 't1_pre', 't1_gd', 'flair'],
                        help='Modalities to include in composite (default: bravo seg)')
    parser.add_argument('--base-path', type=str, default=None,
                        help='Base dataset path for patient mode (default: /home/mode/NTNU/MedicalDataSets/brainmetshare-3)')
    parser.add_argument('-o', '--output-file', type=str, default=None,
                        help='Output file for patient mode')

    args = parser.parse_args()

    # Patient mode
    if args.patient:
        if args.slice is None:
            raise ValueError("--slice is required in patient mode")

        output_file = args.output_file or args.output
        if not output_file:
            # Default pattern: patient_slice_modality.png
            patient_name = args.patient.replace('/', '_')
            output_file = f"{patient_name}_slice{args.slice}_{{modality}}.png"

        base_path = Path(args.base_path) if args.base_path else None

        print(f"Saving slices for {args.patient}, slice {args.slice}")
        print(f"Modalities: {', '.join(args.modalities)}")

        saved_files = save_patient_slices(
            args.patient,
            args.slice,
            args.modalities,
            output_file,
            base_path,
            args.axis,
            percentile_clip=False  # Always use min-max (matches sample_dataset.py)
        )

        print(f"\n✓ Saved {len(saved_files)} files")

    # Single file mode
    else:
        if not args.input or not args.output:
            parser.error("input and output are required in single file mode (or use --patient for patient mode)")

        # Convert paths
        input_path = Path(args.input)
        output_path = Path(args.output)

        # Validate input
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Convert
        normalize = not args.no_normalize

        if args.all_slices:
            convert_all_slices(input_path, output_path, args.axis, normalize)
        else:
            # Ensure output has .png extension
            if output_path.suffix.lower() != '.png':
                output_path = output_path.with_suffix('.png')

            convert_single_slice(input_path, output_path, args.slice, args.axis, normalize, args.channel)


if __name__ == '__main__':
    main()
