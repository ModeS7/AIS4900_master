#!/usr/bin/env python3
"""
Diffusion scheduler comparison visualization script.
Creates configurable visualizations for comparing different noise schedulers.

USAGE:
------
1. Edit the CONFIGURATION section below to select:
   - Which visualizations to generate (image grid, metrics plots, etc.)
   - Which schedulers to compare (Cosine, Linear, Sigmoid, RFlow)
   - Which metrics to plot (SNR, PSNR, MSE, etc.)

2. Run the script: python vis2.py

EXAMPLE CONFIGURATIONS:
-----------------------
For thesis Theory section (Linear vs Cosine with SNR):
    GENERATE_IMAGE_COMPARISON = True
    GENERATE_SIDE_BY_SIDE = True
    INCLUDE_SCHEDULERS: Cosine=True, Linear=True, others=False
    AVAILABLE_METRICS: snr_db=True, others=False

For comprehensive analysis (all schedulers, all metrics):
    GENERATE_IMAGE_COMPARISON = True
    GENERATE_NORMALIZED_METRICS = True
    GENERATE_SIDE_BY_SIDE = True
    All schedulers = True
    All metrics = True
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ============================================================================
# CONFIGURATION - Edit these to control what gets generated
# ============================================================================

# Which visualizations to generate
GENERATE_IMAGE_COMPARISON = True      # Image grid showing noise progression
GENERATE_NORMALIZED_METRICS = False   # Normalized metrics (all on 0-1 scale)
GENERATE_SIDE_BY_SIDE = True          # Side-by-side metric comparison
PRINT_STATISTICS = False              # Print summary statistics

# Scheduler selection (set to False to exclude from visualizations)
INCLUDE_SCHEDULERS = {
    'Linear': True,
    'Cosine': True,
    'Sigmoid': False,
    'Logit-Normal RFlow': True
}

# Folder paths for each scheduler
FOLDER_PATHS = {
    'Linear': 'linear_diffusion_bravo_patient13_slice76_Mets_036',
    'Cosine': 'cosine_diffusion_bravo_patient13_slice76_Mets_036',
    'Sigmoid': 'sigmoid_diffusion_bravo_patient13_slice76_Mets_036',
    'Logit-Normal RFlow': 'logit-normal_rflow_bravo_patient13_slice76_Mets_036'
}

# Timesteps to display in image comparison
TIMESTEPS = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]

# Metrics to plot (set to False to exclude)
AVAILABLE_METRICS = {
    'snr_db': False,             # Signal-to-Noise Ratio (dB)
    'psnr_db': False,            # Peak SNR
    'mse': False,                # Mean Squared Error
    'mae': False,                # Mean Absolute Error
    'rmse': False,               # Root Mean Squared Error
    'noise_power': False,        # Noise power
    'noise_std': True,          # Noise standard deviation - BEST for showing noise progression
    'variance_difference': False, # Variance difference
    'mean_noisy': False,         # Mean of noisy image
    'std_noisy': False,          # Std of noisy image
    'range_noisy': False         # Range of noisy values
}

# ============================================================================
# Derived configuration (don't edit below unless you know what you're doing)
# ============================================================================

# Filter active schedulers and metrics
ACTIVE_SCHEDULERS = [name for name, include in INCLUDE_SCHEDULERS.items() if include]
FOLDER_NAMES = [FOLDER_PATHS[name] for name in ACTIVE_SCHEDULERS]
SCHEDULER_NAMES = ACTIVE_SCHEDULERS
KEY_METRICS = [metric for metric, include in AVAILABLE_METRICS.items() if include]


def load_images_from_folder(folder_path):
    """Load all diffusion images from a folder in timestep order."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    images = {}
    files = os.listdir(folder_path)

    # Filter PNG files and extract timestep
    for file in files:
        if file.endswith('.png'):
            # Extract timestep from filename (assuming format: *_noise{timestep:03d}_*.png)
            try:
                parts = file.split('_noise')
                if len(parts) > 1:
                    timestep_part = parts[1].split('_')[0]
                    timestep = int(timestep_part)
                    images[timestep] = os.path.join(folder_path, file)
            except (ValueError, IndexError):
                print(f"Warning: Could not parse timestep from {file}")

    return images


def load_csv_data(folder_path):
    """Load noise metrics CSV from folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {folder_path}")

    csv_path = os.path.join(folder_path, csv_files[0])
    return pd.read_csv(csv_path)


def create_image_comparison():
    """Create grid comparing images across schedulers and timesteps by concatenating images directly."""
    num_schedulers = len(SCHEDULER_NAMES)
    num_timesteps = len(TIMESTEPS)

    # Load all images first
    all_rows = []

    for scheduler_name, folder in zip(SCHEDULER_NAMES, FOLDER_NAMES):
        try:
            images = load_images_from_folder(folder)
            row_images = []

            for timestep in TIMESTEPS:
                if timestep in images:
                    img = Image.open(images[timestep])
                    # Convert to grayscale if RGB
                    if img.mode == 'RGB' or img.mode == 'RGBA':
                        img = img.convert('L')
                    img_array = np.array(img)
                    row_images.append(img_array)
                else:
                    # Create placeholder if missing
                    if row_images:
                        placeholder = np.zeros_like(row_images[0])
                    else:
                        placeholder = np.zeros((128, 128))  # default size
                    row_images.append(placeholder)

            # Concatenate images horizontally with no gap
            if row_images:
                row_concat = np.concatenate(row_images, axis=1)
                all_rows.append(row_concat)

        except FileNotFoundError as e:
            print(f"Error loading images for {scheduler_name}: {e}")
            # Create empty row
            if all_rows:
                placeholder_row = np.zeros_like(all_rows[0])
            else:
                placeholder_row = np.zeros((128, 128 * num_timesteps))
            all_rows.append(placeholder_row)

    # Add spacing between scheduler rows
    spacing_height = 20  # pixels of white space between rows
    rows_with_spacing = []
    for i, row in enumerate(all_rows):
        rows_with_spacing.append(row)
        if i < len(all_rows) - 1:  # Don't add spacing after last row
            spacer = np.ones((spacing_height, row.shape[1])) * 255  # white spacer
            rows_with_spacing.append(spacer)

    # Concatenate all rows vertically
    full_grid = np.concatenate(rows_with_spacing, axis=0)

    # Create figure with single axis
    fig_width = num_timesteps * 2
    fig_height = num_schedulers * 2
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    ax.imshow(full_grid, cmap='gray', aspect='auto')
    ax.axis('off')

    # Add labels manually as text on the image
    img_height = all_rows[0].shape[0]
    img_width = all_rows[0].shape[1] // num_timesteps

    # Determine if we have RFlow scheduler
    has_rflow = any('rflow' in name.lower() for name in SCHEDULER_NAMES)

    # Add DDPM timestep labels at top
    for col, timestep in enumerate(TIMESTEPS):
        x = (col + 0.5) * img_width
        y = -30
        ax.text(x, y, f't={timestep}', fontsize=10, ha='center', va='bottom')

    # Add RFlow timestep labels if RFlow is included
    if has_rflow:
        # RFlow uses continuous time from 1 (data) to 0 (noise)
        rflow_times = [1.0 - (i / (num_timesteps - 1)) for i in range(num_timesteps)]

        # Position below last row
        total_height = sum(row.shape[0] for row in all_rows) + spacing_height * (len(all_rows) - 1)

        for col, rflow_t in enumerate(rflow_times):
            x = (col + 0.5) * img_width
            y = total_height + 20
            ax.text(x, y, f't={rflow_t:.1f}', fontsize=10, ha='center', va='top', color='black')

    # Add scheduler labels on left
    for row, scheduler_name in enumerate(SCHEDULER_NAMES):
        y = row * (img_height + spacing_height) + img_height / 2
        x = -30
        ax.text(x, y, scheduler_name, fontsize=12, rotation=90, ha='right', va='center')

    # Save the figure
    output_path = 'diffusion_scheduler_image_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Image comparison saved as: {output_path}")
    plt.close()

    return fig


def normalize_metrics_data(df, metrics):
    """Normalize metrics to 0-1 range for comparison plotting."""
    normalized_data = {}

    for metric in metrics:
        if metric in df.columns:
            values = df[metric].values
            if len(values) > 0:
                min_val = np.min(values)
                max_val = np.max(values)
                if max_val != min_val:
                    normalized_data[metric] = (values - min_val) / (max_val - min_val)
                else:
                    normalized_data[metric] = np.zeros_like(values)
            else:
                normalized_data[metric] = np.array([])
        else:
            print(f"Warning: Metric '{metric}' not found in data")
            normalized_data[metric] = np.array([])

    return normalized_data


def create_metrics_comparison():
    """Create metrics comparison plots for all scheduler modes."""
    num_schedulers = len(SCHEDULER_NAMES)
    fig_height = 3 * num_schedulers
    fig, axes = plt.subplots(num_schedulers, 1, figsize=(14, fig_height), squeeze=False)

    colors = plt.cm.tab10(np.linspace(0, 1, len(KEY_METRICS)))

    for row, (folder, scheduler_name) in enumerate(zip(FOLDER_NAMES, SCHEDULER_NAMES)):
        ax = axes[row, 0]

        try:
            df = load_csv_data(folder)

            # Ensure data is sorted by timestep
            df = df.sort_values('timestep')
            timesteps = df['timestep'].values

            # Normalize metrics for comparison
            normalized_data = normalize_metrics_data(df, KEY_METRICS)

            # Plot each metric
            for i, metric in enumerate(KEY_METRICS):
                if len(normalized_data[metric]) > 0:
                    ax.plot(timesteps, normalized_data[metric],
                            label=metric, color=colors[i], linewidth=2, marker='o', markersize=3)

            ax.set_xlabel('Timestep')
            ax.set_ylabel('Normalized Value (0-1)')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 999)
            ax.set_ylim(-0.05, 1.05)

            # Add legend (only for first subplot to avoid repetition)
            if row == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        except FileNotFoundError as e:
            print(f"Error loading CSV for {scheduler_name}: {e}")
            ax.text(0.5, 0.5, f'CSV Not Found\n{scheduler_name}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(f'{scheduler_name} Scheduler - Data Missing', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # Save the figure
    output_path = 'diffusion_scheduler_metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved as: {output_path}")

    return fig


def create_side_by_side_metrics():
    """Create side-by-side comparison of selected metrics across all schedulers."""
    try:
        if not KEY_METRICS:
            print("Warning: No metrics selected for side-by-side comparison")
            return None

        # Load all CSV data
        all_data = {}
        for folder, scheduler_name in zip(FOLDER_NAMES, SCHEDULER_NAMES):
            df = load_csv_data(folder)
            df = df.sort_values('timestep').reset_index(drop=True)

            # Debug: Check what timesteps exist in CSV
            print(f"\n{scheduler_name} CSV data:")
            print(f"  Timesteps in CSV: {df['timestep'].tolist()}")
            print(f"  First row: {df.iloc[0]['timestep']}")

            all_data[scheduler_name] = df

        # Calculate grid layout dynamically based on number of metrics
        num_metrics = len(KEY_METRICS)
        cols = min(4, num_metrics)
        rows = (num_metrics + cols - 1) // cols

        fig_width = 6 * cols
        fig_height = 4 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

        axes_flat = axes.flatten()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

        for i, metric in enumerate(KEY_METRICS):
            ax = axes_flat[i]

            for j, (scheduler_name, df) in enumerate(all_data.items()):
                if metric in df.columns:
                    # Get the data
                    t = df['timestep'].values
                    v = df[metric].values

                    # Handle infinity values (e.g., PSNR at t=0 where MSE=0)
                    # Replace inf with a reasonable upper bound
                    if metric in ['psnr_db', 'snr_db']:
                        v = np.where(np.isinf(v), np.nanmax(v[~np.isinf(v)]) * 1.5, v)

                    # Debug output for first metric only
                    if i == 0:
                        print(f"{scheduler_name}: first timestep={t[0]}, first 3 values={v[:3]}")

                    # Plot with the data
                    ax.plot(t, v, label=scheduler_name,
                            color=colors[j % len(colors)], linewidth=2.5,
                            marker='o', markersize=4)

            # Set axis labels based on metric
            metric_labels = {
                'noise_std': 'Noise Standard Deviation',
                'snr_db': 'SNR (dB)',
                'psnr_db': 'PSNR (dB)',
                'mse': 'Mean Squared Error',
                'mae': 'Mean Absolute Error',
                'rmse': 'Root Mean Squared Error',
                'noise_power': 'Noise Power',
                'variance_difference': 'Variance Difference',
                'mean_noisy': 'Mean Value',
                'std_noisy': 'Standard Deviation',
                'range_noisy': 'Pixel Range'
            }
            ylabel = metric_labels.get(metric, 'Value')

            ax.set_xlabel('Timestep (DDPM)', fontsize=10, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax.set_xlim(-10, 1010)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=9)

            # Add RFlow time axis if RFlow is in schedulers
            has_rflow = any('rflow' in name.lower() for name in SCHEDULER_NAMES)
            if has_rflow:
                ax2 = ax.twiny()
                ax2.set_xlim(-10, 1010)
                # Set RFlow ticks at same positions as DDPM ticks
                ddpm_ticks = [0, 199, 399, 599, 799, 999]
                rflow_ticks = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
                ax2.set_xticks(ddpm_ticks)
                ax2.set_xticklabels([f'{t:.1f}' for t in rflow_ticks])
                ax2.set_xlabel('Timestep (RFlow)', fontsize=10, fontweight='bold')
                ax2.tick_params(axis='x', labelsize=9)

            if i == 0:
                ax.legend(fontsize=10, framealpha=0.9)

        # Hide unused subplots
        for i in range(num_metrics, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()

        # Save the figure
        output_path = 'diffusion_scheduler_key_metrics.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved as: {output_path}")
        plt.close()

        return fig

    except Exception as e:
        print(f"Error creating side-by-side metrics: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary_statistics():
    """Print summary statistics for all schedulers."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for folder, scheduler_name in zip(FOLDER_NAMES, SCHEDULER_NAMES):
        try:
            df = load_csv_data(folder)
            print(f"\n{scheduler_name} Scheduler:")
            print("-" * 40)

            # Key metrics at t=0 and t=999
            t0_data = df[df['timestep'] == 0].iloc[0] if len(df[df['timestep'] == 0]) > 0 else None
            t999_data = df[df['timestep'] == 999].iloc[0] if len(df[df['timestep'] == 999]) > 0 else None

            if t0_data is not None and t999_data is not None:
                print(f"SNR (dB):     t=0: {t0_data['snr_db']:7.2f} | t=999: {t999_data['snr_db']:7.2f}")
                print(f"PSNR (dB):    t=0: {t0_data['psnr_db']:7.2f} | t=999: {t999_data['psnr_db']:7.2f}")
                print(f"MSE:          t=0: {t0_data['mse']:7.4f} | t=999: {t999_data['mse']:7.4f}")
                print(f"Noise Std:    t=0: {t0_data['noise_std']:7.4f} | t=999: {t999_data['noise_std']:7.4f}")

        except FileNotFoundError:
            print(f"\n{scheduler_name} Scheduler: DATA NOT FOUND")


def main():
    """Main execution function."""
    print("Diffusion Scheduler Comparison Visualization")
    print("=" * 60)

    # Show configuration
    print(f"\nActive schedulers: {', '.join(ACTIVE_SCHEDULERS)}")
    print(f"Active metrics: {', '.join(KEY_METRICS) if KEY_METRICS else 'None'}")
    print()

    if not FOLDER_NAMES:
        print("ERROR: No schedulers selected!")
        print("Edit INCLUDE_SCHEDULERS in the configuration section to enable schedulers.")
        return

    if not KEY_METRICS and (GENERATE_SIDE_BY_SIDE or GENERATE_NORMALIZED_METRICS):
        print("WARNING: No metrics selected!")
        print("Edit AVAILABLE_METRICS in the configuration section to enable metrics.")

    # Check if folders exist
    missing_folders = [f for f in FOLDER_NAMES if not os.path.exists(f)]
    if missing_folders:
        print("Warning: Missing folders:")
        for folder in missing_folders:
            print(f"  - {folder}")
        print()

    try:
        generated_files = []

        # Create visualizations based on configuration
        if GENERATE_IMAGE_COMPARISON:
            print("Creating image comparison grid...")
            img_fig = create_image_comparison()
            generated_files.append("diffusion_scheduler_image_comparison.png")

        if GENERATE_NORMALIZED_METRICS:
            print("Creating normalized metrics comparison...")
            metrics_fig = create_metrics_comparison()
            generated_files.append("diffusion_scheduler_metrics_comparison.png")

        if GENERATE_SIDE_BY_SIDE:
            print("Creating side-by-side key metrics...")
            key_metrics_fig = create_side_by_side_metrics()
            generated_files.append("diffusion_scheduler_key_metrics.png")

        # Print summary statistics
        if PRINT_STATISTICS:
            print_summary_statistics()

        print("\nVisualization complete!")
        if generated_files:
            print("Generated files:")
            for file in generated_files:
                print(f"  - {file}")
        else:
            print("No files generated (all visualization options disabled)")

        # Don't show plots in headless environment
        # plt.show()

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()