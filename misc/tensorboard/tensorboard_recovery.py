#!/usr/bin/env python3
"""
Recreate TensorBoard logs from training output text.
Extracts metrics from training logs and writes TensorBoard events.
"""

import re
import os
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter


def parse_training_log(log_text):
    """
    Parse training log text and extract metrics.

    Args:
        log_text (str): Raw training log text

    Returns:
        list: List of epoch data dictionaries
    """
    epochs = []

    # Pattern to match epoch lines like:
    # "Epoch 0: Total Loss = 0.084550, MSE = 0.082721, Perceptual = 1.829062"
    epoch_pattern = r'Epoch (\d+): Total Loss = ([\d.]+), MSE = ([\d.]+), Perceptual = ([\d.]+)'

    matches = re.findall(epoch_pattern, log_text)

    for match in matches:
        epoch_num = int(match[0])
        total_loss = float(match[1])
        mse_loss = float(match[2])
        perceptual_loss = float(match[3])

        epochs.append({
            'epoch': epoch_num,
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'perceptual_loss': perceptual_loss
        })

    return epochs


def extract_training_metadata(log_text):
    """
    Extract training metadata from log text.

    Args:
        log_text (str): Raw training log text

    Returns:
        dict: Training metadata
    """
    metadata = {}

    # Extract start time if available
    # Look for patterns like "Starting training for seg images..."
    if "Starting training for seg images..." in log_text:
        metadata['training_type'] = 'seg'
    elif "Starting training for bravo images..." in log_text:
        metadata['training_type'] = 'bravo'

    # Extract total training time
    time_pattern = r'Training completed in ([\d.]+) seconds \(([\d.]+) hours\)'
    time_match = re.search(time_pattern, log_text)
    if time_match:
        metadata['total_seconds'] = float(time_match.group(1))
        metadata['total_hours'] = float(time_match.group(2))

    # Extract model save epochs
    save_pattern = r'✓ Saved model at epoch (\d+):'
    save_epochs = [int(match) for match in re.findall(save_pattern, log_text)]
    metadata['saved_epochs'] = save_epochs

    return metadata


def calculate_timestamps(epochs, total_seconds, start_time=None):
    """
    Calculate realistic timestamps for each epoch.

    Args:
        epochs (list): List of epoch data
        total_seconds (float): Total training time in seconds
        start_time (datetime, optional): Training start time

    Returns:
        list: Epochs with timestamps added
    """
    if start_time is None:
        # Use a reasonable default based on the log date
        start_time = datetime(2025, 9, 19, 13, 27, 22)

    num_epochs = len(epochs)
    if num_epochs == 0:
        return epochs

    # Calculate seconds per epoch (assuming linear progression)
    seconds_per_epoch = total_seconds / num_epochs

    # Add timestamps to each epoch
    for i, epoch_data in enumerate(epochs):
        epoch_time = start_time + timedelta(seconds=i * seconds_per_epoch)
        epoch_data['timestamp'] = epoch_time

    return epochs


def write_tensorboard_logs(epochs, log_dir, training_type='seg'):
    """
    Write TensorBoard logs from epoch data.

    Args:
        epochs (list): List of epoch data with timestamps
        log_dir (str): Directory to write TensorBoard logs
        training_type (str): Type of training (for log naming)
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Writing TensorBoard logs to: {log_dir}")
    print(f"Total epochs parsed: {len(epochs)}")

    # Debug: Show first and last few epochs
    if len(epochs) > 0:
        print(f"First epoch: {epochs[0]['epoch']} - Loss: {epochs[0]['total_loss']:.6f}")
        print(f"Last epoch: {epochs[-1]['epoch']} - Loss: {epochs[-1]['total_loss']:.6f}")

    written_count = 0
    for i, epoch_data in enumerate(epochs):
        epoch = epoch_data['epoch']

        # Write loss metrics
        writer.add_scalar('Loss/Total', epoch_data['total_loss'], epoch)
        writer.add_scalar('Loss/MSE', epoch_data['mse_loss'], epoch)
        writer.add_scalar('Loss/Perceptual', epoch_data['perceptual_loss'], epoch)

        written_count += 1

        # Flush periodically to ensure data is written
        if (i + 1) % 50 == 0:
            writer.flush()

    # Final flush and close
    writer.flush()
    writer.close()

    print("✓ TensorBoard logs written successfully!")
    print(f"✓ Written {written_count} epochs to TensorBoard")
    print(f"Run: tensorboard --logdir {log_dir}")


def recreate_tensorboard_from_log(log_file_path, output_dir, training_type=None):
    """
    Main function to recreate TensorBoard from training log.

    Args:
        log_file_path (str): Path to training log file
        output_dir (str): Output directory for TensorBoard logs
        training_type (str, optional): Training type override
    """
    # Read log file
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()

    # Parse training data
    epochs = parse_training_log(log_text)
    metadata = extract_training_metadata(log_text)

    # Determine training type
    if training_type is None:
        training_type = metadata.get('training_type', 'unknown')

    # Extract actual training start time from log path if available
    start_time = None
    save_path_pattern = r'/trained_model/RFlow_seg_128_(\d{8})-(\d{6})/'
    save_match = re.search(save_path_pattern, log_text)
    if save_match:
        date_str = save_match.group(1)  # 20250919
        time_str = save_match.group(2)  # 132722
        try:
            start_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except:
            pass

    # Calculate timestamps if total time is available
    if 'total_seconds' in metadata:
        epochs = calculate_timestamps(epochs, metadata['total_seconds'], start_time)

    # Create output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, f"RFlow_{training_type}_128_{timestamp}")

    # Write TensorBoard logs
    write_tensorboard_logs(epochs, log_dir, training_type)

    # Print summary
    print("\nTraining Summary:")
    print(f"- Training type: {training_type}")
    print(f"- Total epochs: {len(epochs)}")
    if 'total_hours' in metadata:
        print(f"- Training time: {metadata['total_hours']:.2f} hours")
    if 'saved_epochs' in metadata:
        print(f"- Model saves: {metadata['saved_epochs']}")

    return log_dir


def debug_parsing(log_text):
    """
    Debug function to check parsing results.

    Args:
        log_text (str): Raw training log text
    """
    print("=== PARSING DEBUG ===")

    # Test regex pattern
    epoch_pattern = r'Epoch (\d+): Total Loss = ([\d.]+), MSE = ([\d.]+), Perceptual = ([\d.]+)'

    matches = re.findall(epoch_pattern, log_text)
    print(f"Total regex matches found: {len(matches)}")

    if len(matches) > 0:
        print(
            f"First match: Epoch {matches[0][0]} - Total: {matches[0][1]}, MSE: {matches[0][2]}, Perceptual: {matches[0][3]}")
        print(
            f"Second match: Epoch {matches[1][0]} - Total: {matches[1][1]}, MSE: {matches[1][2]}, Perceptual: {matches[1][3]}" if len(
                matches) > 1 else "No second match")
        print(
            f"Last match: Epoch {matches[-1][0]} - Total: {matches[-1][1]}, MSE: {matches[-1][2]}, Perceptual: {matches[-1][3]}")

        # Check for any obvious issues in epoch sequence
        epoch_numbers = [int(match[0]) for match in matches]
        print(f"Epoch range: {min(epoch_numbers)} to {max(epoch_numbers)}")
        print(f"Expected 500 epochs (0-499), got {len(epoch_numbers)} epochs")

        # Check for duplicates
        if len(set(epoch_numbers)) != len(epoch_numbers):
            print("WARNING: Duplicate epoch numbers found!")

        # Check for missing epochs in sequence
        expected_epochs = set(range(500))  # 0 to 499
        found_epochs = set(epoch_numbers)
        missing_epochs = expected_epochs - found_epochs
        if missing_epochs:
            print(f"Missing epochs: {sorted(list(missing_epochs))[:10]}...")  # Show first 10

    print("=== END DEBUG ===\n")


def recreate_from_text_directly_debug(log_text, output_dir, training_type='seg'):
    """
    Recreate TensorBoard directly from log text with debugging.
    """
    # Debug parsing first
    debug_parsing(log_text)

    # Parse training data
    epochs = parse_training_log(log_text)
    metadata = extract_training_metadata(log_text)

    print(f"Parsed epochs: {len(epochs)}")
    if len(epochs) == 0:
        print("ERROR: No epochs parsed! Check log format.")
        return None

    # Calculate timestamps if total time is available
    if 'total_seconds' in metadata:
        # Extract actual training start time from log path if available
        start_time = None
        save_path_pattern = r'/trained_model/RFlow_seg_128_(\d{8})-(\d{6})/'
        save_match = re.search(save_path_pattern, log_text)
        if save_match:
            date_str = save_match.group(1)  # 20250919
            time_str = save_match.group(2)  # 132722
            try:
                start_time = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            except:
                pass
        epochs = calculate_timestamps(epochs, metadata['total_seconds'], start_time)

    # Create output directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, f"RFlow_{training_type}_128_recreated_{timestamp}")

    # Write TensorBoard logs
    write_tensorboard_logs(epochs, log_dir, training_type)

    return log_dir


# Example usage
if __name__ == "__main__":
    with open('seg_training_log.txt', 'r') as f:
        seg_log_text = f.read()

    # Alternative: paste directly
    # seg_log_text = """Starting training for seg images...
    # Epoch 0: Total Loss = 0.084550, MSE = 0.082721, Perceptual = 1.829062
    # ...
    # Training for seg images completed."""

    # Recreate TensorBoard logs with debugging
    output_directory = "./tensorboard_logs"
    log_dir = recreate_from_text_directly_debug(seg_log_text, output_directory, 'seg')

    if log_dir:
        print("\nTo view the logs, run:")
        print(f"tensorboard --logdir {log_dir}")
    else:
        print("Failed to create TensorBoard logs. Check the debug output above.")
