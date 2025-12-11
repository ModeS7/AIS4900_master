#!/usr/bin/env python3
"""
Performance benchmarking for diffusion model training and generation.

Tests three configurations:
  1. Baseline: No AMP, no compile (slowest, lowest VRAM)
  2. AMP only: Automatic Mixed Precision with bfloat16 (faster, less VRAM)
  3. AMP + Compile: Both AMP and torch.compile (fastest)

Metrics tracked:
  - Training: iterations/sec, VRAM usage, estimated time for 500 epochs (438,500 iterations)
  - Generation: samples/sec, VRAM usage, estimated time for 15,000 images
  - Compilation time and warmup overhead

Usage:
  # Local single GPU
  python misc/benchmark.py --mode both --image_size 128 --train_iterations 200

  # Local multi-GPU (requires torchrun)
  torchrun --nproc_per_node=4 misc/benchmark.py --mode both --multi_gpu

  # Cluster single GPU
  sbatch IDUN/misc/benchmark.slurm

  # Cluster multi-GPU (4 GPUs)
  sbatch IDUN/misc/benchmark_multigpu.slurm

Arguments:
  --mode {train,generate,both}  What to benchmark (default: both)
  --image_size {128,256}        Image resolution (default: 128)
  --batch_size N                Training batch size (default: 16)
  --train_iterations N          Training iterations to benchmark (default: 100, recommend 200+)
  --gen_iterations N            Generation batches to benchmark (default: 10)
  --gen_batch_size N            Images per generation batch (default: 4)
  --gen_steps N                 Diffusion steps (default: 1000)
  --multi_gpu                   Enable multi-GPU with DDP
  --compute {local,cluster}     Environment (default: local)
  --output PATH                 Results directory (default: benchmark_results)

Output files:
  benchmark_results/training_benchmark_YYYYMMDD_HHMMSS.csv
  benchmark_results/generation_benchmark_YYYYMMDD_HHMMSS.csv
  benchmark_results/benchmark_YYYYMMDD_HHMMSS.json

Note: torch.compile needs 20-30 warmup iterations to fully optimize.
"""

import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import subprocess
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.append(str(project_root / 'Generation' / 'TrainGen'))

from config import PathConfig
from core.trainer import DiffusionTrainer
from core.data import create_dataloader
from core.strategies import DDPMStrategy
from core.modes import SegmentationMode
from monai.networks.nets import DiffusionModelUNet
from monai.losses import PerceptualLoss


class GPUMonitor:
    """Monitor GPU memory usage during benchmark."""

    def __init__(self, device_id=0):
        self.device_id = device_id

    def update(self):
        """Update memory stats (forces sync)."""
        torch.cuda.synchronize(self.device_id)

    def reset(self):
        """Reset peak memory tracking."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_id)

    def get_stats(self):
        """Get current stats using PyTorch's built-in tracking."""
        torch.cuda.synchronize(self.device_id)
        # Use max_memory_reserved which matches nvidia-smi VRAM usage
        # (includes CUDA context, cuDNN workspaces, cached memory)
        peak_reserved_gb = torch.cuda.max_memory_reserved(self.device_id) / (1024 ** 3)
        current_reserved_gb = torch.cuda.memory_reserved(self.device_id) / (1024 ** 3)
        peak_allocated_gb = torch.cuda.max_memory_allocated(self.device_id) / (1024 ** 3)
        current_allocated_gb = torch.cuda.memory_allocated(self.device_id) / (1024 ** 3)

        return {
            'current_gb': current_reserved_gb,
            'peak_gb': peak_reserved_gb,
            'peak_allocated_gb': peak_allocated_gb,
            'current_allocated_gb': current_allocated_gb
        }

    def get_memory_summary(self):
        """Get detailed memory breakdown."""
        torch.cuda.synchronize(self.device_id)
        mem_stats = torch.cuda.memory_stats(self.device_id)

        allocated = mem_stats.get('allocated_bytes.all.current', 0) / (1024 ** 3)
        reserved = mem_stats.get('reserved_bytes.all.current', 0) / (1024 ** 3)
        peak_allocated = mem_stats.get('allocated_bytes.all.peak', 0) / (1024 ** 3)
        peak_reserved = mem_stats.get('reserved_bytes.all.peak', 0) / (1024 ** 3)

        # Active memory (allocated - freed but not returned)
        active = mem_stats.get('active_bytes.all.current', 0) / (1024 ** 3)
        inactive = mem_stats.get('inactive_split_bytes.all.current', 0) / (1024 ** 3)

        # Number of allocations
        num_allocs = mem_stats.get('allocation.all.current', 0)
        num_frees = mem_stats.get('free_calls.all.current', 0)

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'peak_allocated_gb': peak_allocated,
            'peak_reserved_gb': peak_reserved,
            'active_gb': active,
            'inactive_gb': inactive,
            'num_allocations': num_allocs,
            'num_frees': num_frees,
            'fragmentation_ratio': (reserved - allocated) / reserved if reserved > 0 else 0
        }


class BenchmarkConfig:
    """Configuration for benchmark test."""

    def __init__(self, name, use_amp, use_compile, use_optimizations, compile_mode=None):
        self.name = name
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.use_optimizations = use_optimizations
        self.compile_mode = compile_mode  # None = use defaults, or specify: "default", "reduce-overhead", "max-autotune"


def setup_distributed():
    """Setup distributed training (matching DiffusionTrainer._setup_distributed)"""
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])

        if 'SLURM_NTASKS' in os.environ:
            world_size = int(os.environ['SLURM_NTASKS'])
        else:
            nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
            tasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', 1))
            world_size = nodes * tasks_per_node

        if 'SLURM_JOB_NODELIST' in os.environ:
            nodelist = os.environ['SLURM_JOB_NODELIST']
            master_addr = nodelist.split(',')[0].split('[')[0]
            os.environ['MASTER_ADDR'] = master_addr
        else:
            os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')

        # Use SLURM_JOB_ID to generate unique port
        if 'SLURM_JOB_ID' in os.environ:
            job_id = int(os.environ['SLURM_JOB_ID'])
            port = 12000 + (job_id % 53000)
            os.environ['MASTER_PORT'] = str(port)
            if rank == 0:
                print(f"Using dynamic port: {port} (from SLURM_JOB_ID: {job_id})")
        else:
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    else:
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    return rank, local_rank, world_size, device


def enable_standard_optimizations():
    """Enable standard PyTorch optimizations (matching production)."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch._dynamo.config.cache_size_limit = 32


def get_gpu_stats(device_id=0):
    """Query nvidia-smi for detailed GPU stats."""
    import subprocess
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,temperature.gpu',
            '--format=csv,noheader,nounits', f'--id={device_id}'
        ], capture_output=True, text=True, timeout=1)

        if result.returncode == 0:
            gpu_util, mem_util, power, temp = map(float, result.stdout.strip().split(','))
            return {
                'gpu_compute_util_pct': gpu_util,
                'gpu_memory_util_pct': mem_util,
                'power_draw_watts': power,
                'temperature_celsius': temp
            }
    except:
        pass
    return None


def benchmark_training(config, args, dataloader, train_dataset, device, rank=0, world_size=1, local_rank=0):
    """
    Benchmark training performance.

    Args:
        config: BenchmarkConfig
        args: Command line arguments
        dataloader: Training dataloader
        train_dataset: Training dataset
        device: torch device
        rank: Process rank (for multi-GPU)
        world_size: Number of processes (for multi-GPU)
        local_rank: Local rank on node (for multi-GPU)

    Returns:
        dict: Results including it/s and VRAM usage
    """
    is_main = (rank == 0)
    use_multi_gpu = (world_size > 1)

    if is_main:
        print(f"\n{'=' * 80}")
        print(f"TRAINING BENCHMARK: {config.name}")
        print(f"AMP: {config.use_amp} | Compile: {config.use_compile}")
        if use_multi_gpu:
            print(f"Multi-GPU: {world_size} GPUs")
        print(f"{'=' * 80}\n")

    # Reset GPU memory
    monitor = GPUMonitor(device_id=device.index if device.index is not None else 0)
    monitor.reset()

    # Create mode and get model config
    mode = SegmentationMode()
    model_config = mode.get_model_config()

    # Create model (matching DiffusionTrainer.setup_model)
    raw_model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256
    ).to(device)

    # Apply DDP if multi-GPU
    if use_multi_gpu:
        if is_main:
            print(f"Wrapping model with DDP (device_ids=[{local_rank}])")
        # DDP settings matching DiffusionTrainer
        ddp_model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False
        )

        # Apply compile after DDP if needed
        if config.use_compile:
            # Determine compile mode
            if config.compile_mode is not None:
                compile_mode = config.compile_mode
            else:
                compile_mode = "reduce-overhead"  # Default for multi-GPU

            if is_main:
                print(f"Compiling DDP model (mode={compile_mode})...")
            compile_start = time.time()
            model = torch.compile(ddp_model, mode=compile_mode)
            compile_time = time.time() - compile_start
            if is_main:
                print(f"Compilation completed in {compile_time:.2f}s")
        else:
            model = ddp_model
            compile_time = 0.0
    else:
        # Apply compile if needed (single GPU)
        if config.use_compile:
            # Determine compile mode
            if config.compile_mode is not None:
                compile_mode = config.compile_mode
            else:
                compile_mode = "default"  # Default for single GPU

            if is_main:
                print(f"Compiling model (mode={compile_mode})...")
            compile_start = time.time()
            model = torch.compile(raw_model, mode=compile_mode)
            compile_time = time.time() - compile_start
            if is_main:
                print(f"Compilation completed in {compile_time:.2f}s")
        else:
            model = raw_model
            compile_time = 0.0

    # Create optimizer (always use raw model parameters, matching DiffusionTrainer)
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-4)

    # Create perceptual loss (matching DiffusionTrainer.setup_model)
    cache_dir = project_root / 'model_cache'
    cache_dir.mkdir(exist_ok=True)

    perceptual_loss = PerceptualLoss(
        spatial_dims=2,
        network_type="radimagenet_resnet50",
        cache_dir=str(cache_dir),
        pretrained=True,
    ).to(device)

    if config.use_compile:
        perceptual_loss = torch.compile(perceptual_loss, mode="reduce-overhead")

    # Create strategy and setup scheduler
    strategy = DDPMStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=args.image_size)

    model.train()

    # Synchronize before training
    if use_multi_gpu:
        dist.barrier()

    # Dynamic warmup: 3 epochs with compile, 1 epoch without compile
    warmup_epochs = 3 if config.use_compile else 1

    # Calculate total epochs
    total_epochs = warmup_epochs + args.train_epochs

    if is_main:
        print(f"Running {total_epochs} epochs ({warmup_epochs} warmup + {args.train_epochs} measured)")
        print(f"  Each epoch = full pass through dataloader (~877 iterations)")
        if config.use_compile:
            print(f"  Note: Using {warmup_epochs} warmup epochs to allow torch.compile to optimize")

    times = []
    training_start = time.time()
    time_to_second_epoch = None

    # Per-epoch tracking
    per_epoch_times = []
    per_epoch_throughput = []
    per_epoch_memory_peaks = []

    # GPU utilization tracking
    gpu_stats_samples = []
    gpu_sample_interval = 100  # Sample every 100 iterations

    # Multi-GPU sync overhead tracking
    total_sync_time = 0.0

    # Epoch loop (matching real training)
    for epoch in range(total_epochs):
        # CRITICAL: Set epoch on sampler before iteration (enables proper shuffling)
        if use_multi_gpu and hasattr(dataloader.sampler, 'set_epoch'):
            sync_start = time.time()
            dataloader.sampler.set_epoch(epoch)
            if use_multi_gpu:
                dist.barrier()
                total_sync_time += (time.time() - sync_start)

        epoch_times = []
        epoch_start = time.time()
        iteration_count = 0

        # Full dataloader iteration (matching real training)
        for step, batch in enumerate(dataloader):
            # Use mode to prepare batch (matching trainer)
            prepared = mode.prepare_batch(batch, device)
            images = prepared['images']
            labels_dict = {'labels': prepared.get('labels')}

            # Time this iteration
            torch.cuda.synchronize()
            start = time.time()

            # Training step (matching trainer.train_step)
            with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
                noise = torch.randn_like(images).to(device)
                timesteps = strategy.sample_timesteps(images)
                noisy_images = strategy.add_noise(images, noise, timesteps)
                model_input = mode.format_model_input(noisy_images, labels_dict)
                prediction = strategy.predict_noise_or_velocity(model, model_input, timesteps)
                mse_loss, predicted_clean = strategy.compute_loss(prediction, images, noise, timesteps)
                perc_loss = perceptual_loss(predicted_clean.float(), images.float())
                total_loss = mse_loss + 0.001 * perc_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            end = time.time()

            iteration_time = end - start
            epoch_times.append(iteration_time)

            # Only record times after warmup epochs
            if epoch >= warmup_epochs:
                times.append(iteration_time)

            monitor.update()

            # Sample GPU stats periodically (not every iteration to minimize overhead)
            if iteration_count % gpu_sample_interval == 0:
                gpu_stats = get_gpu_stats(device.index if device.index is not None else 0)
                if gpu_stats:
                    gpu_stats_samples.append(gpu_stats)

            iteration_count += 1

        # Track per-epoch metrics
        epoch_duration = time.time() - epoch_start
        avg_it_per_sec = len(epoch_times) / sum(epoch_times) if epoch_times else 0
        per_epoch_times.append(epoch_duration)
        per_epoch_throughput.append(avg_it_per_sec)

        # Track memory per epoch
        epoch_memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)  # GB
        epoch_memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)  # GB
        per_epoch_memory_peaks.append({
            'allocated_gb': epoch_memory_allocated,
            'reserved_gb': epoch_memory_reserved
        })

        # Track time to second epoch (includes first epoch + compile overhead)
        if epoch == 1 and time_to_second_epoch is None:
            time_to_second_epoch = time.time() - training_start

        # Log epoch metrics (rank 0 only)
        if is_main:
            status = "warmup" if epoch < warmup_epochs else "measured"
            memory_stats = monitor.get_stats()
            allocation_efficiency = epoch_memory_allocated / epoch_memory_reserved if epoch_memory_reserved > 0 else 0

            print(f"\n  Epoch {epoch + 1}/{total_epochs} ({status}):")
            print(f"    Duration: {epoch_duration:.1f}s")
            print(f"    Iterations: {len(epoch_times)}")
            print(f"    Avg it/s: {avg_it_per_sec:.2f}")
            print(f"    Iteration times: min={min(epoch_times):.3f}s, max={max(epoch_times):.3f}s, std={np.std(epoch_times):.3f}s")
            print(f"    VRAM (peak): {memory_stats['peak_gb']:.2f} GB, efficiency: {allocation_efficiency:.2f}")

            # Show time to second epoch after first epoch completes
            if epoch == 0:
                print(f"    Time to complete epoch 1 (includes compile): {epoch_duration:.1f}s")

    # Compute statistics (only on rank 0)
    if is_main:
        times = np.array(times)
        it_per_sec = 1.0 / np.mean(times)
        it_per_sec_std = np.std(1.0 / times)

        # Account for multi-GPU: total throughput = single GPU throughput * world_size
        total_it_per_sec = it_per_sec * world_size

        memory_stats = monitor.get_stats()

        # Calculate full training estimates (877 it/epoch × 500 epochs = 438,500 iterations)
        iterations_per_epoch = 877
        total_epochs = 500
        total_training_iterations = iterations_per_epoch * total_epochs
        estimated_training_time_sec = total_training_iterations / total_it_per_sec
        estimated_training_hours = estimated_training_time_sec / 3600

        # Calculate convergence epoch (when throughput stabilizes, CV < 5%)
        measured_throughput = per_epoch_throughput[warmup_epochs:]
        convergence_epoch = None
        for i in range(2, len(measured_throughput)):
            recent = measured_throughput[i-2:i+1]
            cv = np.std(recent) / np.mean(recent) if np.mean(recent) > 0 else 1.0
            if cv < 0.05:
                convergence_epoch = warmup_epochs + i
                break

        # Calculate memory efficiency metrics using PEAK values (not end-of-epoch)
        peak_allocated_gb = memory_stats['peak_allocated_gb']
        peak_reserved_gb = memory_stats['peak_gb']
        allocation_efficiency = peak_allocated_gb / peak_reserved_gb if peak_reserved_gb > 0 else 0
        fragmentation_ratio = (peak_reserved_gb - peak_allocated_gb) / peak_reserved_gb if peak_reserved_gb > 0 else 0
        memory_per_sample = (peak_allocated_gb * 1024) / args.batch_size  # MB per sample
        samples_per_gb = args.batch_size / peak_allocated_gb if peak_allocated_gb > 0 else 0

        # Get detailed memory summary for diagnostics
        mem_summary = monitor.get_memory_summary()

        # Calculate GPU utilization averages
        gpu_metrics = {}
        if gpu_stats_samples:
            gpu_metrics = {
                'avg_gpu_compute_util_pct': np.mean([s['gpu_compute_util_pct'] for s in gpu_stats_samples]),
                'avg_gpu_memory_util_pct': np.mean([s['gpu_memory_util_pct'] for s in gpu_stats_samples]),
                'avg_power_draw_watts': np.mean([s['power_draw_watts'] for s in gpu_stats_samples]),
                'peak_temperature_celsius': max([s['temperature_celsius'] for s in gpu_stats_samples]),
                'thermal_throttling_detected': any(s['temperature_celsius'] > 83 for s in gpu_stats_samples)
            }

        # Calculate multi-GPU metrics
        multi_gpu_metrics = {}
        if use_multi_gpu:
            total_training_time = sum(per_epoch_times)
            sync_overhead_pct = (total_sync_time / total_training_time * 100) if total_training_time > 0 else 0
            actual_speedup = total_it_per_sec / it_per_sec if it_per_sec > 0 else 0
            communication_efficiency = actual_speedup / world_size if world_size > 0 else 0

            multi_gpu_metrics = {
                'total_sync_time_sec': total_sync_time,
                'sync_overhead_pct': sync_overhead_pct,
                'communication_efficiency': communication_efficiency,
                'actual_speedup': actual_speedup
            }

        results = {
            'config': config.name,
            'use_amp': config.use_amp,
            'use_compile': config.use_compile,
            'use_optimizations': config.use_optimizations,
            'compile_mode': config.compile_mode if config.use_compile else None,
            'multi_gpu': use_multi_gpu,
            'num_gpus': world_size,
            'warmup_epochs': warmup_epochs,
            'measured_epochs': args.train_epochs,
            'total_iterations': len(times),
            'time_to_second_epoch_sec': time_to_second_epoch,
            'mean_it_per_sec_per_gpu': it_per_sec,
            'total_it_per_sec': total_it_per_sec,
            'std_it_per_sec': it_per_sec_std,
            'mean_sec_per_it': np.mean(times),
            'std_sec_per_it': np.std(times),
            'vram_peak_gb': memory_stats['peak_gb'],
            'vram_current_gb': memory_stats['current_gb'],
            'estimated_full_training_hours': estimated_training_hours,
            # Per-epoch metrics (enhancement #1)
            'per_epoch_times': per_epoch_times,
            'per_epoch_throughput': per_epoch_throughput,
            'convergence_epoch': convergence_epoch,
            # Memory efficiency metrics (enhancement #2)
            'memory_metrics': {
                'peak_allocated_gb': peak_allocated_gb,
                'peak_reserved_gb': peak_reserved_gb,
                'allocation_efficiency': allocation_efficiency,
                'fragmentation_ratio': fragmentation_ratio,
                'memory_per_sample_mb': memory_per_sample,
                'samples_per_gb': samples_per_gb,
                'per_epoch_memory_peaks': per_epoch_memory_peaks,
                'detailed_summary': mem_summary
            },
            # GPU utilization metrics (enhancement #6)
            'gpu_utilization': gpu_metrics,
            # Multi-GPU metrics (enhancement #8)
            'multi_gpu_metrics': multi_gpu_metrics if use_multi_gpu else None
        }

        print(f"\nFinal Results ({args.train_epochs} measured epochs, excluding {warmup_epochs} warmup epochs):")
        if use_multi_gpu:
            print(f"  Iterations/sec (per GPU): {it_per_sec:.2f} ± {it_per_sec_std:.2f}")
            print(f"  Total throughput ({world_size} GPUs): {total_it_per_sec:.2f} it/s")
        else:
            print(f"  Iterations/sec: {it_per_sec:.2f} ± {it_per_sec_std:.2f}")
        print(f"  Seconds/iteration: {np.mean(times):.3f} ± {np.std(times):.3f}")

        print(f"\n  Memory Efficiency:")
        print(f"    Peak reserved: {peak_reserved_gb:.2f} GB (what nvidia-smi shows)")
        print(f"    Peak allocated: {peak_allocated_gb:.2f} GB (actual tensors)")
        print(f"    Allocation efficiency: {allocation_efficiency:.2%} (allocated/reserved)")
        print(f"    Fragmentation: {fragmentation_ratio:.2%} (wasted space)")
        print(f"    Memory per sample: {memory_per_sample:.1f} MB")
        print(f"    Throughput density: {samples_per_gb:.1f} samples/GB")
        print(f"\n  Memory Breakdown (detailed):")
        print(f"    Active memory: {mem_summary['active_gb']:.2f} GB (in use)")
        print(f"    Inactive/cached: {mem_summary['inactive_gb']:.2f} GB (freed but cached)")
        print(f"    Total allocations: {mem_summary['num_allocations']:,}")
        print(f"    Total frees: {mem_summary['num_frees']:,}")

        if convergence_epoch:
            print(f"\n  Convergence: Throughput stabilized at epoch {convergence_epoch}")

        if gpu_metrics:
            print(f"\n  GPU Utilization (sampled {len(gpu_stats_samples)} times):")
            print(f"    Compute: {gpu_metrics['avg_gpu_compute_util_pct']:.1f}%")
            print(f"    Memory BW: {gpu_metrics['avg_gpu_memory_util_pct']:.1f}%")
            print(f"    Power: {gpu_metrics['avg_power_draw_watts']:.1f} W")
            print(f"    Peak temp: {gpu_metrics['peak_temperature_celsius']:.1f}°C" +
                  (" [THROTTLING DETECTED]" if gpu_metrics['thermal_throttling_detected'] else ""))

        if use_multi_gpu and multi_gpu_metrics:
            print(f"\n  Multi-GPU Efficiency:")
            print(f"    Sync overhead: {multi_gpu_metrics['sync_overhead_pct']:.2f}%")
            print(f"    Actual speedup: {multi_gpu_metrics['actual_speedup']:.2f}x ({world_size} GPUs)")
            print(f"    Communication efficiency: {multi_gpu_metrics['communication_efficiency']:.2%}")

        print(f"\n  Estimated full training time ({total_epochs} epochs, {iterations_per_epoch} it/epoch):")
        print(f"    Total iterations: {total_training_iterations:,}")
        print(f"    Estimated time: {estimated_training_hours:.1f} hours ({estimated_training_hours/24:.1f} days)")
    else:
        results = None

    # Cleanup
    del model, optimizer, perceptual_loss, strategy
    torch.cuda.empty_cache()

    # Synchronize before returning
    if use_multi_gpu:
        dist.barrier()

    return results


def benchmark_generation(config, args, train_dataset, device):
    """
    Benchmark generation performance.

    Args:
        config: BenchmarkConfig
        args: Command line arguments
        train_dataset: Training dataset (for getting image stats)
        device: torch device

    Returns:
        dict: Results including samples/s and VRAM usage
    """
    print(f"\n{'=' * 80}")
    print(f"GENERATION BENCHMARK: {config.name}")
    print(f"AMP: {config.use_amp} | Compile: {config.use_compile}")
    print(f"{'=' * 80}\n")

    # Reset GPU memory
    monitor = GPUMonitor(device_id=device.index if device.index is not None else 0)
    monitor.reset()

    # Create mode and get model config
    mode = SegmentationMode()
    model_config = mode.get_model_config()

    # Create model (matching DiffusionTrainer.setup_model)
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=model_config['in_channels'],
        out_channels=model_config['out_channels'],
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=1,
        num_head_channels=256
    ).to(device)
    model.eval()

    # Apply compile if needed
    if config.use_compile:
        # Determine compile mode
        if config.compile_mode is not None:
            compile_mode = config.compile_mode
        else:
            compile_mode = "max-autotune"  # Default for generation

        print(f"Compiling model for generation (mode={compile_mode})...")
        compile_start = time.time()
        model = torch.compile(model, mode=compile_mode)
        compile_time = time.time() - compile_start
        print(f"Compilation completed in {compile_time:.2f}s")
    else:
        compile_time = 0.0

    # Create strategy and setup scheduler
    strategy = DDPMStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=args.image_size)

    # Warmup - torch.compile needs multiple passes to optimize
    warmup_passes = 3 if config.use_compile else 1
    print(f"Warming up ({warmup_passes} generation passes)...")
    if config.use_compile:
        print("  Note: torch.compile optimizes over multiple generations")

    warmup_start = time.time()

    with torch.no_grad():
        for i in range(warmup_passes):
            noise = torch.randn((args.gen_batch_size, 1, args.image_size, args.image_size), device=device)
            # For seg mode: no conditioning, just pass noise through
            model_input = mode.format_model_input(noise, {'labels': None})
            with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
                _ = strategy.generate(model, model_input, num_steps=args.gen_steps, device=device)

            if config.use_compile:
                print(f"  Warmup pass {i + 1}/{warmup_passes}")

    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    monitor.update()
    print(f"Warmup complete ({warmup_time:.2f}s)")

    # Actual benchmark
    print(f"Generating {args.gen_iterations} batches of {args.gen_batch_size} images...")
    times = []

    with torch.no_grad():
        for i in range(args.gen_iterations):
            noise = torch.randn((args.gen_batch_size, 1, args.image_size, args.image_size), device=device)
            # For seg mode: no conditioning, just pass noise through
            model_input = mode.format_model_input(noise, {'labels': None})

            torch.cuda.synchronize()
            start = time.time()

            with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
                samples = strategy.generate(model, model_input, num_steps=args.gen_steps, device=device)

            torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)
            monitor.update()

    # Compute statistics
    times = np.array(times)
    samples_per_sec = args.gen_batch_size / np.mean(times)
    samples_per_sec_std = args.gen_batch_size / np.std(times)

    memory_stats = monitor.get_stats()

    # Calculate full generation estimates (15,000 total images)
    total_images_to_generate = 15000
    estimated_generation_time_sec = total_images_to_generate / samples_per_sec
    estimated_generation_hours = estimated_generation_time_sec / 3600

    results = {
        'config': config.name,
        'use_amp': config.use_amp,
        'use_compile': config.use_compile,
        'use_optimizations': config.use_optimizations,
        'compile_mode': config.compile_mode if config.use_compile else None,
        'compile_time_sec': compile_time,
        'warmup_time_sec': warmup_time,
        'iterations': args.gen_iterations,
        'batch_size': args.gen_batch_size,
        'gen_steps': args.gen_steps,
        'mean_samples_per_sec': samples_per_sec,
        'std_samples_per_sec': samples_per_sec_std,
        'mean_sec_per_batch': np.mean(times),
        'std_sec_per_batch': np.std(times),
        'vram_peak_gb': memory_stats['peak_gb'],
        'vram_current_gb': memory_stats['current_gb'],
        'estimated_full_generation_hours': estimated_generation_hours
    }

    print(f"\nResults:")
    if config.use_compile:
        print(f"  Compilation time: {compile_time:.2f}s")
        print(f"  Warmup time: {warmup_time:.2f}s")
    print(f"  Samples/sec: {samples_per_sec:.2f} ± {samples_per_sec_std:.2f}")
    print(f"  Seconds/batch ({args.gen_batch_size} images): {np.mean(times):.3f} ± {np.std(times):.3f}")
    print(f"  VRAM (peak): {memory_stats['peak_gb']:.2f} GB")
    print(f"  VRAM (current): {memory_stats['current_gb']:.2f} GB")
    print(f"\n  Estimated full generation time ({total_images_to_generate:,} images):")
    print(f"    Estimated time: {estimated_generation_hours:.1f} hours ({estimated_generation_hours/24:.1f} days)")

    # Cleanup
    del model, strategy
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark training and generation performance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # General options
    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'both'], default='both',
                        help='Benchmark mode (default: both)')
    parser.add_argument('--image_size', type=int, default=128, choices=[128, 256],
                        help='Image size (default: 128)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--compute', type=str, choices=['local', 'cluster'], default='local',
                        help='Compute environment (default: local)')

    # Training benchmark options
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='DEPRECATED: Warmup epochs are now automatic (3 with compile, 1 without). This argument is ignored.')
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='Number of training epochs to benchmark (default: 10)')

    # Generation benchmark options
    parser.add_argument('--gen_iterations', type=int, default=10,
                        help='Number of generation iterations to benchmark (default: 10)')
    parser.add_argument('--gen_batch_size', type=int, default=4,
                        help='Batch size for generation (default: 4)')
    parser.add_argument('--gen_steps', type=int, default=1000,
                        help='Number of diffusion steps for generation (default: 1000)')

    # Output options
    parser.add_argument('--output', type=str, default='benchmark_results',
                        help='Output directory for results (default: benchmark_results)')

    # Multi-GPU options
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Use multi-GPU training with DDP')

    # Compile mode testing
    parser.add_argument('--test-compile-modes', action='store_true',
                        help='Test different torch.compile modes (default, reduce-overhead, max-autotune) for training and generation')

    args = parser.parse_args()

    # Setup device and distributed
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    if args.multi_gpu:
        # Setup distributed training
        rank, local_rank, world_size, device = setup_distributed()
        is_main = (rank == 0)
    else:
        device = torch.device('cuda:0')
        rank = 0
        local_rank = 0
        world_size = 1
        is_main = True

    if is_main:
        print(f"\n{'=' * 80}")
        print(f"PERFORMANCE BENCHMARK")
        print(f"{'=' * 80}")
        if args.multi_gpu:
            print(f"Multi-GPU: {world_size} GPUs")
            print(f"Device (rank {rank}): {torch.cuda.get_device_name(local_rank)}")
        else:
            print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Mode: {args.mode}")
        print(f"{'=' * 80}\n")

    # Define configurations to test
    # Config format: (name, use_amp, use_compile, use_optimizations, compile_mode)
    configs = [
        ("Baseline (FP32, no opts)", False, False, False, None),
        ("Baseline (FP32, standard opts)", False, False, True, None),
        ("AMP only", True, False, True, None),
        ("AMP + Compile", True, True, True, None),
    ]

    # Add compile mode tests if requested
    if args.test_compile_modes:
        # Add 3 compile mode variants for training and generation
        compile_modes = ["default", "reduce-overhead", "max-autotune"]
        for mode in compile_modes:
            configs.append((f"AMP + Compile ({mode})", True, True, True, mode))

    all_results = {
        'training': [],
        'generation': []
    }

    # Setup PathConfig for data paths
    path_config = PathConfig(compute=args.compute)
    data_prefix = str(path_config.base_prefix)

    # Setup data (only if training benchmark)
    if args.mode in ['train', 'both']:
        if is_main:
            print("Loading training data...")

        dataloader, train_dataset = create_dataloader(
            prefix=data_prefix,
            image_type='seg',
            image_size=args.image_size,
            batch_size=args.batch_size,
            use_distributed=args.multi_gpu,
            rank=rank,
            world_size=world_size
        )
        if is_main:
            print(f"Loaded {len(train_dataset)} training samples\n")
    else:
        train_dataset = None
        dataloader = None

    # Run benchmarks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for config_tuple in configs:
        # Create BenchmarkConfig from tuple
        config = BenchmarkConfig(*config_tuple)

        # Set optimizations based on config
        if config.use_optimizations:
            enable_standard_optimizations()

        # Training benchmark
        if args.mode in ['train', 'both']:
            try:
                results = benchmark_training(config, args, dataloader, train_dataset, device,
                                            rank, world_size, local_rank)
                # Only rank 0 collects results
                if is_main and results is not None:
                    all_results['training'].append(results)
            except Exception as e:
                if is_main:
                    print(f"Error in training benchmark: {e}")
                    import traceback
                    traceback.print_exc()

        # Generation benchmark
        if args.mode in ['generate', 'both']:
            # Need dataset for generation (for normalization stats)
            if train_dataset is None:
                print("Loading data for generation benchmark...")

                _, train_dataset = create_dataloader(
                    prefix=data_prefix,
                    image_type='seg',
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    use_distributed=False,
                    rank=0,
                    world_size=1
                )

            try:
                results = benchmark_generation(config, args, train_dataset, device)
                all_results['generation'].append(results)
            except Exception as e:
                print(f"Error in generation benchmark: {e}")
                import traceback
                traceback.print_exc()

        # Clear memory and compilation cache between configs
        torch.cuda.empty_cache()
        # Clear torch.compile cache to ensure fresh compilation for each mode
        torch._dynamo.reset()
        time.sleep(2)

    # Save and display results (only on rank 0)
    if is_main:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training results
        if all_results['training']:
            df_train = pd.DataFrame(all_results['training'])
            print(f"\n{'=' * 80}")
            print("TRAINING BENCHMARK SUMMARY")
            print(f"{'=' * 80}")
            print(df_train.to_string(index=False))
            print(f"{'=' * 80}\n")

            # Save to CSV
            csv_path = output_dir / f'training_benchmark_{timestamp}.csv'
            df_train.to_csv(csv_path, index=False)
            print(f"Training results saved to: {csv_path}")

        # Generation results
        if all_results['generation']:
            df_gen = pd.DataFrame(all_results['generation'])
            print(f"\n{'=' * 80}")
            print("GENERATION BENCHMARK SUMMARY")
            print(f"{'=' * 80}")
            print(df_gen.to_string(index=False))
            print(f"{'=' * 80}\n")

            # Save to CSV
            csv_path = output_dir / f'generation_benchmark_{timestamp}.csv'
            df_gen.to_csv(csv_path, index=False)
            print(f"Generation results saved to: {csv_path}")

        # Save JSON for programmatic access
        json_path = output_dir / f'benchmark_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'args': vars(args),
                'device': torch.cuda.get_device_name(local_rank),
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__,
                'multi_gpu': args.multi_gpu,
                'world_size': world_size,
                'results': all_results
            }, f, indent=2)
        print(f"Full results saved to: {json_path}\n")

    # Cleanup distributed
    if args.multi_gpu:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
