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
  python misc/bench/benchmark.py --mode both --image_size 128 --train_epochs 2

  # Local multi-GPU (requires torchrun)
  torchrun --nproc_per_node=4 misc/bench/benchmark.py --mode both --multi_gpu

  # Cluster single GPU
  sbatch IDUN/misc/benchmark.slurm

  # Cluster multi-GPU (4 GPUs)
  sbatch IDUN/misc/benchmark_multigpu.slurm

Arguments:
  --mode {train,generate,both}  What to benchmark (default: both)
  --image_size {128,256}        Image resolution (default: 128)
  --batch_size N                Training batch size (default: 16)
  --train_epochs N              Training epochs to benchmark (default: 2)
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
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import PathConfig
from monai.networks.nets import DiffusionModelUNet
from monai.losses import PerceptualLoss

# Import from medgen package
from medgen.diffusion import DDPMStrategy, SegmentationMode
from medgen.core import setup_cuda_optimizations


class GPUMonitor:
    """Monitor GPU memory usage during benchmark."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id

    def update(self) -> None:
        """Update memory stats (forces sync)."""
        torch.cuda.synchronize(self.device_id)

    def reset(self) -> None:
        """Reset peak memory tracking."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device_id)

    def get_stats(self) -> Dict[str, float]:
        """Get current stats using PyTorch's built-in tracking."""
        torch.cuda.synchronize(self.device_id)
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

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get detailed memory breakdown."""
        torch.cuda.synchronize(self.device_id)
        mem_stats = torch.cuda.memory_stats(self.device_id)

        allocated = mem_stats.get('allocated_bytes.all.current', 0) / (1024 ** 3)
        reserved = mem_stats.get('reserved_bytes.all.current', 0) / (1024 ** 3)
        peak_allocated = mem_stats.get('allocated_bytes.all.peak', 0) / (1024 ** 3)
        peak_reserved = mem_stats.get('reserved_bytes.all.peak', 0) / (1024 ** 3)
        active = mem_stats.get('active_bytes.all.current', 0) / (1024 ** 3)
        inactive = mem_stats.get('inactive_split_bytes.all.current', 0) / (1024 ** 3)
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

    def __init__(
        self,
        name: str,
        use_amp: bool,
        use_compile: bool,
        use_optimizations: bool,
        compile_mode: Optional[str] = None
    ):
        self.name = name
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.use_optimizations = use_optimizations
        self.compile_mode = compile_mode


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """Setup distributed training."""
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


def get_gpu_stats(device_id: int = 0) -> Optional[Dict[str, float]]:
    """Query nvidia-smi for detailed GPU stats."""
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
    except Exception:
        pass
    return None


def create_synthetic_dataset(
    num_samples: int,
    image_size: int,
    seed: int = 42
) -> Dataset:
    """Create synthetic dataset for benchmarking."""

    class SyntheticSegDataset(Dataset):
        """Synthetic segmentation dataset for benchmarking."""

        def __init__(self, num_samples: int, image_size: int, seed: int):
            self.num_samples = num_samples
            self.image_size = image_size
            self.seed = seed

        def __len__(self) -> int:
            return self.num_samples

        def __getitem__(self, idx: int) -> torch.Tensor:
            # Generate deterministic random data
            gen = torch.Generator()
            gen.manual_seed(self.seed + idx)
            # Return segmentation mask (1 channel)
            data = torch.rand((1, self.image_size, self.image_size), generator=gen)
            # Make it binary-ish
            data = (data > 0.5).float()
            return data

    return SyntheticSegDataset(num_samples, image_size, seed)


def create_benchmark_dataloader(
    dataset: Dataset,
    batch_size: int,
    use_distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create dataloader for benchmarking."""
    if use_distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


def benchmark_training(
    config: BenchmarkConfig,
    args: argparse.Namespace,
    dataloader: DataLoader,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0
) -> Optional[Dict[str, Any]]:
    """Benchmark training performance."""
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

    # Create model
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
    compile_time = 0.0
    if use_multi_gpu:
        if is_main:
            print(f"Wrapping model with DDP (device_ids=[{local_rank}])")
        ddp_model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=False
        )

        if config.use_compile:
            compile_mode = config.compile_mode or "reduce-overhead"
            if is_main:
                print(f"Compiling DDP model (mode={compile_mode})...")
            compile_start = time.time()
            model = torch.compile(ddp_model, mode=compile_mode)
            compile_time = time.time() - compile_start
            if is_main:
                print(f"Compilation completed in {compile_time:.2f}s")
        else:
            model = ddp_model
    else:
        if config.use_compile:
            compile_mode = config.compile_mode or "default"
            if is_main:
                print(f"Compiling model (mode={compile_mode})...")
            compile_start = time.time()
            model = torch.compile(raw_model, mode=compile_mode)
            compile_time = time.time() - compile_start
            if is_main:
                print(f"Compilation completed in {compile_time:.2f}s")
        else:
            model = raw_model

    # Create optimizer
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=1e-4)

    # Create perceptual loss
    cache_dir = project_root / 'model_cache'
    cache_dir.mkdir(exist_ok=True)

    perceptual_loss = PerceptualLoss(
        spatial_dims=2,
        network_type="squeeze",
        is_fake_3d=False,
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
    total_epochs = warmup_epochs + args.train_epochs

    if is_main:
        print(f"Running {total_epochs} epochs ({warmup_epochs} warmup + {args.train_epochs} measured)")

    times: List[float] = []
    per_epoch_times: List[float] = []
    per_epoch_throughput: List[float] = []
    gpu_stats_samples: List[Dict] = []
    gpu_sample_interval = 100

    for epoch in range(total_epochs):
        if use_multi_gpu and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            dist.barrier()

        epoch_times: List[float] = []
        epoch_start = time.time()
        iteration_count = 0

        for step, batch in enumerate(dataloader):
            images = batch.to(device)

            torch.cuda.synchronize()
            start = time.time()

            with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
                noise = torch.randn_like(images).to(device)
                timesteps = strategy.sample_timesteps(images)
                noisy_images = strategy.add_noise(images, noise, timesteps)
                prediction = model(noisy_images, timesteps)
                mse_loss = torch.nn.functional.mse_loss(prediction, noise)

                # Compute predicted clean for perceptual loss
                predicted_clean = strategy.predict_x0_from_noise(noisy_images, prediction, timesteps)
                perc_loss = perceptual_loss(predicted_clean.float(), images.float())
                total_loss = mse_loss + 0.001 * perc_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            end = time.time()

            iteration_time = end - start
            epoch_times.append(iteration_time)

            if epoch >= warmup_epochs:
                times.append(iteration_time)

            monitor.update()

            if iteration_count % gpu_sample_interval == 0:
                gpu_stats = get_gpu_stats(device.index if device.index is not None else 0)
                if gpu_stats:
                    gpu_stats_samples.append(gpu_stats)

            iteration_count += 1

        epoch_duration = time.time() - epoch_start
        avg_it_per_sec = len(epoch_times) / sum(epoch_times) if epoch_times else 0
        per_epoch_times.append(epoch_duration)
        per_epoch_throughput.append(avg_it_per_sec)

        if is_main:
            status = "warmup" if epoch < warmup_epochs else "measured"
            memory_stats = monitor.get_stats()
            print(f"\n  Epoch {epoch + 1}/{total_epochs} ({status}):")
            print(f"    Duration: {epoch_duration:.1f}s, Iterations: {len(epoch_times)}")
            print(f"    Avg it/s: {avg_it_per_sec:.2f}, VRAM peak: {memory_stats['peak_gb']:.2f} GB")

    # Compute statistics (only on rank 0)
    if is_main:
        times_arr = np.array(times)
        it_per_sec = 1.0 / np.mean(times_arr) if len(times_arr) > 0 else 0
        it_per_sec_std = np.std(1.0 / times_arr) if len(times_arr) > 0 else 0
        total_it_per_sec = it_per_sec * world_size

        memory_stats = monitor.get_stats()

        # Calculate full training estimates
        iterations_per_epoch = 877
        full_epochs = 500
        total_training_iterations = iterations_per_epoch * full_epochs
        estimated_training_time_sec = total_training_iterations / total_it_per_sec if total_it_per_sec > 0 else 0
        estimated_training_hours = estimated_training_time_sec / 3600

        results = {
            'config': config.name,
            'use_amp': config.use_amp,
            'use_compile': config.use_compile,
            'compile_mode': config.compile_mode if config.use_compile else None,
            'multi_gpu': use_multi_gpu,
            'num_gpus': world_size,
            'warmup_epochs': warmup_epochs,
            'measured_epochs': args.train_epochs,
            'total_iterations': len(times),
            'mean_it_per_sec_per_gpu': it_per_sec,
            'total_it_per_sec': total_it_per_sec,
            'std_it_per_sec': it_per_sec_std,
            'vram_peak_gb': memory_stats['peak_gb'],
            'estimated_full_training_hours': estimated_training_hours,
        }

        print(f"\nFinal Results:")
        if use_multi_gpu:
            print(f"  Iterations/sec (per GPU): {it_per_sec:.2f}")
            print(f"  Total throughput ({world_size} GPUs): {total_it_per_sec:.2f} it/s")
        else:
            print(f"  Iterations/sec: {it_per_sec:.2f}")
        print(f"  VRAM peak: {memory_stats['peak_gb']:.2f} GB")
        print(f"  Estimated full training: {estimated_training_hours:.1f} hours")
    else:
        results = None

    # Cleanup
    del model, optimizer, perceptual_loss, strategy
    torch.cuda.empty_cache()

    if use_multi_gpu:
        dist.barrier()

    return results


def benchmark_generation(
    config: BenchmarkConfig,
    args: argparse.Namespace,
    device: torch.device
) -> Dict[str, Any]:
    """Benchmark generation performance."""
    print(f"\n{'=' * 80}")
    print(f"GENERATION BENCHMARK: {config.name}")
    print(f"AMP: {config.use_amp} | Compile: {config.use_compile}")
    print(f"{'=' * 80}\n")

    monitor = GPUMonitor(device_id=device.index if device.index is not None else 0)
    monitor.reset()

    mode = SegmentationMode()
    model_config = mode.get_model_config()

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

    compile_time = 0.0
    if config.use_compile:
        compile_mode = config.compile_mode or "max-autotune"
        print(f"Compiling model for generation (mode={compile_mode})...")
        compile_start = time.time()
        model = torch.compile(model, mode=compile_mode)
        compile_time = time.time() - compile_start
        print(f"Compilation completed in {compile_time:.2f}s")

    strategy = DDPMStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=args.image_size)

    # Warmup
    warmup_passes = 3 if config.use_compile else 1
    print(f"Warming up ({warmup_passes} generation passes)...")

    warmup_start = time.time()
    with torch.no_grad():
        for i in range(warmup_passes):
            noise = torch.randn((args.gen_batch_size, 1, args.image_size, args.image_size), device=device)
            with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
                _ = strategy.generate(model, noise, num_steps=args.gen_steps, device=device)
            if config.use_compile:
                print(f"  Warmup pass {i + 1}/{warmup_passes}")

    torch.cuda.synchronize()
    warmup_time = time.time() - warmup_start
    print(f"Warmup complete ({warmup_time:.2f}s)")

    # Actual benchmark
    print(f"Generating {args.gen_iterations} batches of {args.gen_batch_size} images...")
    times: List[float] = []

    with torch.no_grad():
        for i in range(args.gen_iterations):
            noise = torch.randn((args.gen_batch_size, 1, args.image_size, args.image_size), device=device)

            torch.cuda.synchronize()
            start = time.time()

            with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16):
                _ = strategy.generate(model, noise, num_steps=args.gen_steps, device=device)

            torch.cuda.synchronize()
            end = time.time()

            times.append(end - start)
            monitor.update()

    times_arr = np.array(times)
    samples_per_sec = args.gen_batch_size / np.mean(times_arr)
    memory_stats = monitor.get_stats()

    total_images = 15000
    estimated_hours = (total_images / samples_per_sec) / 3600

    results = {
        'config': config.name,
        'use_amp': config.use_amp,
        'use_compile': config.use_compile,
        'compile_mode': config.compile_mode if config.use_compile else None,
        'compile_time_sec': compile_time,
        'warmup_time_sec': warmup_time,
        'iterations': args.gen_iterations,
        'batch_size': args.gen_batch_size,
        'gen_steps': args.gen_steps,
        'mean_samples_per_sec': samples_per_sec,
        'vram_peak_gb': memory_stats['peak_gb'],
        'estimated_full_generation_hours': estimated_hours
    }

    print(f"\nResults:")
    print(f"  Samples/sec: {samples_per_sec:.2f}")
    print(f"  VRAM peak: {memory_stats['peak_gb']:.2f} GB")
    print(f"  Estimated {total_images:,} images: {estimated_hours:.1f} hours")

    del model, strategy
    torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Benchmark training and generation performance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', type=str, choices=['train', 'generate', 'both'], default='both')
    parser.add_argument('--image_size', type=int, default=128, choices=[128, 256])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--compute', type=str, choices=['local', 'cluster'], default='local')
    parser.add_argument('--train_epochs', type=int, default=2)
    parser.add_argument('--gen_iterations', type=int, default=10)
    parser.add_argument('--gen_batch_size', type=int, default=4)
    parser.add_argument('--gen_steps', type=int, default=1000)
    parser.add_argument('--output', type=str, default='benchmark_results')
    parser.add_argument('--multi_gpu', action='store_true')
    parser.add_argument('--test-compile-modes', action='store_true')

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return

    if args.multi_gpu:
        rank, local_rank, world_size, device = setup_distributed()
        is_main = (rank == 0)
    else:
        device = torch.device('cuda:0')
        rank = local_rank = 0
        world_size = 1
        is_main = True

    if is_main:
        print(f"\n{'=' * 80}")
        print(f"PERFORMANCE BENCHMARK")
        print(f"{'=' * 80}")
        print(f"Device: {torch.cuda.get_device_name(local_rank)}")
        print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
        print(f"Image size: {args.image_size}, Mode: {args.mode}")
        if args.multi_gpu:
            print(f"Multi-GPU: {world_size} GPUs")
        print(f"{'=' * 80}\n")

    # Setup CUDA optimizations
    setup_cuda_optimizations()

    # Define configurations
    configs = [
        BenchmarkConfig("Baseline (FP32)", False, False, False),
        BenchmarkConfig("AMP only", True, False, True),
        BenchmarkConfig("AMP + Compile", True, True, True),
    ]

    if args.test_compile_modes:
        for mode in ["default", "reduce-overhead", "max-autotune"]:
            configs.append(BenchmarkConfig(f"AMP + Compile ({mode})", True, True, True, mode))

    all_results: Dict[str, List] = {'training': [], 'generation': []}

    # Create synthetic dataset for benchmarking
    if args.mode in ['train', 'both']:
        if is_main:
            print("Creating synthetic dataset for benchmarking...")
        dataset = create_synthetic_dataset(
            num_samples=1000,
            image_size=args.image_size,
        )
        dataloader = create_benchmark_dataloader(
            dataset,
            batch_size=args.batch_size,
            use_distributed=args.multi_gpu,
            rank=rank,
            world_size=world_size,
        )
        if is_main:
            print(f"Created {len(dataset)} synthetic samples\n")
    else:
        dataloader = None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for config in configs:
        # Training benchmark
        if args.mode in ['train', 'both'] and dataloader is not None:
            try:
                results = benchmark_training(
                    config, args, dataloader, device, rank, world_size, local_rank
                )
                if is_main and results is not None:
                    all_results['training'].append(results)
            except Exception as e:
                if is_main:
                    print(f"Error in training benchmark: {e}")
                    import traceback
                    traceback.print_exc()

        # Generation benchmark (only on rank 0)
        if args.mode in ['generate', 'both'] and is_main:
            try:
                results = benchmark_generation(config, args, device)
                all_results['generation'].append(results)
            except Exception as e:
                print(f"Error in generation benchmark: {e}")
                import traceback
                traceback.print_exc()

        torch.cuda.empty_cache()
        torch._dynamo.reset()
        time.sleep(2)

    # Save results (only on rank 0)
    if is_main:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if all_results['training']:
            df_train = pd.DataFrame(all_results['training'])
            print(f"\n{'=' * 80}")
            print("TRAINING BENCHMARK SUMMARY")
            print(f"{'=' * 80}")
            print(df_train.to_string(index=False))

            csv_path = output_dir / f'training_benchmark_{timestamp}.csv'
            df_train.to_csv(csv_path, index=False)
            print(f"\nSaved to: {csv_path}")

        if all_results['generation']:
            df_gen = pd.DataFrame(all_results['generation'])
            print(f"\n{'=' * 80}")
            print("GENERATION BENCHMARK SUMMARY")
            print(f"{'=' * 80}")
            print(df_gen.to_string(index=False))

            csv_path = output_dir / f'generation_benchmark_{timestamp}.csv'
            df_gen.to_csv(csv_path, index=False)
            print(f"\nSaved to: {csv_path}")

        json_path = output_dir / f'benchmark_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'args': vars(args),
                'device': torch.cuda.get_device_name(local_rank),
                'results': all_results
            }, f, indent=2)
        print(f"Full results saved to: {json_path}\n")

    if args.multi_gpu:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
