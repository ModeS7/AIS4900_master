"""Distributed training utilities.

Provides setup functions for DDP (Distributed Data Parallel) training
with automatic SLURM cluster detection.
"""
import datetime
import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Setup distributed training with dynamic port allocation.

    Automatically detects SLURM environment or falls back to environment
    variables for local multi-GPU training.

    Returns:
        Tuple of (rank, local_rank, world_size, device).
    """
    if 'SLURM_PROCID' in os.environ:
        # SLURM cluster environment
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])

        if 'SLURM_NTASKS' in os.environ:
            world_size = int(os.environ['SLURM_NTASKS'])
        else:
            nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
            # SLURM_NTASKS_PER_NODE can be comma-separated (e.g., "4,4")
            ntasks_str = os.environ.get('SLURM_NTASKS_PER_NODE', '1')
            tasks_per_node = int(ntasks_str.split(',')[0])
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
                logger.info(f"Using dynamic port: {port} (from SLURM_JOB_ID: {job_id})")
        else:
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    else:
        # Local multi-GPU or single-GPU environment
        rank = int(os.environ.get('RANK', 0))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),  # Prevent infinite hangs
    )

    return rank, local_rank, world_size, device
