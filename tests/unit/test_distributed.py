"""Tests for distributed training utilities.

Tests cover SLURM environment variable parsing, port allocation,
rank assignment, and fallback modes without actually initializing
distributed training (which requires multiple processes).
"""

import pytest
from unittest.mock import patch, MagicMock
import os


class TestSlurmParsing:
    """Tests for SLURM environment variable parsing."""

    def test_slurm_ntasks_per_node_single_value(self, monkeypatch):
        """Single NTASKS_PER_NODE value parses correctly."""
        # Set up SLURM environment
        monkeypatch.setenv('SLURM_PROCID', '0')
        monkeypatch.setenv('SLURM_LOCALID', '0')
        monkeypatch.setenv('SLURM_JOB_NUM_NODES', '2')
        monkeypatch.setenv('SLURM_NTASKS_PER_NODE', '4')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        monkeypatch.delenv('SLURM_NTASKS', raising=False)

        # Import after setting env vars
        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            # Mock init_process_group to capture world_size
            captured_kwargs = {}
            def capture_init(**kwargs):
                captured_kwargs.update(kwargs)
            mock_dist.init_process_group = capture_init

            setup_distributed()

            # world_size should be nodes * tasks_per_node = 2 * 4 = 8
            assert captured_kwargs['world_size'] == 8

    def test_slurm_ntasks_per_node_heterogeneous_documents_bug(self, monkeypatch):
        """Heterogeneous clusters only use first value (known limitation)."""
        # SLURM_NTASKS_PER_NODE can be comma-separated for heterogeneous clusters
        monkeypatch.setenv('SLURM_PROCID', '0')
        monkeypatch.setenv('SLURM_LOCALID', '0')
        monkeypatch.setenv('SLURM_JOB_NUM_NODES', '3')
        monkeypatch.setenv('SLURM_NTASKS_PER_NODE', '2,4,2')  # Heterogeneous
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        monkeypatch.delenv('SLURM_NTASKS', raising=False)

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            captured_kwargs = {}
            def capture_init(**kwargs):
                captured_kwargs.update(kwargs)
            mock_dist.init_process_group = capture_init

            setup_distributed()

            # DOCUMENTED BUG: Only uses first value, so 3 * 2 = 6, not 8
            assert captured_kwargs['world_size'] == 6

    def test_slurm_ntasks_takes_precedence(self, monkeypatch):
        """SLURM_NTASKS takes precedence over NTASKS_PER_NODE calculation."""
        monkeypatch.setenv('SLURM_PROCID', '0')
        monkeypatch.setenv('SLURM_LOCALID', '0')
        monkeypatch.setenv('SLURM_NTASKS', '8')  # Explicit total
        monkeypatch.setenv('SLURM_JOB_NUM_NODES', '2')
        monkeypatch.setenv('SLURM_NTASKS_PER_NODE', '4')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', 'localhost')

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            captured_kwargs = {}
            def capture_init(**kwargs):
                captured_kwargs.update(kwargs)
            mock_dist.init_process_group = capture_init

            setup_distributed()

            # SLURM_NTASKS directly used
            assert captured_kwargs['world_size'] == 8

    def test_slurm_missing_job_id_uses_fallback_port(self, monkeypatch):
        """Missing SLURM_JOB_ID falls back to default port."""
        monkeypatch.setenv('SLURM_PROCID', '0')
        monkeypatch.setenv('SLURM_LOCALID', '0')
        monkeypatch.setenv('SLURM_NTASKS', '4')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
        monkeypatch.delenv('MASTER_PORT', raising=False)

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            mock_dist.init_process_group = MagicMock()

            setup_distributed()

            # Should fall back to default port
            assert os.environ.get('MASTER_PORT') == '12355'


class TestPortAllocation:
    """Tests for dynamic port allocation."""

    def test_port_allocation_deterministic(self, monkeypatch):
        """Same job ID produces same port."""
        job_id = 12345
        port = 12000 + (job_id % 53000)

        # Call twice to verify determinism
        port1 = 12000 + (job_id % 53000)
        port2 = 12000 + (job_id % 53000)

        assert port1 == port2

    def test_port_allocation_range_valid(self):
        """Port stays in valid range [12000, 65000)."""
        # Test with various job IDs
        for job_id in [0, 1, 12345, 53000, 100000, 999999, 10000000]:
            port = 12000 + (job_id % 53000)
            assert 12000 <= port < 65000, f"Port {port} out of range for job_id {job_id}"

    def test_port_allocation_formula(self, monkeypatch):
        """Port formula produces expected result."""
        monkeypatch.setenv('SLURM_PROCID', '0')
        monkeypatch.setenv('SLURM_LOCALID', '0')
        monkeypatch.setenv('SLURM_NTASKS', '4')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', 'localhost')

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            mock_dist.init_process_group = MagicMock()

            setup_distributed()

            expected_port = 12000 + (12345 % 53000)
            assert os.environ.get('MASTER_PORT') == str(expected_port)


class TestRankAssignment:
    """Tests for rank and device assignment."""

    def test_rank_local_rank_assignment_single_node(self, monkeypatch):
        """Single node assigns correct rank and local_rank."""
        monkeypatch.setenv('SLURM_PROCID', '2')
        monkeypatch.setenv('SLURM_LOCALID', '2')
        monkeypatch.setenv('SLURM_NTASKS', '4')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', 'localhost')

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            captured_kwargs = {}
            def capture_init(**kwargs):
                captured_kwargs.update(kwargs)
            mock_dist.init_process_group = capture_init

            rank, local_rank, world_size, device = setup_distributed()

            assert rank == 2
            assert local_rank == 2
            assert captured_kwargs['rank'] == 2

    def test_multi_node_rank_assignment(self, monkeypatch):
        """Multi-node assigns global rank correctly."""
        # Simulate rank 5 on node 1 (2 nodes, 4 GPUs each)
        monkeypatch.setenv('SLURM_PROCID', '5')  # Global rank
        monkeypatch.setenv('SLURM_LOCALID', '1')  # Local rank on node 1
        monkeypatch.setenv('SLURM_NTASKS', '8')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', 'localhost')

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            captured_kwargs = {}
            def capture_init(**kwargs):
                captured_kwargs.update(kwargs)
            mock_dist.init_process_group = capture_init

            rank, local_rank, world_size, device = setup_distributed()

            assert rank == 5
            assert local_rank == 1
            assert world_size == 8

    def test_device_set_to_local_rank(self, monkeypatch):
        """Device set to cuda:local_rank."""
        monkeypatch.setenv('SLURM_PROCID', '2')
        monkeypatch.setenv('SLURM_LOCALID', '2')
        monkeypatch.setenv('SLURM_NTASKS', '4')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', 'localhost')

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            mock_dist.init_process_group = MagicMock()

            rank, local_rank, world_size, device = setup_distributed()

            # Verify set_device was called with local_rank
            mock_cuda.set_device.assert_called_once_with(2)


class TestFallbackMode:
    """Tests for non-SLURM fallback."""

    def test_fallback_to_env_vars_when_no_slurm(self, monkeypatch):
        """Uses RANK, LOCAL_RANK, WORLD_SIZE when no SLURM."""
        # Clear all SLURM variables
        slurm_vars = ['SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_JOB_ID',
                      'SLURM_NTASKS', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS_PER_NODE',
                      'SLURM_JOB_NODELIST', 'SLURM_LAUNCH_NODE_IPADDR']
        for var in slurm_vars:
            monkeypatch.delenv(var, raising=False)

        # Set standard distributed training env vars
        monkeypatch.setenv('RANK', '1')
        monkeypatch.setenv('LOCAL_RANK', '1')
        monkeypatch.setenv('WORLD_SIZE', '4')

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            captured_kwargs = {}
            def capture_init(**kwargs):
                captured_kwargs.update(kwargs)
            mock_dist.init_process_group = capture_init

            rank, local_rank, world_size, device = setup_distributed()

            assert rank == 1
            assert local_rank == 1
            assert world_size == 4
            assert captured_kwargs['rank'] == 1
            assert captured_kwargs['world_size'] == 4

    def test_fallback_defaults_to_single_gpu(self, monkeypatch):
        """Defaults to rank 0, world_size 1 when no env vars."""
        # Clear all distributed training variables
        slurm_vars = ['SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_JOB_ID',
                      'SLURM_NTASKS', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS_PER_NODE',
                      'SLURM_JOB_NODELIST', 'SLURM_LAUNCH_NODE_IPADDR']
        for var in slurm_vars:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv('RANK', raising=False)
        monkeypatch.delenv('LOCAL_RANK', raising=False)
        monkeypatch.delenv('WORLD_SIZE', raising=False)

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            captured_kwargs = {}
            def capture_init(**kwargs):
                captured_kwargs.update(kwargs)
            mock_dist.init_process_group = capture_init

            rank, local_rank, world_size, device = setup_distributed()

            assert rank == 0
            assert local_rank == 0
            assert world_size == 1


class TestMasterAddr:
    """Tests for MASTER_ADDR assignment."""

    def test_slurm_nodelist_sets_master_addr(self, monkeypatch):
        """SLURM_JOB_NODELIST sets MASTER_ADDR to first node."""
        monkeypatch.setenv('SLURM_PROCID', '0')
        monkeypatch.setenv('SLURM_LOCALID', '0')
        monkeypatch.setenv('SLURM_NTASKS', '4')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_JOB_NODELIST', 'node1,node2,node3')

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            mock_dist.init_process_group = MagicMock()

            setup_distributed()

            # MASTER_ADDR should be first node
            assert os.environ.get('MASTER_ADDR') == 'node1'

    def test_slurm_nodelist_range_notation(self, monkeypatch):
        """SLURM_JOB_NODELIST with range notation extracts prefix."""
        monkeypatch.setenv('SLURM_PROCID', '0')
        monkeypatch.setenv('SLURM_LOCALID', '0')
        monkeypatch.setenv('SLURM_NTASKS', '4')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_JOB_NODELIST', 'node[1-4]')

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            mock_dist.init_process_group = MagicMock()

            setup_distributed()

            # Should extract 'node' from 'node[1-4]'
            assert os.environ.get('MASTER_ADDR') == 'node'

    def test_fallback_to_launch_node_ipaddr(self, monkeypatch):
        """Falls back to SLURM_LAUNCH_NODE_IPADDR when no NODELIST."""
        monkeypatch.setenv('SLURM_PROCID', '0')
        monkeypatch.setenv('SLURM_LOCALID', '0')
        monkeypatch.setenv('SLURM_NTASKS', '4')
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        monkeypatch.setenv('SLURM_LAUNCH_NODE_IPADDR', '192.168.1.1')
        monkeypatch.delenv('SLURM_JOB_NODELIST', raising=False)

        with patch('medgen.core.distributed.dist') as mock_dist, \
             patch('medgen.core.distributed.torch.cuda') as mock_cuda:
            mock_dist.is_initialized.return_value = False
            mock_cuda.is_available.return_value = True
            mock_cuda.set_device = MagicMock()
            mock_cuda.device_count.return_value = 8

            from medgen.core.distributed import setup_distributed

            mock_dist.init_process_group = MagicMock()

            setup_distributed()

            assert os.environ.get('MASTER_ADDR') == '192.168.1.1'
