"""
Codebook utilization tracking for VQ-VAE models.

Monitors codebook health by tracking:
- Perplexity: Effective number of codes used (exp of entropy)
- Utilization: Percentage of codes used at least once
- Usage histogram: Distribution of code selections

Reference: https://arxiv.org/abs/1711.00937 (VQ-VAE paper)
"""
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class CodebookTracker:
    """Track VQ-VAE codebook utilization metrics.

    Accumulates codebook indices over batches and computes:
    - Perplexity: exp(-sum(p * log(p))) - effective number of codes
    - Utilization: fraction of codes used at least once
    - Dead codes: number of codes never selected

    Args:
        num_embeddings: Total number of codes in codebook.
        device: Device for tensor operations.

    Example:
        >>> tracker = CodebookTracker(512, device)
        >>> for batch in dataloader:
        ...     indices = model.index_quantize(batch)
        ...     tracker.update(indices)
        >>> metrics = tracker.compute()
        >>> tracker.log_to_tensorboard(writer, epoch)
        >>> tracker.reset()
    """

    def __init__(self, num_embeddings: int, device: torch.device) -> None:
        """Initialize codebook tracker.

        Args:
            num_embeddings: Total number of codes in codebook.
            device: Device for tensor operations.
        """
        self.num_embeddings = num_embeddings
        self.device = device

        # Accumulate usage counts across batches
        self._usage_counts = torch.zeros(num_embeddings, dtype=torch.long, device=device)
        self._total_tokens = 0

    def update(self, indices: torch.Tensor) -> None:
        """Update usage counts with batch of indices.

        Args:
            indices: Codebook indices tensor of any shape.
                Values should be in [0, num_embeddings).
        """
        # Flatten indices and count occurrences
        flat_indices = indices.flatten()
        self._total_tokens += flat_indices.numel()

        # Count each code's usage
        for idx in range(self.num_embeddings):
            self._usage_counts[idx] += (flat_indices == idx).sum()

    def update_fast(self, indices: torch.Tensor) -> None:
        """Update usage counts efficiently using bincount.

        Faster than update() for large batches.

        Args:
            indices: Codebook indices tensor of any shape.
        """
        flat_indices = indices.flatten().long()
        self._total_tokens += flat_indices.numel()

        # Use bincount for efficient counting
        counts = torch.bincount(flat_indices, minlength=self.num_embeddings)
        self._usage_counts += counts

    def compute(self) -> dict[str, float]:
        """Compute codebook metrics from accumulated counts.

        Returns:
            Dict with:
                - perplexity: Effective number of codes used
                - utilization: Fraction of codes used (0-1)
                - dead_codes: Number of codes never used
                - entropy: Shannon entropy of code distribution
        """
        if self._total_tokens == 0:
            return {
                'perplexity': 0.0,
                'utilization': 0.0,
                'dead_codes': self.num_embeddings,
                'entropy': 0.0,
            }

        # Compute probabilities
        probs = self._usage_counts.float() / self._total_tokens

        # Compute entropy (only for non-zero probs to avoid log(0))
        nonzero_mask = probs > 0
        entropy = -torch.sum(probs[nonzero_mask] * torch.log(probs[nonzero_mask] + 1e-10))

        # Perplexity = exp(entropy)
        perplexity = torch.exp(entropy)

        # Utilization = fraction of codes used at least once
        codes_used = (self._usage_counts > 0).sum()
        utilization = codes_used.float() / self.num_embeddings

        # Dead codes = codes never used
        dead_codes = self.num_embeddings - codes_used.item()

        return {
            'perplexity': perplexity.item(),
            'utilization': utilization.item(),
            'dead_codes': dead_codes,
            'entropy': entropy.item(),
        }

    def get_usage_histogram(self) -> torch.Tensor:
        """Get normalized usage histogram.

        Returns:
            Tensor of shape (num_embeddings,) with usage probabilities.
        """
        if self._total_tokens == 0:
            return torch.zeros(self.num_embeddings, device=self.device)
        return self._usage_counts.float() / self._total_tokens

    def reset(self) -> None:
        """Reset accumulated counts for new epoch."""
        self._usage_counts.zero_()
        self._total_tokens = 0

    def log_to_tensorboard(
        self,
        writer: SummaryWriter,
        epoch: int,
        prefix: str = 'Codebook',
    ) -> dict[str, float]:
        """Log codebook metrics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter.
            epoch: Current epoch number.
            prefix: Prefix for metric names.

        Returns:
            Computed metrics dict.
        """
        metrics = self.compute()

        writer.add_scalar(f'{prefix}/perplexity', metrics['perplexity'], epoch)
        writer.add_scalar(f'{prefix}/utilization', metrics['utilization'], epoch)
        writer.add_scalar(f'{prefix}/dead_codes', metrics['dead_codes'], epoch)
        writer.add_scalar(f'{prefix}/entropy', metrics['entropy'], epoch)

        # Log perplexity as percentage of codebook size for easier interpretation
        perplexity_pct = 100.0 * metrics['perplexity'] / self.num_embeddings
        writer.add_scalar(f'{prefix}/perplexity_pct', perplexity_pct, epoch)

        return metrics

    def log_summary(self) -> None:
        """Log a summary of codebook metrics to console."""
        metrics = self.compute()
        perplexity_pct = 100.0 * metrics['perplexity'] / self.num_embeddings

        logger.info(
            f"Codebook: perplexity={metrics['perplexity']:.1f}/{self.num_embeddings} "
            f"({perplexity_pct:.1f}%), "
            f"utilization={metrics['utilization']*100:.1f}%, "
            f"dead={metrics['dead_codes']}"
        )
