"""
MC Dropout uncertainty estimation for segmentation models.

Enables dropout at inference time and runs multiple forward passes
to estimate prediction uncertainty via variance of the outputs.
"""
import logging

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class MCDropoutEvaluator:
    """MC Dropout uncertainty estimation for segmentation models.

    Enables dropout layers during inference while keeping BatchNorm in eval mode.
    Runs N stochastic forward passes and computes mean prediction + uncertainty.

    Usage:
        evaluator = MCDropoutEvaluator(model, n_samples=10)
        mean_pred, uncertainty = evaluator.predict_with_uncertainty(images)
        conf_metrics = evaluator.compute_confidence_metrics(mean_pred, uncertainty, target)
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def predict_with_uncertainty(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Run N forward passes with dropout enabled.

        Saves and restores RNG state to avoid affecting training reproducibility.

        Args:
            images: Input images [B, C, ...].

        Returns:
            Tuple of (mean_prediction, uncertainty_map), both [B, 1, ...].
            mean_prediction is the average sigmoid output.
            uncertainty is the pixel-wise variance across passes.
        """
        # Save RNG state
        cpu_state = torch.random.get_rng_state()
        gpu_state = torch.cuda.get_rng_state(self.device) if torch.cuda.is_available() else None

        # Enable dropout only (keep BN in eval)
        dropout_modules = []
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                if not module.training:
                    module.train()
                    dropout_modules.append(module)

        # Collect sigmoid outputs from N passes
        outputs = []
        for _ in range(self.n_samples):
            logits = self.model(images)
            outputs.append(torch.sigmoid(logits))

        # Restore dropout to eval mode
        for module in dropout_modules:
            module.eval()

        # Restore RNG state
        torch.random.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state(gpu_state, self.device)

        # Stack and compute statistics: [N, B, 1, ...]
        stacked = torch.stack(outputs, dim=0)
        mean_pred = stacked.mean(dim=0)      # [B, 1, ...]
        uncertainty = stacked.var(dim=0)      # [B, 1, ...]

        return mean_pred, uncertainty

    def compute_confidence_metrics(
        self,
        mean_pred: Tensor,
        uncertainty: Tensor,
        target: Tensor,
    ) -> dict[str, float]:
        """Compute confidence metrics on TP/FP/FN regions + ECE.

        Args:
            mean_pred: Mean sigmoid prediction [B, 1, ...].
            uncertainty: Variance map [B, 1, ...].
            target: Binary ground truth [B, 1, ...].

        Returns:
            Dict with 'confidence_tp', 'confidence_fp', 'confidence_fn', 'ece'.
        """
        confidence = 1.0 - uncertainty
        pred_binary = (mean_pred > 0.5).bool()
        target_binary = target.bool()

        tp_mask = pred_binary & target_binary
        fp_mask = pred_binary & ~target_binary
        fn_mask = ~pred_binary & target_binary

        results: dict[str, float] = {}

        # Mean confidence on each region
        if tp_mask.sum() > 0:
            results['confidence_tp'] = confidence[tp_mask].mean().item()
        else:
            results['confidence_tp'] = 0.0

        if fp_mask.sum() > 0:
            results['confidence_fp'] = confidence[fp_mask].mean().item()
        else:
            results['confidence_fp'] = 0.0

        if fn_mask.sum() > 0:
            results['confidence_fn'] = confidence[fn_mask].mean().item()
        else:
            results['confidence_fn'] = 0.0

        # Expected Calibration Error (10 bins)
        results['ece'] = self._compute_ece(mean_pred, target, n_bins=10)

        return results

    @staticmethod
    def _compute_ece(pred_prob: Tensor, target: Tensor, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error.

        Args:
            pred_prob: Predicted probabilities [B, 1, ...].
            target: Binary ground truth [B, 1, ...].
            n_bins: Number of calibration bins.

        Returns:
            ECE value (lower is better, range [0, 1]).
        """
        pred_flat = pred_prob.view(-1)
        tgt_flat = target.float().view(-1)
        n_total = pred_flat.numel()

        if n_total == 0:
            return 0.0

        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=pred_flat.device)

        for i in range(n_bins):
            mask = (pred_flat > bin_boundaries[i]) & (pred_flat <= bin_boundaries[i + 1])
            n_in_bin = mask.sum().item()

            if n_in_bin == 0:
                continue

            avg_confidence = pred_flat[mask].mean().item()
            avg_accuracy = tgt_flat[mask].mean().item()
            ece += (n_in_bin / n_total) * abs(avg_accuracy - avg_confidence)

        return ece
