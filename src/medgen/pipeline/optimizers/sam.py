"""
Sharpness-Aware Minimization (SAM) optimizer wrapper.

SAM seeks parameters that lie in neighborhoods having uniformly low loss,
resulting in better generalization. It requires two forward-backward passes
per optimization step.

Reference: https://arxiv.org/abs/2010.01412
"""
import torch
from torch.optim import Optimizer
from typing import Callable, Optional


class SAM(Optimizer):
    """Sharpness-Aware Minimization optimizer wrapper.

    Wraps any base optimizer (e.g., AdamW) to perform SAM optimization.
    Requires two forward-backward passes per step:
    1. first_step(): Compute gradient and perturb weights to find sharp region
    2. second_step(): Compute gradient at perturbed point and update

    Args:
        params: Model parameters to optimize.
        base_optimizer: Base optimizer class (e.g., torch.optim.AdamW).
        rho: Perturbation radius for SAM (default: 0.05).
        adaptive: Use adaptive SAM (ASAM) which normalizes by weight magnitude.
        **kwargs: Arguments passed to base optimizer (lr, weight_decay, etc.).

    Example:
        >>> optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=1e-4, rho=0.05)
        >>>
        >>> # Training step (two forward-backward passes)
        >>> loss = compute_loss(model, batch)
        >>> loss.backward()
        >>> optimizer.first_step(zero_grad=True)
        >>>
        >>> loss = compute_loss(model, batch)  # Second forward
        >>> loss.backward()
        >>> optimizer.second_step(zero_grad=True)
    """

    def __init__(
        self,
        params,
        base_optimizer: type,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs
    ):
        assert rho >= 0.0, f"Invalid rho: {rho}, should be >= 0"

        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        # Track step count for LR scheduler compatibility
        self._step_count = 0

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """First step: perturb weights toward sharp region.

        Computes the perturbation direction from gradients and moves
        weights to maximize loss locally (find sharp region).

        Args:
            zero_grad: Whether to zero gradients after perturbation.
        """
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Store original weights (initialize state dict if needed)
                if p not in self.state:
                    self.state[p] = {}
                self.state[p]["old_p"] = p.data.clone()

                # Compute perturbation (adaptive uses weight magnitude normalization)
                if group["adaptive"]:
                    # ASAM: normalize by |p| (element-wise absolute value)
                    e_w = (torch.abs(p) * p.grad * scale).to(p)
                else:
                    e_w = (p.grad * scale).to(p)

                # Perturb weights: move in gradient direction (toward sharp region)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Second step: restore weights and apply SAM update.

        Restores original weights and updates using gradient computed
        at the perturbed point (sharpness-aware gradient).

        Args:
            zero_grad: Whether to zero gradients after update.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Restore original weights before update
                p.data = self.state[p]["old_p"]

        # Update with base optimizer using gradient from perturbed point
        self.base_optimizer.step()
        self._step_count += 1  # Track for LR scheduler compatibility

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Standard optimizer step (for compatibility).

        Note: For SAM, you should use first_step() and second_step() separately.
        This method is provided for interface compatibility but raises an error
        to prevent accidental misuse.
        """
        raise NotImplementedError(
            "SAM requires two steps: first_step() and second_step(). "
            "Use the closure-based step_with_closure() if you need single-call API."
        )

    def step_with_closure(self, closure: Callable) -> torch.Tensor:
        """Single-call SAM step using a closure.

        Convenience method that handles both forward-backward passes internally.

        Args:
            closure: A closure that computes the loss and calls backward().
                     Will be called twice (for both SAM steps).

        Returns:
            Loss value from the second forward pass.

        Example:
            >>> def closure():
            ...     optimizer.zero_grad()
            ...     loss = compute_loss(model, batch)
            ...     loss.backward()
            ...     return loss
            >>> loss = optimizer.step_with_closure(closure)
        """
        # First forward-backward
        closure()
        self.first_step(zero_grad=True)

        # Second forward-backward
        loss = closure()
        self.second_step(zero_grad=True)

        return loss

    def _grad_norm(self) -> torch.Tensor:
        """Compute the L2 norm of all gradients."""
        shared_device = self.param_groups[0]["params"][0].device

        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )

        return norm

    def load_state_dict(self, state_dict: dict) -> None:
        """Load optimizer state including SAM-specific state.

        Handles both new format (with sam_state) and legacy format
        (base optimizer only) for backwards compatibility.
        """
        if isinstance(state_dict, dict) and 'base_optimizer' in state_dict:
            # New format with SAM state
            self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
            self.param_groups = self.base_optimizer.param_groups

            # Restore SAM state (old_p tensors for weight restoration)
            sam_state = state_dict.get('sam_state', {})
            for group_idx, group in enumerate(self.param_groups):
                for param_idx, p in enumerate(group["params"]):
                    key = f"{group_idx}_{param_idx}"
                    if key in sam_state:
                        if p not in self.state:
                            self.state[p] = {}
                        self.state[p]["old_p"] = sam_state[key]
        else:
            # Backwards compatibility: old format without SAM state
            super().load_state_dict(state_dict)
            self.base_optimizer.param_groups = self.param_groups

    def state_dict(self) -> dict:
        """Return optimizer state including SAM-specific state.

        SAM state includes old_p tensors used for weight restoration during
        second_step(). We serialize by parameter index (group_idx_param_idx)
        rather than id(p) since object IDs change between sessions.
        """
        # Create mapping of param index -> SAM state
        sam_state = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, p in enumerate(group["params"]):
                if p in self.state and "old_p" in self.state[p]:
                    key = f"{group_idx}_{param_idx}"
                    sam_state[key] = self.state[p]["old_p"]

        return {
            'base_optimizer': self.base_optimizer.state_dict(),
            'sam_state': sam_state,
        }
