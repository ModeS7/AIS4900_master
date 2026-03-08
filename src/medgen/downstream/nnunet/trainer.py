"""nnU-Net trainers for brain metastasis segmentation.

Two trainer classes:
    - nnUNetTrainerTensorBoard: Default DC+CE loss with TensorBoard + 100% oversampling
    - nnUNetTrainerBrainMets: Dice+TopK10 loss optimized for tiny lesions + TensorBoard

Brain metastases occupy 0.001-0.06% of volume. Default nnU-Net (DC+CE with smooth=1e-5,
SGD at LR=0.01) collapses to all-background because the Dice gradient is near-zero
for such tiny targets. The BrainMets trainer fixes this with:
    - TopK10 loss: only backpropagates from the hardest 10% of voxels
    - smooth=0: sharper Dice gradients (no smoothing to hide behind)
    - batch_dice=False: per-sample Dice (forced, ignores plans)
    - oversample_foreground_percent=1.0: every patch contains tumor

Uses absolute imports only (no relative imports) because this file gets
symlinked into the nnunetv2 package for trainer discovery.
"""
import os

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# =============================================================================
# TensorBoard mixin (shared by all trainers)
# =============================================================================

class _TensorBoardMixin:
    """Mixin that adds TensorBoard logging to any nnUNetTrainer subclass."""

    def _init_tb(self):
        self._tb_writer = None

    def _start_tb(self):
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(self.output_folder, 'tensorboard')
        self._tb_writer = SummaryWriter(log_dir=tb_dir)
        self.print_to_log_file(f"TensorBoard logging to: {tb_dir}")

    def _log_train(self):
        if self._tb_writer is None:
            return
        epoch = self.current_epoch
        log = self.logger.my_fantastic_logging
        if log.get('train_losses'):
            self._tb_writer.add_scalar('train/loss', log['train_losses'][-1], epoch)
        if log.get('lrs'):
            self._tb_writer.add_scalar('train/learning_rate', log['lrs'][-1], epoch)

    def _log_val(self):
        if self._tb_writer is None:
            return
        epoch = self.current_epoch
        log = self.logger.my_fantastic_logging
        if log.get('val_losses'):
            self._tb_writer.add_scalar('val/loss', log['val_losses'][-1], epoch)
        if log.get('mean_fg_dice'):
            self._tb_writer.add_scalar('val/mean_fg_dice', log['mean_fg_dice'][-1], epoch)
        if log.get('ema_fg_dice'):
            self._tb_writer.add_scalar('val/ema_fg_dice', log['ema_fg_dice'][-1], epoch)
        if log.get('dice_per_class_or_region'):
            dice_per_class = log['dice_per_class_or_region'][-1]
            if hasattr(dice_per_class, '__iter__'):
                for i, dice_val in enumerate(dice_per_class):
                    self._tb_writer.add_scalar(f'val/dice_class_{i}', dice_val, epoch)

    def _flush_tb(self):
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def _close_tb(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None


# =============================================================================
# Default trainer: DC+CE loss + TensorBoard + 100% oversampling
# =============================================================================

class nnUNetTrainerTensorBoard(_TensorBoardMixin, nnUNetTrainer):
    """nnUNetTrainer with TensorBoard logging and 100% foreground oversampling.

    Uses the default DC+CE loss. Good baseline for comparison.

    Usage:
        python -m medgen.scripts.train_nnunet --trainer nnUNetTrainerTensorBoard
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 1.0
        self._init_tb()

    def on_train_start(self):
        super().on_train_start()
        self._start_tb()

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self._log_train()

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        self._log_val()

    def on_epoch_end(self):
        super().on_epoch_end()
        self._flush_tb()

    def on_train_end(self):
        super().on_train_end()
        self._close_tb()


# =============================================================================
# Brain metastasis trainer: Dice+TopK10 + TensorBoard + 100% oversampling
# =============================================================================

class nnUNetTrainerBrainMets(_TensorBoardMixin, nnUNetTrainer):
    """nnUNetTrainer optimized for extremely small lesions (brain metastases).

    Changes from default nnUNetTrainer:
        - Loss: Dice + TopK10 (focuses on hardest 10% of voxels)
        - smooth=0: no Dice smoothing — sharper gradients
        - batch_dice=False: per-sample Dice (forced, ignores plans)
        - oversample_foreground_percent=1.0: every patch contains tumor
        - TensorBoard logging

    Based on nnU-Net's built-in nnUNetTrainerDiceTopK10Loss but with
    additional oversampling and TensorBoard.

    Usage:
        python -m medgen.scripts.train_nnunet --trainer nnUNetTrainerBrainMets
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.oversample_foreground_percent = 1.0
        self._init_tb()

    def _build_loss(self):
        from nnunetv2.training.loss.compound_losses import DC_and_topk_loss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

        loss = DC_and_topk_loss(
            {
                'batch_dice': False,
                'smooth': 0,
                'do_bg': False,
                'ddp': self.is_ddp,
            },
            {'k': 10, 'label_smoothing': 0.0},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_dummy_2d_data_aug:
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def on_train_start(self):
        super().on_train_start()
        self._start_tb()
        self.print_to_log_file(
            "BrainMets trainer: Dice+TopK10, smooth=0, batch_dice=False, "
            "oversample=1.0"
        )

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        self._log_train()

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        self._log_val()

    def on_epoch_end(self):
        super().on_epoch_end()
        self._flush_tb()

    def on_train_end(self):
        super().on_train_end()
        self._close_tb()
