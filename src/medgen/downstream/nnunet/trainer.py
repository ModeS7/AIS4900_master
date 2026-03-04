"""nnU-Net trainer with TensorBoard logging.

Subclass of nnUNetTrainer that adds TensorBoard logging for all
training/validation metrics. Uses the parent's fp16 mixed precision
and GradScaler unchanged.

Uses absolute imports only (no relative imports) because this file gets
symlinked into the nnunetv2 package for trainer discovery.
"""
import os

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerTensorBoard(nnUNetTrainer):
    """nnUNetTrainer with TensorBoard logging.

    Changes from base nnUNetTrainer:
        - TensorBoard: logs train/val loss, Dice, LR per epoch

    Training and validation steps are inherited unchanged from the parent
    (fp16 mixed precision with GradScaler).

    Usage:
        # Via train_nnunet.py (registers this trainer automatically)
        python -m medgen.scripts.train_nnunet --dataset-id 501 --fold 0

        # Via nnU-Net CLI (after manual registration)
        nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerTensorBoard
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._tb_writer = None

    def on_train_start(self):
        super().on_train_start()

        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(self.output_folder, 'tensorboard')
        self._tb_writer = SummaryWriter(log_dir=tb_dir)
        self.print_to_log_file(f"TensorBoard logging to: {tb_dir}")

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if self._tb_writer is None:
            return

        epoch = self.current_epoch
        log = self.logger.my_fantastic_logging

        # Training loss (logged by parent as last entry in train_losses)
        if log.get('train_losses'):
            self._tb_writer.add_scalar(
                'train/loss', log['train_losses'][-1], epoch,
            )

        # Learning rate
        if log.get('lrs'):
            self._tb_writer.add_scalar(
                'train/learning_rate', log['lrs'][-1], epoch,
            )

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        if self._tb_writer is None:
            return

        epoch = self.current_epoch
        log = self.logger.my_fantastic_logging

        # Validation loss
        if log.get('val_losses'):
            self._tb_writer.add_scalar(
                'val/loss', log['val_losses'][-1], epoch,
            )

        # Mean foreground Dice
        if log.get('mean_fg_dice'):
            self._tb_writer.add_scalar(
                'val/mean_fg_dice', log['mean_fg_dice'][-1], epoch,
            )

        # EMA foreground Dice (used for best checkpoint selection)
        if log.get('ema_fg_dice'):
            self._tb_writer.add_scalar(
                'val/ema_fg_dice', log['ema_fg_dice'][-1], epoch,
            )

        # Per-class Dice (we have 1 foreground class: tumor)
        if log.get('dice_per_class_or_region'):
            dice_per_class = log['dice_per_class_or_region'][-1]
            if hasattr(dice_per_class, '__iter__'):
                for i, dice_val in enumerate(dice_per_class):
                    self._tb_writer.add_scalar(
                        f'val/dice_class_{i}', dice_val, epoch,
                    )

    def on_epoch_end(self):
        super().on_epoch_end()
        if self._tb_writer is not None:
            self._tb_writer.flush()

    def on_train_end(self):
        super().on_train_end()
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
