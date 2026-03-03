"""nnU-Net trainer with TensorBoard logging and bf16 mixed precision.

Subclass of nnUNetTrainer that:
    - Adds TensorBoard logging for all training/validation metrics
    - Switches mixed precision from fp16 to bf16 (better numerical stability,
      no GradScaler needed since bf16 has same exponent range as fp32)

Uses absolute imports only (no relative imports) because this file gets
symlinked into the nnunetv2 package for trainer discovery.
"""
import os
from datetime import datetime, timedelta
from time import time

import numpy as np
import torch
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerTensorBoard(nnUNetTrainer):
    """nnUNetTrainer with TensorBoard logging and bf16 mixed precision.

    Changes from base nnUNetTrainer:
        - Mixed precision: bf16 instead of fp16 (no GradScaler needed)
        - TensorBoard: logs train/val loss, Dice, LR per epoch

    Usage:
        # Via train_nnunet.py (registers this trainer automatically)
        python -m medgen.scripts.train_nnunet --dataset-id 501 --fold 0

        # Via nnU-Net CLI (after manual registration)
        nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerTensorBoard
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=None):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self._tb_writer = None

    def on_train_start(self):
        super().on_train_start()

        # Switch from fp16 to bf16: disable GradScaler (not needed for bf16)
        self.grad_scaler = None
        self.print_to_log_file("Mixed precision: bf16 (GradScaler disabled)")

        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(self.output_folder, 'tensorboard')
        self._tb_writer = SummaryWriter(log_dir=tb_dir)
        self.print_to_log_file(f"TensorBoard logging to: {tb_dir}")

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            output = self.network(data)
            loss = self.loss(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        return {'loss': loss.detach().cpu().numpy()}

    def validation_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            output = self.network(data)
            del data
            val_loss = self.loss(output, target)

        # Only need full-resolution output (if deep supervision enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(
                output.shape, device=output.device, dtype=torch.float32,
            )
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            predicted_segmentation_onehot, target, axes=axes, mask=mask,
        )

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # Remove background class
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': val_loss.detach().cpu().numpy(),
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }

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
        # Full override of nnUNetTrainer.on_epoch_end() to control output.
        # Replicates all checkpointing logic, replaces verbose prints with
        # a compact progress line.
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        log = self.logger.my_fantastic_logging
        epoch = self.current_epoch
        pct = 100.0 * epoch / self.num_epochs
        now = datetime.now().strftime('%H:%M:%S')

        train_loss = log['train_losses'][-1] if log.get('train_losses') else float('nan')
        val_dice = log['mean_fg_dice'][-1] if log.get('mean_fg_dice') else float('nan')
        ema_dice = log['ema_fg_dice'][-1] if log.get('ema_fg_dice') else float('nan')

        # Epoch time and ETA
        t_start = log.get('epoch_start_timestamps', [None])[-1]
        t_end = log.get('epoch_end_timestamps', [None])[-1]
        if t_start is not None and t_end is not None:
            epoch_time = t_end - t_start
            remaining = (self.num_epochs - epoch) * epoch_time
            eta = (datetime.now() + timedelta(seconds=remaining)).strftime('%H:%M')
            h, m = int(remaining // 3600), int((remaining % 3600) // 60)
            time_str = f" | Time: {epoch_time:.1f}s | ETA: {h}h {m}m"
        else:
            epoch_time = None
            time_str = ""
            eta = ""

        self.print_to_log_file(
            f"[{now}] Epoch {epoch}/{self.num_epochs} ({pct:5.1f}%) | "
            f"Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | "
            f"EMA Dice: {ema_dice:.4f}{time_str}",
            add_timestamp=False,
        )
        if eta:
            self.print_to_log_file(f"  ({eta})", add_timestamp=False)

        # Periodic checkpoint
        if (epoch + 1) % self.save_every == 0 and epoch != (self.num_epochs - 1):
            self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_latest.pth'))

        # Best checkpoint (EMA Dice)
        if self._best_ema is None or ema_dice > self._best_ema:
            self._best_ema = ema_dice
            self.print_to_log_file(
                f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}",
            )
            self.save_checkpoint(os.path.join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

        if self._tb_writer is not None:
            self._tb_writer.flush()

    def on_train_end(self):
        super().on_train_end()
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
