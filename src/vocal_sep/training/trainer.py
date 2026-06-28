"""Training loop with mixed precision and early stopping."""

from __future__ import annotations

import contextlib
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from vocal_sep.audio.spectrogram import denormalize
from vocal_sep.audio.wav import write_waveform
from vocal_sep.config import TrainSettings
from vocal_sep.device import log_gpu_memory, uses_cuda
from vocal_sep.evaluation.metrics import separation_sdr
from vocal_sep.log import epoch_progress, format_loss, format_lr, get_logger
from vocal_sep.models.unet import MaskUNet
from vocal_sep.paths import RUNS_DIR

logger = get_logger(__name__)


@dataclass
class RunPaths:
    root: Path
    checkpoints: Path
    audio: Path
    plots: Path


class Trainer:
    def __init__(self, settings: TrainSettings, device=None):
        self.settings = settings
        if device is None:
            from vocal_sep.device import resolve_device

            device = resolve_device(settings.device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.stft = settings.stft
        self.model = MaskUNet()

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Initialized [bold]MaskUNet[/bold] with [bold cyan]%s[/bold cyan] parameters",
            f"{param_count:,}",
        )

        if settings.compile_model and hasattr(torch, "compile") and uses_cuda(self.device):
            logger.info("Compiling model with [bold]torch.compile[/bold]")
            self.model = torch.compile(self.model)

        self.model.to(self.device)
        if settings.channels_last and uses_cuda(self.device):
            self.model.to(memory_format=torch.channels_last)
            logger.info("Using [bold]channels_last[/bold] memory format for convolutions")

        logger.info("Model on [bold magenta]%s[/bold magenta]", self.device)
        log_gpu_memory(self.device, "After model load")

        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=settings.learning_rate,
            amsgrad=settings.amsgrad,
        )
        self.scheduler = self._build_scheduler(settings)
        self.scaler = GradScaler(self.device.type) if uses_cuda(self.device) else None
        if self.scaler is not None:
            logger.info("Mixed precision enabled ([bold]AMP[/bold] + GradScaler)")
        logger.debug("Optimizer: Adam(lr=%s, amsgrad=%s)", settings.learning_rate, settings.amsgrad)
        if self.scheduler is not None:
            logger.info(
                "LR scheduler: [bold]ReduceLROnPlateau[/bold] (patience=%d, factor=%g, min_lr=%g)",
                settings.lr_patience,
                settings.lr_factor,
                settings.min_lr,
            )
        logger.info(
            "Early stopping — patience=%d, min_delta=%g on val loss",
            settings.patience,
            settings.min_delta,
        )

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val = float("inf")
        self.patience_counter = 0
        self.stopped_early = False
        self.run_paths = self._create_run_paths()

        if settings.resume:
            checkpoint = _find_latest_checkpoint(RUNS_DIR)
            if checkpoint is not None:
                self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
                logger.info("Resumed from checkpoint [bold]%s[/bold]", checkpoint)
            else:
                logger.warning("Resume requested but no checkpoint was found")

        if settings.eval_only:
            logger.warning("Running in [bold yellow]eval-only[/bold yellow] mode")

    def _build_scheduler(self, settings: TrainSettings):
        if settings.lr_scheduler != "plateau":
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=settings.lr_factor,
            patience=settings.lr_patience,
            min_lr=settings.min_lr,
        )

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _progress_fields(self, epoch: int, loss: str = "—") -> dict[str, str]:
        return {
            "epoch": str(epoch + 1),
            "loss": loss,
            "best": format_loss(self.best_val),
            "lr": format_lr(self._current_lr()),
            "patience": f"{self.patience_counter}/{self.settings.patience}",
        }

    def _step_scheduler(self, val_loss: float) -> None:
        if self.scheduler is None:
            return
        previous_lr = self._current_lr()
        self.scheduler.step(val_loss)
        new_lr = self._current_lr()
        if new_lr < previous_lr - 1e-15:
            logger.info(
                "[bold yellow]LR reduced[/bold yellow] %.2e → %.2e (val=%.4f)",
                previous_lr,
                new_lr,
                val_loss,
            )

    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        """Return True if training should stop. Saves best checkpoint when improved."""
        if val_loss < self.best_val - self.settings.min_delta:
            improvement = self.best_val - val_loss if self.best_val != float("inf") else val_loss
            self.best_val = val_loss
            self.patience_counter = 0
            if not self.settings.eval_only:
                path = self.run_paths.checkpoints / "best.pt"
                torch.save(self.model.state_dict(), path)
                logger.info(
                    "[bold green]New best[/bold green] val=%.4f (Δ=%.4f) → saved [bold]%s[/bold]",
                    val_loss,
                    improvement,
                    path.name,
                )
            return False

        self.patience_counter += 1
        logger.info(
            "No val improvement (%d/%d patience, min_delta=%g)",
            self.patience_counter,
            self.settings.patience,
            self.settings.min_delta,
        )
        if self.patience_counter >= self.settings.patience:
            self.stopped_early = True
            logger.warning(
                "[bold yellow]Early stopping[/bold yellow] at epoch %d — "
                "best val=%.4f after %d epochs without ≥%g improvement",
                epoch + 1,
                self.best_val,
                self.settings.patience,
                self.settings.min_delta,
            )
            return True
        return False

    def _restore_best_weights(self) -> None:
        if self.settings.eval_only:
            return
        best_path = self.run_paths.checkpoints / "best.pt"
        if not best_path.exists():
            return
        state = torch.load(best_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        logger.info(
            "[bold green]Restored best checkpoint[/bold green] (val=%.4f) from [bold]%s[/bold]",
            self.best_val,
            best_path.name,
        )

    def _create_run_paths(self) -> RunPaths:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = RUNS_DIR / stamp
        paths = RunPaths(
            root=root,
            checkpoints=root / "checkpoints",
            audio=root / "audio",
            plots=root / "plots",
        )
        for path in (paths.root, paths.checkpoints, paths.audio, paths.plots):
            path.mkdir(parents=True, exist_ok=True)
        logger.info("Run directory: [bold]%s[/bold]", root)
        return paths

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        logger.info(
            "Starting fit for up to [bold]%d[/bold] epochs (early stop patience=%d)",
            self.settings.epochs,
            self.settings.patience,
        )
        for epoch in range(self.settings.epochs):
            logger.info(
                "[bold magenta]Epoch %d/%d[/bold magenta]  lr=%s",
                epoch + 1,
                self.settings.epochs,
                format_lr(self._current_lr()),
            )

            if not self.settings.eval_only:
                train_loss = self._train_epoch(train_loader, epoch)
                self.train_losses.append(train_loss)

            val_loss = self._validate_epoch(val_loader, epoch)
            self.val_losses.append(val_loss)
            self._step_scheduler(val_loss)

            if self._check_early_stopping(val_loss, epoch):
                self._log_epoch(epoch, val_loss)
                break

            self._log_epoch(epoch, val_loss)

            if (
                not self.settings.eval_only
                and epoch > 0
                and epoch % self.settings.log_loss_every == 0
            ):
                plot_path = self._save_loss_plot(epoch)
                logger.info("Saved loss plot to [bold]%s[/bold]", plot_path)

        self._restore_best_weights()

    def _log_epoch(self, epoch: int, val_loss: float) -> None:
        if self.settings.eval_only:
            logger.info(
                "Epoch [bold]%d[/bold] val loss: [yellow]%.4f[/yellow]",
                epoch,
                val_loss,
            )
            return
        logger.info(
            "Epoch [bold]%d[/bold] losses — train: [green]%.4f[/green]  "
            "val: [yellow]%.4f[/yellow]  best: [bold yellow]%.4f[/bold yellow]",
            epoch,
            self.train_losses[-1],
            val_loss,
            self.best_val,
        )

    def _autocast(self):
        if uses_cuda(self.device):
            return autocast(self.device.type)
        return contextlib.nullcontext()

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device, dtype=torch.float32, non_blocking=True)
        if self.settings.channels_last and uses_cuda(self.device) and tensor.ndim == 4:
            return tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total = 0.0
        progress = epoch_progress("train", "bold green")
        with progress:
            task = progress.add_task("train", total=len(loader), **self._progress_fields(epoch))
            for batch_index, batch in enumerate(loader):
                mix = self._to_device(batch["mix"])
                vocal = self._to_device(batch["vocal"])
                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler is not None:
                    with self._autocast():
                        pred = self.model(mix)
                        loss = self.criterion(pred, vocal) * self.settings.loss_scale
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(mix)
                    loss = self.criterion(pred, vocal) * self.settings.loss_scale
                    loss.backward()
                    self.optimizer.step()

                total += loss.item()
                running_loss = total / (batch_index + 1)
                progress.update(
                    task,
                    advance=1,
                    **self._progress_fields(epoch, format_loss(running_loss)),
                )
                if batch_index == 0:
                    log_gpu_memory(self.device, "After first training batch")

        return total / len(loader)

    def _validate_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.eval()
        total = 0.0
        export_batch = random.randrange(max(len(loader), 1))

        progress = epoch_progress("valid", "bold yellow")
        with progress:
            task = progress.add_task("valid", total=len(loader), **self._progress_fields(epoch))
            with torch.no_grad():
                for batch_index, batch in enumerate(loader):
                    mix = self._to_device(batch["mix"])
                    vocal = self._to_device(batch["vocal"])
                    with self._autocast():
                        pred = self.model(mix)
                        loss = self.criterion(pred, vocal) * self.settings.loss_scale
                    total += loss.item()

                    should_export = batch_index == export_batch and (
                        self.settings.eval_only
                        or (epoch > 0 and epoch % self.settings.log_audio_every == 0)
                    )
                    if should_export:
                        self._export_audio(epoch, batch, pred)

                    running_loss = total / (batch_index + 1)
                    progress.update(
                        task,
                        advance=1,
                        **self._progress_fields(epoch, format_loss(running_loss)),
                    )

        return total / len(loader)

    def _export_audio(self, epoch: int, batch: dict[str, torch.Tensor], pred: torch.Tensor):
        index = random.randrange(batch["mix"].shape[0])
        mix = batch["mix"][index].cpu()
        vocal = batch["vocal"][index].cpu()
        prediction = pred[index].cpu()

        mix_mag = denormalize(mix.squeeze(0), batch["mix_min"][index], batch["mix_max"][index])
        vocal_mag = denormalize(
            vocal.squeeze(0), batch["vocal_min"][index], batch["vocal_max"][index]
        )
        pred_mag = denormalize(
            prediction.squeeze(0),
            batch["vocal_min"][index],
            batch["vocal_max"][index],
        )

        mix_path = self.run_paths.audio / f"{epoch}_mix.wav"
        vocal_path = self.run_paths.audio / f"{epoch}_vocal.wav"
        pred_path = self.run_paths.audio / f"{epoch}_pred.wav"

        write_waveform(mix_path, mix_mag, batch["mix_phase"][index], self.stft)
        write_waveform(vocal_path, vocal_mag, batch["vocal_phase"][index], self.stft)
        write_waveform(pred_path, pred_mag, batch["mix_phase"][index], self.stft)

        logger.info(
            "Exported audio samples for epoch [bold]%d[/bold] → [dim]%s[/dim]",
            epoch,
            self.run_paths.audio,
        )

        sdr = separation_sdr(vocal_path, pred_path)
        if sdr is None:
            logger.warning("Epoch [bold]%d[/bold]: no vocal energy in validation sample", epoch)
        else:
            logger.info(
                "Epoch [bold]%d[/bold] validation SDR: [green]%.2f dB[/green]",
                epoch,
                sdr,
            )

    def _save_loss_plot(self, epoch: int) -> Path:
        plt.figure()
        plt.plot(self.train_losses, label="train")
        plt.plot(self.val_losses, label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        path = self.run_paths.plots / f"loss_{epoch}.png"
        plt.savefig(path)
        plt.close()
        return path


def _find_latest_checkpoint(runs_dir: Path) -> Path | None:
    if not runs_dir.exists():
        return None
    candidates = sorted(runs_dir.glob("*/checkpoints/best.pt"))
    return candidates[-1] if candidates else None
