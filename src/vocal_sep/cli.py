"""Command-line interface."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from vocal_sep.config import TrainSettings
from vocal_sep.data.dataset import create_dataset, split_sample_refs
from vocal_sep.data.loaders import build_dataloader
from vocal_sep.data.pipeline import (
    build_memmap_cache_from_settings,
    ensure_memmap_cache,
    load_sample_index,
)
from vocal_sep.device import clamp_batch_size, configure_runtime, resolve_device, uses_cuda
from vocal_sep.log import (
    get_logger,
    log_banner,
    log_device_summary,
    log_rule,
    log_settings,
    setup_logging,
)
from vocal_sep.training.trainer import Trainer

logger = get_logger(__name__)


def _load_settings(config: Path, overrides: dict | None = None) -> TrainSettings:
    settings = TrainSettings.from_yaml(config, overrides=overrides)
    logger.debug("Loaded settings from %s", config)
    return settings


def _prepare_dataloaders(settings: TrainSettings, device):
    log_rule("Dataset", style="bright_green")
    tracks, sample_refs = load_sample_index(settings)
    train_idx, val_idx = split_sample_refs(sample_refs, settings.val_ratio)

    if uses_cuda(device) and settings.dataset_mode == "lazy":
        logger.warning(
            "[bold yellow]lazy[/bold yellow] mode computes STFT on CPU and starves the GPU. "
            "Switching to [bold cyan]memmap[/bold cyan] for CUDA training."
        )
        settings = settings.with_updates(dataset_mode="memmap")

    cache_dir: Path | None = None
    if settings.dataset_mode == "memmap":
        cache_dir = ensure_memmap_cache(settings, tracks, sample_refs)
        del tracks
        gc.collect()
        logger.debug("Released MUSDB track audio from RAM after cache build")
        tracks = None

    logger.info(
        "Sample split: [bold green]%d[/bold green] train / "
        "[bold yellow]%d[/bold yellow] val (mode=[bold cyan]%s[/bold cyan])",
        len(train_idx),
        len(val_idx),
        settings.dataset_mode,
    )

    train_set = create_dataset(tracks, sample_refs, settings, cache_dir, row_indices=train_idx)
    val_set = create_dataset(tracks, sample_refs, settings, cache_dir, row_indices=val_idx)
    return (
        build_dataloader(train_set, settings, shuffle=True),
        build_dataloader(val_set, settings, shuffle=False),
        settings,
    )


def _optimize_settings_for_device(settings: TrainSettings, device) -> TrainSettings:
    if not uses_cuda(device):
        return settings

    safe_batch = clamp_batch_size(settings.batch_size, device)
    updates: dict = {"batch_size": safe_batch}
    if settings.num_workers < 4:
        updates["num_workers"] = 4
    return settings.with_updates(**updates)


def cmd_train(args: argparse.Namespace) -> None:
    log_banner("train")
    overrides: dict = {
        k: v
        for k, v in {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "num_samples": args.num_samples,
            "resume": args.resume or None,
            "log_level": args.log_level,
            "device": args.device,
        }.items()
        if v is not None
    }
    if args.eval:
        overrides.update(eval_only=True, epochs=1, resume=True)
    if args.verbose:
        overrides["log_level"] = "DEBUG"
    if args.quiet:
        overrides["log_level"] = "WARNING"

    settings = _load_settings(Path(args.config), overrides=overrides)
    setup_logging(settings.log_level, show_path=settings.log_level == "DEBUG")

    device = resolve_device(settings.device)
    configure_runtime(device)

    log_rule("Configuration", style="bright_blue")
    log_settings(settings, config_path=str(args.config))
    log_device_summary(device)

    settings = _optimize_settings_for_device(settings, device)
    logger.info(
        "Training with batch_size=[bold]%d[/bold] on [bold]%s[/bold]",
        settings.batch_size,
        device,
    )

    train_loader, val_loader, settings = _prepare_dataloaders(settings, device)

    log_rule("Training", style="bright_magenta")
    trainer = Trainer(settings, device=device)
    trainer.fit(train_loader, val_loader)
    logger.info(
        "[bold green]Done.[/bold green] Artifacts saved under [bold]%s[/bold]",
        trainer.run_paths.root,
    )


def cmd_cache(args: argparse.Namespace) -> None:
    log_banner("cache")
    overrides: dict = {}
    if args.verbose:
        overrides["log_level"] = "DEBUG"
    if args.quiet:
        overrides["log_level"] = "WARNING"
    if args.log_level:
        overrides["log_level"] = args.log_level

    settings = _load_settings(Path(args.config), overrides=overrides or None)
    setup_logging(settings.log_level, show_path=settings.log_level == "DEBUG")

    log_rule("Configuration", style="bright_blue")
    log_settings(settings, config_path=str(args.config))

    log_rule("Cache Build", style="bright_yellow")
    cache_dir = build_memmap_cache_from_settings(settings)
    logger.info("[bold green]Cache ready[/bold green] at [bold]%s[/bold]", cache_dir)


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging verbosity",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only warnings and errors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vocal source separation")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train or evaluate the separator")
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--num-samples", type=int, default=None)
    train.add_argument("--eval", action="store_true", help="Evaluation-only mode")
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Compute device (default: cuda from config)",
    )
    _add_logging_args(train)
    train.set_defaults(func=cmd_train)

    cache = sub.add_parser("cache", help="Build memmap spectrogram cache for fast training")
    cache.add_argument("--config", default="configs/default.yaml")
    _add_logging_args(cache)
    cache.set_defaults(func=cmd_cache)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
