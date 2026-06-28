"""Colored logging utilities built on Rich."""

from __future__ import annotations

import logging
from dataclasses import fields
from typing import Any

import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from vocal_sep.config import StftSettings, TrainSettings
from vocal_sep.device import cuda_available, uses_cuda

_console = Console(stderr=True)
_configured = False


def setup_logging(level: str = "INFO", *, show_path: bool = False) -> None:
    """Configure root logging with Rich color output."""
    global _configured
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    handler = RichHandler(
        console=_console,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
        show_time=True,
        show_level=True,
        show_path=show_path,
        log_time_format="[%X]",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    if not _configured:
        setup_logging()
    return logging.getLogger(name)


def console() -> Console:
    return _console


def log_banner(command: str) -> None:
    title = Text("vocal-sep", style="bold magenta")
    subtitle = Text(f" {command}", style="bold cyan")
    _console.print(Panel(title + subtitle, border_style="bright_blue", padding=(0, 2)))


def log_rule(title: str, style: str = "bright_blue") -> None:
    _console.print(Rule(title, style=style))


def log_kv_table(title: str, rows: dict[str, Any], *, style: str = "cyan") -> None:
    table = Table(title=title, show_header=True, header_style=f"bold {style}")
    table.add_column("Setting", style="bold")
    table.add_column("Value", style="white")
    for key, value in rows.items():
        table.add_row(str(key), str(value))
    _console.print(table)


def log_settings(settings: TrainSettings, config_path: str | None = None) -> None:
    rows: dict[str, Any] = {}
    if config_path:
        rows["config"] = config_path
    for field in fields(TrainSettings):
        if field.name == "stft":
            continue
        rows[field.name] = getattr(settings, field.name)
    log_kv_table("Training Settings", rows)

    stft_rows = {field.name: getattr(settings.stft, field.name) for field in fields(StftSettings)}
    log_kv_table("STFT Settings", stft_rows, style="green")


def log_device_summary(device: torch.device | None = None) -> None:
    if device is None:
        device = torch.device("cuda" if cuda_available() else "cpu")
    rows = {
        "device": str(device),
        "torch": torch.__version__,
    }
    if uses_cuda(device):
        rows["gpu"] = torch.cuda.get_device_name(device)
        rows["cuda"] = torch.version.cuda or "unknown"
        memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        rows["gpu_memory_gb"] = f"{memory_gb:.1f}"
    log_kv_table("Runtime", rows, style="magenta")


def download_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    )


def task_progress(description: str) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold cyan]{description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=_console,
        transient=False,
    )


def format_loss(value: float) -> str:
    if value == float("inf") or value != value:
        return "—"
    return f"{value:.4f}"


def format_lr(value: float) -> str:
    return f"{value:.1e}"


def epoch_progress(description: str, style: str) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[{style}]{description}[/{style}] e{{task.fields[epoch]}}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("loss [cyan]{task.fields[loss]}[/cyan]"),
        TextColumn("best [yellow]{task.fields[best]}[/yellow]"),
        TextColumn("lr [magenta]{task.fields[lr]}[/magenta]"),
        TextColumn("pat [dim]{task.fields[patience]}[/dim]"),
        TimeElapsedColumn(),
        console=_console,
        transient=True,
    )
