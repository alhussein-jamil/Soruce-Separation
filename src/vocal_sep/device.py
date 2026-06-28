"""CUDA device selection and GPU-oriented PyTorch tuning."""

from __future__ import annotations

import functools
import logging
import os

import torch

from vocal_sep.models.unet import MaskUNet

logger = logging.getLogger(__name__)


def cuda_available() -> bool:
    return torch.cuda.is_available()


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the best available device."""
    choice = requested.lower().strip()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available. "
                f"PyTorch build: {torch.__version__}, cuda: {torch.version.cuda}"
            )
        index = 0 if choice == "cuda" else int(choice.split(":")[1])
        return torch.device(f"cuda:{index}")

    if choice == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available on this system.")
        return torch.device("mps")

    return torch.device("cpu")


def uses_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def configure_runtime(device: torch.device) -> None:
    """Enable backend optimizations appropriate for the selected device."""
    if not uses_cuda(device):
        return

    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Reduce fragmentation on 8 GB laptop GPUs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    name = torch.cuda.get_device_name(device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
    logger.info(
        "CUDA runtime configured for [bold]%s[/bold] (%.1f GB) — cuDNN benchmark + TF32 enabled",
        name,
        total_gb,
    )


def probe_max_batch_size(device: torch.device, model: torch.nn.Module | None = None) -> int:
    """Find the largest safe batch size for the U-Net on the current GPU."""
    if not uses_cuda(device):
        return 64

    model = model if model is not None else _probe_model()
    model = model.to(device)
    model.eval()
    candidates = [512, 384, 256, 192, 128, 96, 64, 32]
    for batch_size in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            sample = torch.randn(batch_size, 1, 512, 128, device=device)
            with torch.autocast(device.type):
                output = model(sample)
            torch.cuda.synchronize(device)
            del sample, output
            logger.debug("Batch probe succeeded at size %d", batch_size)
            return batch_size
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            logger.debug("Batch probe OOM at size %d", batch_size)

    return 32


def clamp_batch_size(
    requested: int,
    device: torch.device,
    model: torch.nn.Module | None = None,
) -> int:
    """Cap batch size to what fits in VRAM and log the decision."""
    if not uses_cuda(device):
        return requested

    safe = probe_max_batch_size(device, model)
    if requested > safe:
        logger.warning(
            "Requested batch_size=%d exceeds GPU capacity; using [bold]%d[/bold] instead",
            requested,
            safe,
        )
        return safe

    logger.info(
        "Batch size [bold]%d[/bold] fits on GPU (tested up to %d)",
        requested,
        safe,
    )
    return requested


def log_gpu_memory(device: torch.device, label: str = "GPU memory") -> None:
    if not uses_cuda(device):
        return
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    logger.info(
        "%s — allocated [bold cyan]%.2f GB[/bold cyan] / "
        "reserved [yellow]%.2f GB[/yellow] / "
        "total [dim]%.2f GB[/dim]",
        label,
        allocated,
        reserved,
        total,
    )


@functools.lru_cache(maxsize=1)
def _probe_model() -> MaskUNet:
    return MaskUNet()
