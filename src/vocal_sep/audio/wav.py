"""WAV export helpers."""

from pathlib import Path

import soundfile as sf
import torch

from vocal_sep.audio.spectrogram import reconstruct_waveform
from vocal_sep.config import StftSettings
from vocal_sep.log import get_logger

logger = get_logger(__name__)


def write_waveform(
    path: Path,
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    settings: StftSettings,
) -> None:
    waveform = reconstruct_waveform(magnitude, phase, settings)
    sf.write(path, waveform.cpu().numpy(), settings.sample_rate, subtype="PCM_16")
    logger.debug("Wrote WAV → %s (%d samples)", path, len(waveform))
