"""Validation separation metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from vocal_sep.log import get_logger

logger = get_logger(__name__)


def _load_mono(path: Path) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        return audio[:, 0]
    return audio.mean(axis=1)


def _signal_to_distortion_ratio(reference: np.ndarray, estimate: np.ndarray) -> float:
    length = min(len(reference), len(estimate))
    reference = reference[:length]
    estimate = estimate[:length]

    if np.sum(reference**2) < 1e-8:
        raise ValueError("Reference source has no energy")

    error = estimate - reference
    source_power = float(np.sum(reference**2))
    error_power = float(np.sum(error**2))
    return 10.0 * np.log10(source_power / (error_power + 1e-8))


def separation_sdr(reference_wav: Path, estimate_wav: Path) -> float | None:
    try:
        reference = _load_mono(reference_wav)
        estimate = _load_mono(estimate_wav)
        sdr = _signal_to_distortion_ratio(reference, estimate)
        logger.debug("Computed SDR for %s — %.2f dB", estimate_wav.name, sdr)
        return sdr
    except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
        logger.debug("Could not score %s: %s", estimate_wav.name, exc)
        return None
