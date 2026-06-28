"""Shared MUSDB index and memmap cache setup for CLI commands."""

from __future__ import annotations

import gc
from pathlib import Path

from vocal_sep.config import TrainSettings
from vocal_sep.data.cache import build_memmap_cache
from vocal_sep.data.index import SampleRef, build_sample_index
from vocal_sep.data.musdb import ensure_musdb
from vocal_sep.log import get_logger
from vocal_sep.paths import CACHE_DIR

logger = get_logger(__name__)


def select_tracks(db, settings: TrainSettings):
    tracks = db.tracks
    if settings.num_tracks is not None:
        tracks = tracks[: settings.num_tracks]
        logger.info(
            "Using [bold]%d[/bold] of [bold]%d[/bold] MUSDB tracks",
            len(tracks),
            len(db.tracks),
        )
    else:
        logger.info("Using all [bold]%d[/bold] MUSDB tracks", len(tracks))
    return tracks


def load_sample_index(settings: TrainSettings) -> tuple[list, list[SampleRef]]:
    """Load MUSDB tracks and build the master sample window index."""
    db = ensure_musdb()
    tracks = select_tracks(db, settings)
    sample_refs = build_sample_index(
        tracks,
        settings.num_samples,
        settings.stft.window_samples,
        settings.stft.sample_rate,
    )
    return tracks, sample_refs


def ensure_memmap_cache(settings: TrainSettings, tracks, sample_refs: list[SampleRef]) -> Path:
    """Return an existing memmap cache or build it on disk."""
    cache_dir = CACHE_DIR / settings.cache_key()
    if (cache_dir / "meta.json").exists():
        logger.info("Using existing spectrogram cache at [bold]%s[/bold]", cache_dir)
        return cache_dir

    logger.info(
        "[bold yellow]Building spectrogram cache[/bold yellow] "
        "(one-time step; keeps the GPU fed during training)"
    )
    build_memmap_cache(
        tracks,
        sample_refs,
        settings.stft,
        cache_dir,
        max_workers=settings.cache_workers,
        use_gpu=settings.cache_use_gpu,
        stft_chunk_size=settings.cache_stft_chunk_size,
        window_batch_size=settings.cache_window_batch,
    )
    return cache_dir


def build_memmap_cache_from_settings(settings: TrainSettings) -> Path:
    """Load MUSDB and build the spectrogram cache (cache CLI command)."""
    tracks, sample_refs = load_sample_index(settings)
    cache_dir = ensure_memmap_cache(settings, tracks, sample_refs)
    del tracks
    gc.collect()
    return cache_dir
