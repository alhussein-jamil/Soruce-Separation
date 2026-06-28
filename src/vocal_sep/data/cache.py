"""Parallel spectrogram cache builder with streaming memmap writes."""

from __future__ import annotations

import gc
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch

from vocal_sep.audio.spectrogram import (
    SpectrogramSample,
    batch_windows_to_spectrograms,
    waveform_to_spectrogram,
)
from vocal_sep.config import StftSettings
from vocal_sep.data.index import SampleRef, TrackAudioCache
from vocal_sep.device import cuda_available
from vocal_sep.log import get_logger, task_progress

logger = get_logger(__name__)

_write_lock = threading.Lock()
_gpu_stft_lock = threading.Lock()


@dataclass(frozen=True)
class CacheBuildOptions:
    max_workers: int = 2
    use_gpu: bool = True
    stft_chunk_size: int = 512
    window_batch_size: int = 32


def _log_ram(label: str) -> None:
    used_gb = psutil.Process().memory_info().rss / 1e9
    avail_gb = psutil.virtual_memory().available / 1e9
    logger.info(
        "[dim]%s[/dim] — process RAM [bold]%.1f GB[/bold] ([dim]%.1f GB system free[/dim])",
        label,
        used_gb,
        avail_gb,
    )


def _open_memmaps(cache_dir: Path, shapes: dict[str, tuple]) -> dict[str, np.memmap]:
    return {
        name: np.lib.format.open_memmap(
            cache_dir / f"{name}.npy",
            mode="r+",
            dtype=np.float16,
            shape=shape,
        )
        for name, shape in shapes.items()
    }


def _write_sample(arrays: dict[str, np.memmap], row: int, sample: SpectrogramSample) -> None:
    with _write_lock:
        arrays["mix"][row, 0] = sample.mix.numpy().astype(np.float16)
        arrays["vocal"][row, 0] = sample.vocal.numpy().astype(np.float16)
        arrays["mix_phase"][row] = sample.mix_phase.numpy().astype(np.float16)
        arrays["vocal_phase"][row] = sample.vocal_phase.numpy().astype(np.float16)
        arrays["mix_min"][row] = sample.mix_min.numpy().astype(np.float16)
        arrays["mix_max"][row] = sample.mix_max.numpy().astype(np.float16)
        arrays["vocal_min"][row] = sample.vocal_min.numpy().astype(np.float16)
        arrays["vocal_max"][row] = sample.vocal_max.numpy().astype(np.float16)


def _probe_sample_shape(
    tracks,
    sample_refs: list[SampleRef],
    settings: StftSettings,
) -> SpectrogramSample:
    cache = TrackAudioCache(tracks, settings.sample_rate)
    window = settings.window_samples
    for ref in sample_refs:
        mix, vocal = cache.get(ref.track_index)
        end = ref.start + window
        vocal_window = vocal[ref.start : end]
        if torch.norm(vocal_window) < 1e-3:
            continue
        return waveform_to_spectrogram(mix[ref.start : end], vocal_window, settings)
    raise RuntimeError("No valid vocal samples found while probing cache shapes.")


def _process_track(
    track_index: int,
    refs: list[SampleRef],
    tracks,
    settings: StftSettings,
    ref_positions: dict[tuple[int, int], int],
    cache_dir: Path,
    shapes: dict[str, tuple],
    options: CacheBuildOptions,
) -> tuple[int, int]:
    """Process one track and stream spectrograms into the shared memmap files."""
    arrays = _open_memmaps(cache_dir, shapes)
    audio_cache = TrackAudioCache(tracks, settings.sample_rate)
    window = settings.window_samples
    stft_device = None
    if options.use_gpu and cuda_available():
        stft_device = torch.device("cuda:0")

    mix, vocal = audio_cache.get(track_index)
    written = 0
    dropped = 0

    for batch_start in range(0, len(refs), options.window_batch_size):
        batch_refs = refs[batch_start : batch_start + options.window_batch_size]
        mix_windows = []
        vocal_windows = []
        starts = []
        for ref in batch_refs:
            end = ref.start + window
            mix_windows.append(mix[ref.start : end])
            vocal_windows.append(vocal[ref.start : end])
            starts.append(ref.start)

        mix_batch = torch.stack(mix_windows)
        vocal_batch = torch.stack(vocal_windows)
        keep = torch.norm(vocal_batch, dim=1) >= 1e-3
        dropped += len(batch_refs) - int(keep.sum().item())

        if keep.any():
            if stft_device is not None:
                with _gpu_stft_lock:
                    samples = batch_windows_to_spectrograms(
                        mix_batch[keep],
                        vocal_batch[keep],
                        settings,
                        device=stft_device,
                        chunk_size=options.stft_chunk_size,
                    )
            else:
                samples = batch_windows_to_spectrograms(
                    mix_batch[keep],
                    vocal_batch[keep],
                    settings,
                    device=None,
                    chunk_size=options.stft_chunk_size,
                )
            kept_starts = [start for start, ok in zip(starts, keep.tolist(), strict=False) if ok]
            for start, sample in zip(kept_starts, samples, strict=False):
                row = ref_positions.get((track_index, start))
                if row is not None:
                    _write_sample(arrays, row, sample)
                    written += 1
            del samples

        del mix_windows, vocal_windows, mix_batch, vocal_batch

    audio_cache.release(track_index)
    del mix, vocal
    for array in arrays.values():
        array.flush()
        del array
    if stft_device is not None:
        torch.cuda.empty_cache()
    gc.collect()
    return written, dropped


def build_memmap_cache(
    tracks,
    sample_refs: list[SampleRef],
    settings: StftSettings,
    cache_dir: Path,
    max_workers: int = 2,
    *,
    use_gpu: bool = True,
    stft_chunk_size: int = 512,
    window_batch_size: int = 32,
) -> Path:
    """Precompute spectrograms in parallel and stream them into float16 memmaps."""
    options = CacheBuildOptions(
        max_workers=max(1, max_workers),
        use_gpu=use_gpu,
        stft_chunk_size=stft_chunk_size,
        window_batch_size=max(1, window_batch_size),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    ref_positions = {(ref.track_index, ref.start): index for index, ref in enumerate(sample_refs)}

    grouped: dict[int, list[SampleRef]] = defaultdict(list)
    for ref in sample_refs:
        grouped[ref.track_index].append(ref)

    logger.info(
        "Parallel cache build — [bold]%d[/bold] samples, [bold]%d[/bold] tracks, "
        "[bold]%d[/bold] workers, GPU STFT=%s, chunk=%d",
        len(sample_refs),
        len(grouped),
        options.max_workers,
        options.use_gpu and cuda_available(),
        options.stft_chunk_size,
    )
    _log_ram("Before cache build")

    example = _probe_sample_shape(tracks, sample_refs, settings)
    capacity = len(sample_refs)
    shapes = {
        "mix": (capacity, 1, *example.mix.shape),
        "vocal": (capacity, 1, *example.vocal.shape),
        "mix_phase": (capacity, *example.mix_phase.shape),
        "vocal_phase": (capacity, *example.vocal_phase.shape),
        "mix_min": (capacity, *example.mix_min.shape),
        "mix_max": (capacity, *example.mix_max.shape),
        "vocal_min": (capacity, *example.vocal_min.shape),
        "vocal_max": (capacity, *example.vocal_max.shape),
    }

    # Create empty memmap files once in the parent process.
    placeholders = {
        name: np.lib.format.open_memmap(
            cache_dir / f"{name}.npy",
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
        for name, shape in shapes.items()
    }
    for array in placeholders.values():
        del array

    written = 0
    dropped = 0
    track_indices = sorted(grouped.keys())

    if options.max_workers == 1:
        iterator = track_indices
        with task_progress("Caching STFT") as progress:
            task = progress.add_task("cache", total=len(iterator))
            for track_index in iterator:
                w, d = _process_track(
                    track_index,
                    grouped[track_index],
                    tracks,
                    settings,
                    ref_positions,
                    cache_dir,
                    shapes,
                    options,
                )
                written += w
                dropped += d
                progress.advance(task)
    else:
        with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
            futures = {
                executor.submit(
                    _process_track,
                    track_index,
                    grouped[track_index],
                    tracks,
                    settings,
                    ref_positions,
                    cache_dir,
                    shapes,
                    options,
                ): track_index
                for track_index in track_indices
            }
            with task_progress("Caching STFT (parallel)") as progress:
                task = progress.add_task("cache", total=len(futures))
                for future in as_completed(futures):
                    w, d = future.result()
                    written += w
                    dropped += d
                    progress.advance(task)

    gc.collect()
    if written == 0:
        raise RuntimeError("No valid vocal samples were cached.")

    with (cache_dir / "meta.json").open("w") as handle:
        json.dump(
            {
                "num_samples": capacity,
                "num_written": written,
                "dtype": "float16",
            },
            handle,
        )

    _log_ram("After cache build")
    if dropped:
        logger.warning(
            "Skipped [bold yellow]%d[/bold yellow] silent/invalid windows during cache build",
            dropped,
        )
    logger.info(
        "[bold green]Cache built[/bold green] — %d rows at [bold]%s[/bold] "
        "([dim]~%.1f GB on disk[/dim])",
        capacity,
        cache_dir,
        sum(path.stat().st_size for path in cache_dir.glob("*.npy")) / 1e9,
    )
    return cache_dir
