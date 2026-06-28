"""Datasets for lazy and memmap-backed training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vocal_sep.audio.spectrogram import SpectrogramSample, waveform_to_spectrogram
from vocal_sep.config import StftSettings, TrainSettings
from vocal_sep.data.index import SampleRef, TrackAudioCache
from vocal_sep.log import get_logger

logger = get_logger(__name__)


def _sample_to_dict(sample: SpectrogramSample) -> dict[str, torch.Tensor]:
    return {
        "mix": sample.mix.unsqueeze(0),
        "vocal": sample.vocal.unsqueeze(0),
        "mix_phase": sample.mix_phase,
        "vocal_phase": sample.vocal_phase,
        "mix_min": sample.mix_min,
        "mix_max": sample.mix_max,
        "vocal_min": sample.vocal_min,
        "vocal_max": sample.vocal_max,
    }


class LazyVocalDataset(Dataset):
    """Compute spectrograms on demand with per-track audio caching."""

    def __init__(
        self,
        tracks,
        sample_refs: list[SampleRef],
        settings: StftSettings,
    ):
        self.sample_refs = sample_refs
        self.settings = settings
        self.window_samples = settings.window_samples
        self.cache = TrackAudioCache(tracks, settings.sample_rate)

    def __len__(self) -> int:
        return len(self.sample_refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ref = self.sample_refs[index]
        mix, vocal = self.cache.get(ref.track_index)
        end = ref.start + self.window_samples
        mix_window = mix[ref.start : end]
        vocal_window = vocal[ref.start : end]

        if torch.norm(vocal_window) < 1e-3:
            fallback = (index + 1) % len(self.sample_refs)
            return self.__getitem__(fallback)

        sample = waveform_to_spectrogram(mix_window, vocal_window, self.settings)
        return _sample_to_dict(sample)


class MemmapVocalDataset(Dataset):
    """Read precomputed spectrograms from numpy memmap files.

    Memmap handles are opened lazily so pickling for DataLoader workers (Python 3.14+
    uses forkserver/spawn) does not copy the entire cache into RAM.
    """

    FIELD_NAMES = (
        "mix",
        "vocal",
        "mix_phase",
        "vocal_phase",
        "mix_min",
        "mix_max",
        "vocal_min",
        "vocal_max",
    )

    def __init__(self, cache_dir: Path, row_indices: list[int] | None = None):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / "meta.json"
        with meta_path.open() as handle:
            self.meta = json.load(handle)

        self.row_indices = row_indices
        self._arrays: dict[str, np.memmap] | None = None

    def _open_arrays(self) -> dict[str, np.memmap]:
        if self._arrays is None:
            self._arrays = {
                name: np.load(self.cache_dir / f"{name}.npy", mmap_mode="r")
                for name in self.FIELD_NAMES
            }
        return self._arrays

    def __getstate__(self) -> dict:
        return {
            "cache_dir": self.cache_dir,
            "row_indices": self.row_indices,
            "meta": self.meta,
        }

    def __setstate__(self, state: dict) -> None:
        self.cache_dir = Path(state["cache_dir"])
        self.row_indices = state["row_indices"]
        self.meta = state["meta"]
        self._arrays = None

    def __len__(self) -> int:
        return len(self.row_indices) if self.row_indices is not None else self.meta["num_samples"]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        arrays = self._open_arrays()
        row = self.row_indices[index] if self.row_indices is not None else index
        item = {}
        for name in self.FIELD_NAMES:
            # Memmap slices are read-only; copy so PyTorch gets a writable float32 tensor.
            row_slice = np.array(arrays[name][row], copy=True)
            tensor = torch.from_numpy(row_slice).float()
            if name in ("mix", "vocal") and tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            item[name] = tensor
        return item


def collate_batch(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in items], dim=0) for key in items[0]}


def _refs_for_split(
    sample_refs: list[SampleRef],
    row_indices: list[int] | None,
) -> list[SampleRef]:
    if row_indices is None:
        return sample_refs
    return [sample_refs[index] for index in row_indices]


def create_dataset(
    tracks,
    sample_refs: list[SampleRef],
    settings: TrainSettings,
    cache_dir: Path | None,
    row_indices: list[int] | None = None,
) -> Dataset:
    if settings.dataset_mode == "memmap":
        if cache_dir is None or not (cache_dir / "meta.json").exists():
            raise FileNotFoundError(
                f"Memmap cache missing at {cache_dir}. Run `vocal-sep cache` first."
            )
        dataset = MemmapVocalDataset(cache_dir, row_indices=row_indices)
        logger.info(
            "Using [bold cyan]memmap[/bold cyan] dataset — %d samples from %s",
            len(dataset),
            cache_dir,
        )
        return dataset

    if tracks is None:
        raise ValueError("Lazy dataset mode requires MUSDB tracks.")
    refs = _refs_for_split(sample_refs, row_indices)
    logger.info(
        "Using [bold cyan]lazy[/bold cyan] dataset — %d on-the-fly samples",
        len(refs),
    )
    return LazyVocalDataset(tracks, refs, settings.stft)


def split_sample_refs(
    sample_refs: list[SampleRef],
    val_ratio: float,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Return train/val index lists into the master sample_refs ordering."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(sample_refs))
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_idx = indices[:val_size].tolist()
    train_idx = indices[val_size:].tolist()
    logger.info(
        "Split indices — [bold green]%d[/bold green] train / [bold yellow]%d[/bold yellow] val",
        len(train_idx),
        len(val_idx),
    )
    return train_idx, val_idx
