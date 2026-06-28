"""Configuration loading."""

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

VALID_DATASET_MODES = frozenset({"lazy", "memmap"})
VALID_LR_SCHEDULERS = frozenset({"plateau", "none"})


@dataclass(frozen=True)
class StftSettings:
    sample_rate: int = 8192
    n_fft: int = 1023
    hop_length: int = 768
    win_length: int = 1023
    n_frames: int = 128

    @property
    def window_samples(self) -> int:
        return (self.n_frames - 1) * self.hop_length + 1


@dataclass(frozen=True)
class TrainSettings:
    num_samples: int = 20_000
    num_tracks: int | None = None
    batch_size: int = 64
    epochs: int = 10_000
    val_ratio: float = 0.1
    patience: int = 25
    min_delta: float = 0.1
    loss_scale: float = 100.0
    learning_rate: float = 1e-3
    lr_scheduler: str = "plateau"
    lr_patience: int = 8
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    amsgrad: bool = True
    resume: bool = False
    eval_only: bool = False
    dataset_mode: str = "memmap"
    num_workers: int = 4
    cache_workers: int = 2
    cache_use_gpu: bool = True
    cache_stft_chunk_size: int = 512
    cache_window_batch: int = 32
    pin_memory: bool = True
    prefetch_factor: int = 2
    log_audio_every: int = 10
    log_loss_every: int = 10
    log_level: str = "INFO"
    device: str = "cuda"
    channels_last: bool = True
    auto_build_cache: bool = True
    compile_model: bool = False
    stft: StftSettings = field(default_factory=StftSettings)

    def with_updates(self, **kwargs: Any) -> "TrainSettings":
        return replace(self, **kwargs)

    @classmethod
    def from_yaml(
        cls,
        path: Path | str,
        overrides: dict[str, Any] | None = None,
    ) -> "TrainSettings":
        with open(path) as handle:
            raw = yaml.safe_load(handle)

        if overrides:
            raw.update({k: v for k, v in overrides.items() if v is not None})

        stft_raw = {k: raw.pop(k) for k in list(raw) if k in StftSettings.__dataclass_fields__}
        stft = StftSettings(**stft_raw)
        known = {field.name for field in fields(cls) if field.name != "stft"}
        train_raw = {key: raw[key] for key in raw if key in known}
        settings = cls(stft=stft, **train_raw)
        if settings.dataset_mode not in VALID_DATASET_MODES:
            raise ValueError(
                f"dataset_mode must be one of {sorted(VALID_DATASET_MODES)}, "
                f"got {settings.dataset_mode!r}"
            )
        if settings.lr_scheduler not in VALID_LR_SCHEDULERS:
            raise ValueError(
                f"lr_scheduler must be one of {sorted(VALID_LR_SCHEDULERS)}, "
                f"got {settings.lr_scheduler!r}"
            )
        return settings

    def cache_key(self) -> str:
        tracks = self.num_tracks if self.num_tracks is not None else "all"
        s = self.stft
        return (
            f"s{self.num_samples}_t{tracks}_"
            f"sr{s.sample_rate}_nf{s.n_fft}_h{s.hop_length}_w{s.win_length}_f{s.n_frames}"
        )
