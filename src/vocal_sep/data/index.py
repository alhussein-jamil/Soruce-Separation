"""Sample index and track-level audio caching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from vocal_sep.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SampleRef:
    track_index: int
    start: int


class TrackAudioCache:
    """Load each track's mono mix/vocal pair once and reuse across samples."""

    def __init__(self, tracks, sample_rate: int):
        self.tracks = tracks
        self.sample_rate = sample_rate
        self._mix: dict[int, torch.Tensor] = {}
        self._vocal: dict[int, torch.Tensor] = {}

    def get(self, track_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if track_index not in self._mix:
            track = self.tracks[track_index]
            track.sample_rate = self.sample_rate
            logger.debug("Loading track %d into audio cache: %s", track_index, track.name)
            mix = torch.mean(torch.from_numpy(track.audio.T), dim=0)
            vocal = torch.mean(torch.from_numpy(track.targets["vocals"].audio.T), dim=0)
            self._mix[track_index] = mix
            self._vocal[track_index] = vocal
        return self._mix[track_index], self._vocal[track_index]

    def release(self, track_index: int) -> None:
        """Drop a track from RAM once its windows have been processed."""
        self._mix.pop(track_index, None)
        self._vocal.pop(track_index, None)


def build_sample_index(
    tracks,
    num_samples: int,
    window_samples: int,
    sample_rate: int,
    rng: np.random.Generator | None = None,
) -> list[SampleRef]:
    """Create random window references distributed across tracks."""
    rng = rng or np.random.default_rng()
    num_tracks = len(tracks)
    if num_tracks == 0:
        raise ValueError("Cannot build sample index from an empty track list")
    per_track = max(1, math.ceil(num_samples / num_tracks))
    refs: list[SampleRef] = []

    for track_index, track in enumerate(tracks):
        track.sample_rate = sample_rate
        track_len = int(track.duration * sample_rate)
        if track_len <= window_samples:
            continue
        starts = rng.integers(0, track_len - window_samples, size=per_track)
        starts = np.unique(starts)
        for start in starts:
            refs.append(SampleRef(track_index=track_index, start=int(start)))

    rng.shuffle(refs)
    selected = refs[:num_samples]
    logger.info(
        "Built sample index — [bold]%d[/bold] windows "
        "([bold]%d[/bold] tracks, %d samples/track target)",
        len(selected),
        num_tracks,
        per_track,
    )
    logger.debug("Window length: %d samples", window_samples)
    return selected
