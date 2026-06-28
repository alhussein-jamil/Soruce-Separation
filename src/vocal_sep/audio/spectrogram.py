"""STFT extraction and magnitude normalization."""

from dataclasses import dataclass

import torch

from vocal_sep.config import StftSettings


@dataclass(frozen=True)
class SpectrogramSample:
    mix: torch.Tensor
    vocal: torch.Tensor
    mix_phase: torch.Tensor
    vocal_phase: torch.Tensor
    mix_min: torch.Tensor
    mix_max: torch.Tensor
    vocal_min: torch.Tensor
    vocal_max: torch.Tensor


def _normalize_magnitude(
    magnitude: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Min-max normalize along the time axis using vectorized ops."""
    min_val = magnitude.amin(dim=-1, keepdim=True)
    max_val = magnitude.amax(dim=-1, keepdim=True)
    span = (max_val - min_val).clamp_min(1e-8)
    normalized = (magnitude - min_val) / span
    return normalized, min_val, max_val


def _analysis_window(settings: StftSettings, device: torch.device) -> torch.Tensor:
    return torch.hann_window(settings.win_length, device=device)


def _stft_kwargs(settings: StftSettings, waveform: torch.Tensor) -> dict:
    return dict(
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        window=_analysis_window(settings, waveform.device),
        return_complex=True,
    )


def _spectrogram_from_stft(mix_stft: torch.Tensor, vocal_stft: torch.Tensor) -> SpectrogramSample:
    mix_norm, mix_min, mix_max = _normalize_magnitude(mix_stft.abs())
    vocal_norm, vocal_min, vocal_max = _normalize_magnitude(vocal_stft.abs())
    return SpectrogramSample(
        mix=mix_norm.float(),
        vocal=vocal_norm.float(),
        mix_phase=mix_stft.angle().float(),
        vocal_phase=vocal_stft.angle().float(),
        mix_min=mix_min.float(),
        mix_max=mix_max.float(),
        vocal_min=vocal_min.float(),
        vocal_max=vocal_max.float(),
    )


def waveform_to_spectrogram(
    mix: torch.Tensor,
    vocal: torch.Tensor,
    settings: StftSettings,
) -> SpectrogramSample:
    """Convert mono waveforms shaped [samples] into normalized spectrogram tensors."""
    kwargs = _stft_kwargs(settings, mix)
    mix_stft = torch.stft(mix.unsqueeze(0), **kwargs).squeeze(0)
    vocal_stft = torch.stft(vocal.unsqueeze(0), **kwargs).squeeze(0)
    return _spectrogram_from_stft(mix_stft, vocal_stft)


def batch_windows_to_spectrograms(
    mix_windows: torch.Tensor,
    vocal_windows: torch.Tensor,
    settings: StftSettings,
    *,
    device: torch.device | str | None = None,
    chunk_size: int = 256,
) -> list[SpectrogramSample]:
    """Vectorized STFT for many windows from the same track."""
    if device is not None:
        mix_windows = mix_windows.to(device, non_blocking=True)
        vocal_windows = vocal_windows.to(device, non_blocking=True)

    samples: list[SpectrogramSample] = []
    total = mix_windows.shape[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        mix_chunk = mix_windows[start:end]
        vocal_chunk = vocal_windows[start:end]

        kwargs = _stft_kwargs(settings, mix_chunk)
        mix_stft = torch.stft(mix_chunk, **kwargs)
        vocal_stft = torch.stft(vocal_chunk, **kwargs)

        if device is not None and mix_stft.device.type != "cpu":
            mix_stft = mix_stft.cpu()
            vocal_stft = vocal_stft.cpu()

        for index in range(mix_chunk.shape[0]):
            sample = _spectrogram_from_stft(mix_stft[index], vocal_stft[index])
            samples.append(sample)
    return samples


def denormalize(
    magnitude: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
) -> torch.Tensor:
    return magnitude * (max_val - min_val) + min_val


def reconstruct_waveform(
    magnitude: torch.Tensor,
    phase: torch.Tensor,
    settings: StftSettings,
) -> torch.Tensor:
    complex_spec = torch.polar(magnitude, phase)
    return torch.istft(
        complex_spec,
        n_fft=settings.n_fft,
        hop_length=settings.hop_length,
        win_length=settings.win_length,
        window=_analysis_window(settings, complex_spec.device),
    )
