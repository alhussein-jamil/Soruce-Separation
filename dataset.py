from torch.utils.data import Dataset
import torch
import numpy as np
import random
import copy
import os
import math


def preprocess_track(
    track, only_keep, n_fft, hop_length, window_size, samples_per_track
):
    x = torch.mean(torch.from_numpy(track.audio.T), dim=0)
    y = torch.mean(torch.from_numpy(track.targets["vocals"].audio.T), dim=0)

    starting_ids = np.random.randint(0, len(x) - only_keep, samples_per_track)
    # remove repeated starting ids
    starting_ids = np.unique(starting_ids)
    x = torch.stack(
        [x[starting_id : starting_id + only_keep] for starting_id in starting_ids],
        dim=0,
    )
    y = torch.stack(
        [y[starting_id : starting_id + only_keep] for starting_id in starting_ids],
        dim=0,
    )

    # remove samples with no vocals
    zero_vocals = torch.norm(y, dim=1) < 1e-3
    x = x[~zero_vocals]
    y = y[~zero_vocals]
    if len(x) == 0:
        return False

    x_stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_size,
        return_complex=True,
    )

    y_stft = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=window_size,
        return_complex=True,
    )

    x_amp = torch.abs(x_stft).float()
    y_amp = torch.abs(y_stft).float()

    x_phase = torch.angle(x_stft).float()
    y_phase = torch.angle(y_stft).float()

    min_x_amp, max_x_amp = x_amp.min(dim=2).values, x_amp.max(dim=2).values
    min_y_amp, max_y_amp = y_amp.min(dim=2).values, y_amp.max(dim=2).values

    min_x_amp = min_x_amp.unsqueeze(2).repeat(1, 1, x_amp.shape[2])
    max_x_amp = max_x_amp.unsqueeze(2).repeat(1, 1, x_amp.shape[2])
    min_y_amp = min_y_amp.unsqueeze(2).repeat(1, 1, y_amp.shape[2])
    max_y_amp = max_y_amp.unsqueeze(2).repeat(1, 1, y_amp.shape[2])

    max_x_amp[max_x_amp == min_x_amp] = 1
    min_x_amp[max_x_amp == min_x_amp] = 0
    max_y_amp[max_y_amp == min_y_amp] = 1
    min_y_amp[max_y_amp == min_y_amp] = 0

    x_amp = (x_amp - min_x_amp) / (max_x_amp - min_x_amp)
    y_amp = (y_amp - min_y_amp) / (max_y_amp - min_y_amp)

    return x_amp, y_amp, x_phase, y_phase, min_x_amp, max_x_amp, min_y_amp, max_y_amp


class MusdbDataset(Dataset):
    def __init__(
        self,
        mus,
        size=None,
        chunk_duration=12,
        sample_rate=8192,
        hop_length=768,
        window_size=1023,
        n_fft=1023,
        n_frames=128,
        samples_per_track=20,
        path=None,
    ):
        if size is None:
            size = len(mus.tracks)

        self.tracks = mus.tracks[: min(size, len(mus.tracks))]
        # self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.window_size = window_size
        self.n_fft = n_fft
        self.n_frames = n_frames
        self.samples_per_track = samples_per_track
        self.total_samples = size * samples_per_track
        self.only_keep = (self.n_frames - 1) * self.hop_length + 1

        self.xs = []
        self.ys = []
        self.x_phases = []
        self.y_phases = []
        self.min_x_amps = []
        self.max_x_amps = []
        self.min_y_amps = []
        self.max_y_amps = []

        self.update_tracks()
        if path is None or not os.path.exists(path):
            self.preprocess()
            self.save_dataset(path)
        else:
            self.xs = torch.load(os.path.join(path, "xs.pt"))
            self.ys = torch.load(os.path.join(path, "ys.pt"))
            self.x_phases = torch.load(os.path.join(path, "x_phases.pt"))
            self.y_phases = torch.load(os.path.join(path, "y_phases.pt"))
            self.min_x_amps = torch.load(os.path.join(path, "min_x_amps.pt"))
            self.max_x_amps = torch.load(os.path.join(path, "max_x_amps.pt"))
            self.min_y_amps = torch.load(os.path.join(path, "min_y_amps.pt"))
            self.max_y_amps = torch.load(os.path.join(path, "max_y_amps.pt"))

    def update_tracks(self):
        for track in self.tracks:
            # track.chunk_duration = self.chunk_duration
            track.sample_rate = self.sample_rate

    def preprocess(self):
        results = []
        total_sampled = 0
        for i, track in enumerate(self.tracks):
            # print(f"Preprocessing track {i+1}/{len(self.tracks)}", end="\r")
            output = preprocess_track(
                track,
                self.only_keep,
                self.n_fft,
                self.hop_length,
                self.window_size,
                self.samples_per_track,
            )
            sampled = 0
            if output is not False:
                sampled = output[0].shape[0]
            total_sampled += sampled
            if i < len(self.tracks) - 1:
                # update samples_per_track
                self.samples_per_track = math.ceil(
                    (self.total_samples - total_sampled) / (len(self.tracks) - i - 1)
                )
            print(
                f"Preprocessing track {i+1}/{len(self.tracks)}: {sampled} samples",
                end="\r",
            )
            results.append(output)

        for result in results:
            if result is False:
                continue
            (
                x_amp,
                y_amp,
                x_phase,
                y_phase,
                min_x_amp,
                max_x_amp,
                min_y_amp,
                max_y_amp,
            ) = result
            self.xs.append(x_amp)
            self.ys.append(y_amp)
            self.x_phases.append(x_phase)
            self.y_phases.append(y_phase)
            self.min_x_amps.append(min_x_amp)
            self.max_x_amps.append(max_x_amp)
            self.min_y_amps.append(min_y_amp)
            self.max_y_amps.append(max_y_amp)

        self.xs = torch.cat(self.xs).float()
        self.ys = torch.cat(self.ys).float()
        self.x_phases = torch.cat(self.x_phases).float()
        self.y_phases = torch.cat(self.y_phases).float()
        self.min_x_amps = torch.cat(self.min_x_amps).float()
        self.max_x_amps = torch.cat(self.max_x_amps).float()
        self.min_y_amps = torch.cat(self.min_y_amps).float()
        self.max_y_amps = torch.cat(self.max_y_amps).float()

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return torch.stack(
            [
                torch.stack([self.xs[idx], self.ys[idx]], dim=0),
                torch.stack([self.x_phases[idx], self.y_phases[idx]], dim=0),
                torch.stack([self.min_x_amps[idx], self.min_y_amps[idx]], dim=0),
                torch.stack([self.max_x_amps[idx], self.max_y_amps[idx]], dim=0),
            ],
            dim=0,
        )

    def save_dataset(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.xs, os.path.join(path, "xs.pt"))
        torch.save(self.ys, os.path.join(path, "ys.pt"))
        torch.save(self.x_phases, os.path.join(path, "x_phases.pt"))
        torch.save(self.y_phases, os.path.join(path, "y_phases.pt"))
        torch.save(self.min_x_amps, os.path.join(path, "min_x_amps.pt"))
        torch.save(self.max_x_amps, os.path.join(path, "max_x_amps.pt"))
        torch.save(self.min_y_amps, os.path.join(path, "min_y_amps.pt"))
        torch.save(self.max_y_amps, os.path.join(path, "max_y_amps.pt"))
