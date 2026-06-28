import unittest
import warnings

import torch

from vocal_sep.audio.spectrogram import reconstruct_waveform, waveform_to_spectrogram
from vocal_sep.config import StftSettings


class TestSpectrogram(unittest.TestCase):
    def test_waveform_to_spectrogram(self):
        settings = StftSettings()
        samples = settings.window_samples
        mix = torch.randn(samples)
        vocal = torch.randn(samples)
        sample = waveform_to_spectrogram(mix, vocal, settings)
        self.assertEqual(sample.mix.ndim, 2)
        self.assertTrue(torch.all(sample.mix >= 0))
        self.assertTrue(torch.all(sample.mix <= 1))

    def test_reconstruct_waveform_uses_analysis_window(self):
        settings = StftSettings()
        samples = settings.window_samples
        mix = torch.randn(samples)
        vocal = torch.randn(samples)
        sample = waveform_to_spectrogram(mix, vocal, settings)
        magnitude = sample.mix
        phase = sample.mix_phase
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            waveform = reconstruct_waveform(magnitude, phase, settings)
        window_warnings = [w for w in caught if "window was not provided" in str(w.message)]
        self.assertEqual(waveform.shape[0], samples)
        self.assertEqual(window_warnings, [])


if __name__ == "__main__":
    unittest.main()
