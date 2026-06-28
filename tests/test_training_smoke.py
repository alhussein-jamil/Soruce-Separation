"""Synthetic end-to-end training smoke test."""

import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from vocal_sep.config import StftSettings, TrainSettings
from vocal_sep.data.dataset import collate_batch
from vocal_sep.training.trainer import Trainer


class SyntheticDataset(Dataset):
    def __init__(self, num_samples: int):
        self.items = []
        for _ in range(num_samples):
            self.items.append(
                {
                    "mix": torch.rand(1, 512, 128),
                    "vocal": torch.rand(1, 512, 128),
                    "mix_phase": torch.rand(512, 128),
                    "vocal_phase": torch.rand(512, 128),
                    "mix_min": torch.rand(512, 128) * 0.1,
                    "mix_max": torch.rand(512, 128) * 0.5 + 0.2,
                    "vocal_min": torch.rand(512, 128) * 0.1,
                    "vocal_max": torch.rand(512, 128) * 0.5 + 0.2,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TestTrainingSmoke(unittest.TestCase):
    def test_one_epoch(self):
        settings = TrainSettings(
            num_samples=8,
            batch_size=2,
            epochs=1,
            num_workers=0,
            pin_memory=False,
            lr_scheduler="none",
            stft=StftSettings(),
        )
        dataset = SyntheticDataset(8)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_batch,
        )
        trainer = Trainer(settings, device="cpu")
        trainer.fit(loader, loader)
        self.assertEqual(len(trainer.train_losses), 1)
        self.assertEqual(len(trainer.val_losses), 1)

    def test_early_stopping(self):
        settings = TrainSettings(
            num_samples=8,
            batch_size=2,
            epochs=10,
            patience=1,
            min_delta=1e9,
            num_workers=0,
            pin_memory=False,
            lr_scheduler="none",
            stft=StftSettings(),
        )
        dataset = SyntheticDataset(8)
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=collate_batch,
        )
        trainer = Trainer(settings, device="cpu")
        trainer.fit(loader, loader)
        self.assertTrue(trainer.stopped_early)
        self.assertEqual(len(trainer.val_losses), 2)


if __name__ == "__main__":
    unittest.main()
