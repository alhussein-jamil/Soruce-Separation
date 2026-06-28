"""Memmap dataset shape checks."""

import pickle

from vocal_sep.data.dataset import MemmapVocalDataset, collate_batch


def test_batch_shapes(memmap_dataset):
    item = memmap_dataset[0]
    assert item["mix"].shape == (1, 512, 128)
    assert item["vocal"].shape == (1, 512, 128)

    batch = collate_batch([memmap_dataset[i] for i in range(4)])
    assert batch["mix"].shape == (4, 1, 512, 128)
    assert batch["vocal"].shape == (4, 1, 512, 128)


def test_pickle_does_not_materialize_full_array(memmap_cache_dir):
    dataset = MemmapVocalDataset(memmap_cache_dir)
    payload = pickle.dumps(dataset)
    assert len(payload) < 100_000, f"pickle size {len(payload)} bytes"
