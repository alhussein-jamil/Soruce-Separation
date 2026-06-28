"""Pytest configuration for CPU CI runners."""

import json
import os

import numpy as np
import pytest
import torch

from vocal_sep.data.dataset import MemmapVocalDataset

os.environ["TORCH_USE_ONEDNN"] = "0"
os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX2")

if not torch.cuda.is_available():
    mkldnn = getattr(torch.backends, "mkldnn", None)
    if mkldnn is not None and hasattr(mkldnn, "set_enabled"):
        mkldnn.set_enabled(False)


@pytest.fixture
def memmap_cache_dir(tmp_path):
    """Minimal on-disk memmap cache for dataset tests."""
    num_samples = 8
    shapes = {
        "mix": (num_samples, 1, 512, 128),
        "vocal": (num_samples, 1, 512, 128),
        "mix_phase": (num_samples, 512, 128),
        "vocal_phase": (num_samples, 512, 128),
        "mix_min": (num_samples, 512, 128),
        "mix_max": (num_samples, 512, 128),
        "vocal_min": (num_samples, 512, 128),
        "vocal_max": (num_samples, 512, 128),
    }
    for name, shape in shapes.items():
        np.lib.format.open_memmap(
            tmp_path / f"{name}.npy",
            mode="w+",
            dtype=np.float16,
            shape=shape,
        )
    (tmp_path / "meta.json").write_text(
        json.dumps({"num_samples": num_samples, "dtype": "float16"})
    )
    return tmp_path


@pytest.fixture
def memmap_dataset(memmap_cache_dir):
    return MemmapVocalDataset(memmap_cache_dir)
