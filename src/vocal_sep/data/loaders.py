"""DataLoader construction."""

from torch.utils.data import DataLoader, Dataset

from vocal_sep.config import TrainSettings
from vocal_sep.data.dataset import collate_batch
from vocal_sep.device import cuda_available
from vocal_sep.log import get_logger

logger = get_logger(__name__)


def build_dataloader(dataset: Dataset, settings: TrainSettings, shuffle: bool) -> DataLoader:
    pin_memory = settings.pin_memory and cuda_available()
    loader_kwargs: dict = {
        "batch_size": settings.batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_batch,
        "pin_memory": pin_memory,
    }
    if settings.num_workers > 0:
        loader_kwargs.update(
            num_workers=settings.num_workers,
            persistent_workers=False,
            prefetch_factor=settings.prefetch_factor,
        )

    split = "train" if shuffle else "val"
    logger.info(
        "DataLoader [bold]%s[/bold] — batches=%d, batch_size=%d, workers=%d, pin_memory=%s",
        split,
        max(1, len(dataset) // settings.batch_size),
        settings.batch_size,
        settings.num_workers,
        pin_memory,
    )
    logger.debug(
        "%s loader kwargs: %s",
        split,
        {k: v for k, v in loader_kwargs.items() if k != "collate_fn"},
    )
    return DataLoader(dataset, **loader_kwargs)
