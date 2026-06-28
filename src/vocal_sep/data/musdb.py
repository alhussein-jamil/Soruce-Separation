"""MUSDB18 download and database access."""

import zipfile
from pathlib import Path

import musdb
import requests

from vocal_sep.log import download_progress, get_logger
from vocal_sep.paths import MUSDB_DIR

logger = get_logger(__name__)

MUSDB18_URL = "https://zenodo.org/records/1117372/files/musdb18.zip?download=1"
ARCHIVE_PATH = MUSDB_DIR / "musdb18.zip"


def _download(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    logger.info("Downloading MUSDB18 from [bold blue]%s[/bold blue]", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with download_progress() as progress:
        task = progress.add_task("MUSDB18", total=total or None)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
                    progress.update(task, advance=len(chunk))

    logger.info(
        "Download complete → [bold]%s[/bold] (%.1f GB)",
        destination,
        destination.stat().st_size / 1e9,
    )


def _find_dataset_root(base: Path) -> Path | None:
    if (base / "train").is_dir() or (base / "test").is_dir():
        return base
    for candidate in base.iterdir():
        if candidate.is_dir() and ((candidate / "train").is_dir() or (candidate / "test").is_dir()):
            return candidate
    return None


def ensure_musdb(root: Path = MUSDB_DIR) -> musdb.DB:
    root.mkdir(parents=True, exist_ok=True)
    dataset_root = _find_dataset_root(root)

    if dataset_root is None:
        logger.warning("MUSDB18 not found under [bold]%s[/bold] — fetching dataset", root)
        if not ARCHIVE_PATH.exists():
            _download(MUSDB18_URL, ARCHIVE_PATH)
        else:
            logger.info("Using existing archive [bold]%s[/bold]", ARCHIVE_PATH)
        logger.info("Extracting archive into [bold]%s[/bold] …", root)
        with zipfile.ZipFile(ARCHIVE_PATH, "r") as zip_file:
            zip_file.extractall(root)
        dataset_root = _find_dataset_root(root)
        if dataset_root is None:
            raise RuntimeError(f"MUSDB18 archive did not contain train/test folders under {root}")
        logger.info("Extraction complete")
    else:
        logger.info("Found MUSDB18 at [bold green]%s[/bold green]", dataset_root)

    db = musdb.DB(root=str(dataset_root))
    logger.info(
        "MUSDB ready — [bold]%d[/bold] tracks at [bold green]%s[/bold green]",
        len(db.tracks),
        dataset_root,
    )
    if len(db.tracks) == 0:
        raise RuntimeError(f"No MUSDB tracks found under {dataset_root}")
    return db
