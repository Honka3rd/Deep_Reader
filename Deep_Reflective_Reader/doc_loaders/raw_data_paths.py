from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def resolve_raw_data_dir(base_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the default raw-data directory independent of process cwd."""
    if base_dir is None:
        return DEFAULT_RAW_DATA_DIR

    path = Path(base_dir)
    if path.is_absolute():
        return path

    if path == Path("data/raw") or path == Path("data") / "raw":
        return DEFAULT_RAW_DATA_DIR

    return path
