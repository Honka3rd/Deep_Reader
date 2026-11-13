from pathlib import Path
import os

from shared.document_name import document_name


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


def resolve_raw_source_path(base_dir: Path, name: str, suffix: str) -> Path:
    """Validate containment while retaining the logical name for later verification."""
    name = document_name(name)
    filename = name if name.lower().endswith(suffix) else name + suffix
    root = base_dir.resolve()
    logical_path = root / filename
    if not logical_path.resolve().is_relative_to(root):
        raise ValueError("source_identity_invalid")
    return logical_path
