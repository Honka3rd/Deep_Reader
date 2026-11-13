"""Recoverable publication of the index/records/fingerprint as one local unit."""
from contextlib import contextmanager
from copy import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

FILES = ("index.faiss", "records.json", "meta.json")
JOURNAL = ".index-publication.json"


def at_directory(config, directory):
    result = copy(config)
    result.base_dir = Path(directory)
    result.index_path, result.records_path, result.meta_path = (
        result.base_dir / name for name in FILES)
    return result


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_file(path):
    with open(path, "rb") as stream:
        os.fsync(stream.fileno())


def write_journal(config, directory):
    pending = directory / "journal.json"
    pending.write_text(json.dumps({"transaction": directory.name}))
    sync_file(pending)
    os.replace(pending, config.base_dir / JOURNAL)
    sync_directory(config.base_dir)


@contextmanager
def locked(config, *, write=False):
    if write:
        config.base_dir.mkdir(parents=True, exist_ok=True)
    # Directory locks need no read-time lock-file creation. Publishers must not
    # replace the directory itself; lifecycle renames retain its inode.
    fd = os.open(config.base_dir, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def previous_config(config):
    journal = config.base_dir / JOURNAL
    if not journal.exists():
        return config
    data = json.loads(journal.read_text())
    name = data["transaction"]
    if not isinstance(name, str) or Path(name).name != name or not name.startswith(".index-candidate-"):
        raise ValueError("invalid_index_publication_journal")
    directory = config.base_dir / name / "previous"
    if not directory.is_dir() or directory.resolve().parent.parent != config.base_dir.resolve():
        raise ValueError("invalid_index_publication_backup")
    return at_directory(config, directory)


def restore_locked(config):
    previous = previous_config(config)
    if previous is config:
        return
    for name in FILES:
        old, target = previous.base_dir / name, config.base_dir / name
        if old.exists():
            # Keep the backup until the whole rollback is durable; repeatable
            # even when interrupted during rollback itself.
            pending = config.base_dir / (name + ".restore")
            shutil.copyfile(old, pending)
            sync_file(pending)
            os.replace(pending, target)
        else:
            target.unlink(missing_ok=True)
    sync_directory(config.base_dir)
    (config.base_dir / JOURNAL).unlink()
    sync_directory(config.base_dir)
    shutil.rmtree(previous.base_dir.parent)


@contextmanager
def readable(config):
    """Resolve a coherent committed view, including interrupted publication."""
    with locked(config):
        yield previous_config(config)


def recover(config):
    """Explicit recovery only; reads never repair files."""
    with locked(config, write=True):
        restore_locked(config)


def publish(store, bundle, config, *, text, fingerprint, verify_source):
    """Candidate work precedes active-file changes; journal removal commits."""
    config.base_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".index-candidate-", dir=config.base_dir))
    candidate = at_directory(config, directory / "candidate")
    journal_owned = False
    try:
        store.save(bundle, candidate)
        fingerprint.save(text, candidate.get_raw_meta_path())
        store.load_validated(candidate, text, fingerprint)
        for name in FILES:
            sync_file(candidate.base_dir / name)
        sync_directory(candidate.base_dir)
        with locked(config, write=True):
            restore_locked(config)
            verify_source()
            previous = directory / "previous"
            previous.mkdir()
            for name in FILES:
                active = config.base_dir / name
                if active.exists():
                    shutil.copyfile(active, previous / name)
                    sync_file(previous / name)
            sync_directory(previous)
            sync_directory(directory)
            try:
                # Keep backups even if acknowledgement of journal persistence
                # fails after its rename. No active-file write precedes this.
                journal_owned = True
                write_journal(config, directory)
                for name in FILES:
                    os.replace(candidate.base_dir / name, config.base_dir / name)
                sync_directory(config.base_dir)
                verify_source()
                (config.base_dir / JOURNAL).unlink()
                sync_directory(config.base_dir)
                journal_owned = False
            except Exception:
                if not (config.base_dir / JOURNAL).exists():
                    write_journal(config, directory)
                restore_locked(config)
                journal_owned = False
                raise
    finally:
        # BaseException models termination: a pending journal owns the backup.
        if not journal_owned:
            shutil.rmtree(directory, ignore_errors=True)
