"""Lossless storage names and read-only, unambiguous legacy alias resolution."""
from pathlib import Path
from typing import Callable
from shared.document_name import document_name


class StorageNamespaceHelper:
    DEFAULT_EXTENSIONS = ('.pdf', '.txt')
    DEFAULT_NAMESPACE = 'default'

    @classmethod
    def normalize_namespace(cls, namespace, *, known_extensions=None, fallback_namespace=None):
        # A source suffix is part of its identity, not a decoration.
        return document_name(namespace.strip() or fallback_namespace or cls.DEFAULT_NAMESPACE)

    @classmethod
    def stem(cls, name):
        return name[:-len(Path(name).suffix)] if Path(name).suffix.lower() in cls.DEFAULT_EXTENSIONS else name

    @classmethod
    def raw_names(cls, root, name):
        stem = cls.stem(name)
        if not Path(root).exists():
            return set()
        return {p.name for p in Path(root).iterdir()
                if p.is_file() and p.suffix.lower() in cls.DEFAULT_EXTENSIONS and p.stem == stem}

    @classmethod
    def destination_conflicts(cls, name: str, *, raw_root: Path,
                              contains_document: Callable[[str], bool]) -> bool:
        """Pure occupancy check for a new upload/rename, never a receipt replay.

        Exact repository metadata is essential: resolving a missing qualified
        name would miss an incompatible bare legacy owner of the same stem.
        Distinct qualified sources can coexist when no such owner exists.
        """
        name = cls.normalize_namespace(name)
        stem = cls.stem(name)
        path = cls.checked_path(raw_root, Path(raw_root)/name)
        if path.exists() or path.is_symlink():
            return True
        if any(contains_document(candidate) for candidate in dict.fromkeys((name, stem))):
            return True
        if name == stem and Path(raw_root).exists():
            # Bare targets must not acquire an already occupied raw family.
            return any(p.stem == name and p.suffix.lower() in cls.DEFAULT_EXTENSIONS
                       for p in Path(raw_root).iterdir())
        return False

    @classmethod
    def resolve(cls, name, *, existing, raw_names=()):
        """Exact identities win; only one compatible owner may supply an alias.

        An ambiguous legacy stem is deliberately unavailable until an explicit
        repair/reimport establishes ownership. Reads never migrate artifacts.
        """
        name = cls.normalize_namespace(name)
        stem = cls.stem(name)
        existing = {n for n in existing if cls.stem(n) == stem}
        raw_names = set(raw_names)
        if name == stem and (len(raw_names) > 1 or len(existing - {stem}) > 1):
            raise ValueError('source_ambiguous')
        if name in existing:
            return name
        if name != stem:
            # No evidence of the requested source: do not adopt another format.
            if raw_names != {name}:
                return name
            if existing == {stem}:
                return stem
        return name

    @staticmethod
    def checked_path(root, path):
        try:
            resolved, base = Path(path).resolve(), Path(root).resolve()
        except (OSError, RuntimeError) as error:
            raise ValueError('invalid_document_path') from error
        if resolved == base or not resolved.is_relative_to(base):
            raise ValueError('invalid_document_path')
        return Path(path)

    @classmethod
    def file_identity(cls, name, root, *, suffix='', raw_root=None):
        name = cls.normalize_namespace(name)
        root = Path(root)
        existing = set()
        if root.exists():
            for path in root.iterdir():
                if suffix and not path.name.endswith(suffix):
                    continue
                candidate = path.name[:-len(suffix)] if suffix else path.name
                if cls.stem(candidate) == cls.stem(name):
                    cls.checked_path(root, path)
                    existing.add(candidate)
        return cls.resolve(name, existing=existing,
                           raw_names=cls.raw_names(raw_root or root.parent/'raw', name))
