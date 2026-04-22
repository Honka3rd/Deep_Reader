class StorageNamespaceHelper:
    """Shared helpers for normalizing storage namespaces."""

    DEFAULT_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")
    DEFAULT_NAMESPACE: str = "default"

    @classmethod
    def normalize_namespace(
        cls,
        namespace: str,
        *,
        known_extensions: tuple[str, ...] | None = None,
        fallback_namespace: str | None = None,
    ) -> str:
        """Normalize namespace by trimming and stripping known file extensions."""
        extensions = known_extensions or cls.DEFAULT_EXTENSIONS
        fallback = fallback_namespace or cls.DEFAULT_NAMESPACE

        trimmed = namespace.strip()
        lowered = trimmed.lower()
        for extension in extensions:
            if lowered.endswith(extension):
                trimmed = trimmed[: -len(extension)]
                break
        trimmed = trimmed.strip()
        return trimmed or fallback
