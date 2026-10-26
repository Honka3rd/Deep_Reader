"""Narrow production composition for cleanup; no planning or storage policy."""

from app.standalone_cleanup import StandaloneCleanupService


def build_cleanup_service(*, namespace, coordinator, provider):
    repository = coordinator.document_artifact_repository
    # Execution requires the SAME cross-process lease as enhancement. Do not
    # silently downgrade an unsupported storage backend to an in-process lock.
    if not all(callable(getattr(repository, name, None)) for name in (
            "load_enhancement_snapshot", "enhancement_lease", "commit_cleanup", "lookup_cleanup")):
        return None
    return StandaloneCleanupService(namespace=namespace, repository=repository,
        read_input=coordinator.read_cleanup_input, provider=provider,
        lease=coordinator.enhancement_lease, layout_ready=coordinator.cleanup_layout_ready,
        prepare_layout=coordinator.prepare_cleanup_layout)
