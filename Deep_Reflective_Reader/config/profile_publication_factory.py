"""Assemble existing lifecycle/storage owners for explicit profile publication."""
from app.document_file_identity import DocumentFileIdentity
from profile.profile_publication import ProfilePublicationGuard, ProfilePublicationConflict


def profile_publication_guard(*, name, source, store, target, profile_config):
    if source is None:
        raise ProfilePublicationConflict("profile_source_evidence_missing")
    return ProfilePublicationGuard(
        name=name, source=source,
        lifecycle=DocumentFileIdentity(source.source_path.parent),
        snapshot=lambda: store.profile_publication_snapshot(target),
        profile_path=profile_config.get_raw_profile_path(),
    )
