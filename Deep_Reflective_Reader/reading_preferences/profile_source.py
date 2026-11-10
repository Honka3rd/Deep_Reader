"""Read existing profile language through a server-bound loader; never build it."""

from dataclasses import dataclass
import logging
from time import perf_counter
from typing import Callable, Protocol

from reading_preferences.models import PreferenceError, WorkspaceScope, language


logger = logging.getLogger(__name__)


class SavedProfile(Protocol):
    @property
    def document_language_code(self) -> str: ...


@dataclass(frozen=True, slots=True)
class SourceLanguage:
    code: str | None
    origin: str  # override or profile
    problem: str | None = None


class ExistingProfileLanguageReader:
    """Composition supplies the existing profile store's read-only load function.

    The loader receives exact server scope/document identity and returns a saved
    profile or None. Missing/corrupt/unreadable artifacts remain distinct. Other
    exceptions propagate; no broad fallback hides database/programming failures.
    This reader does not create storage config, detect language or repair data.
    """

    def __init__(self, load: Callable[[WorkspaceScope, str], SavedProfile | None]):
        self._load = load

    def read(self, scope: WorkspaceScope, document_id: str) -> SourceLanguage:
        started = perf_counter()
        outcome = "failed"
        try:
            result = self._read(scope, document_id)
            outcome = result.problem or "available"
            return result
        finally:
            logger.info("reading_preferences operation=profile_language_read outcome=%s duration_ms=%.2f",
                        outcome, (perf_counter() - started) * 1000)

    def _read(self, scope: WorkspaceScope, document_id: str) -> SourceLanguage:
        try:
            profile = self._load(scope, document_id)
            if profile is None:
                return SourceLanguage(None, "profile", "profile_missing")
            code = profile.document_language_code
        except FileNotFoundError:
            return SourceLanguage(None, "profile", "profile_missing")
        except (ValueError, TypeError, KeyError, AttributeError):
            # Existing profile decoding can raise shape/attribute errors for
            # non-object JSON or incorrectly typed document_language fields.
            return SourceLanguage(None, "profile", "profile_invalid")
        except OSError:
            return SourceLanguage(None, "profile", "profile_unavailable")
        if code is None or code == "unknown":
            return SourceLanguage(None, "profile", "source_language_required")
        try:
            language(code)
        except PreferenceError:
            return SourceLanguage(None, "profile", "source_language_invalid")
        return SourceLanguage(code, "profile")
