"""Source-bound enhancement consent. No extraction or persistence side effects."""
from dataclasses import dataclass
from contextlib import contextmanager
import base64
import hashlib
import hmac
import json
import secrets
import time
import logging
from typing import Callable

from document_preparation.accepted_source_snapshot import AcceptedSourceSnapshot, SourceNotReady
from document_structure.enhancement_capacity import capacity_projection, require_capacity_identity

LOGGER = logging.getLogger(__name__)


class EnhancementConflict(ValueError):
    def __init__(self, reason: str, status_code: int = 409, capacity=None):
        self.reason = reason
        self.status_code = status_code
        self.capacity = capacity
        super().__init__(reason)


@dataclass(frozen=True)
class EnhancementState:
    source: AcceptedSourceSnapshot
    interaction_revision: str
    interaction_count: int | None = None
    includes_user_notes: bool | None = None


class EnhancementPreflightService:
    """Reader must use a consistent storage snapshot and trusted source identity.

    Tokens are process-local and expire after five minutes. Restart/another worker
    invalidates them safely. Readiness is not execution availability; commit guards
    remain required. Unknown revisions cannot be replaced by counts or timestamps.
    """
    def __init__(self, reader: Callable[[str], EnhancementState] | None = None,
                 *, clock=time.time, secret: bytes | None = None, lease=None, availability=None,
                 capacity_reader=None):
        self.reader = reader
        self.clock = clock
        self.secret = secret or secrets.token_bytes(32)
        self.lease = lease
        self.availability = availability
        self.capacity_reader = capacity_reader
        from app.enhancement_operations import EnhancementOperations
        self.operations = EnhancementOperations(clock=clock)

    def _capacity(self, state, toc_hint, enhancement_intent):
        try:
            if self.capacity_reader is None:
                raise ValueError("capacity reader unavailable")
            result = self.capacity_reader(state, toc_hint, enhancement_intent)
            if not isinstance(result, dict) or not isinstance(result.get("capacity"), dict):
                raise ValueError("invalid capacity report")
            if result["capacity"].get("status") not in {"fits", "exceeded", "unavailable", "resource_limited"}:
                raise ValueError("invalid capacity status")
            if result["capacity"]["status"] == "fits":
                identity = result.get("fingerprint")
                if not isinstance(identity, str) or len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
                    raise ValueError("missing capacity identity")
            return result
        except Exception:
            LOGGER.warning("enhancement_capacity_unavailable")
            return {"capacity": capacity_projection(reason="capacity_unavailable",
                    source_chars=state.source.text_length), "fingerprint": None}

    def execution_unavailable_reason(self, doc_name):
        if self.availability is None:
            return "enhancement_execution_unavailable"
        return self.availability(doc_name)

    def _state(self, doc_name: str) -> EnhancementState:
        if self.reader is None:
            raise EnhancementConflict("enhancement_storage_guards_unavailable", 503)
        state = self.reader(doc_name)
        if not state.interaction_revision:
            raise EnhancementConflict("interaction_revision_unavailable", 503)
        return state

    @staticmethod
    def _binding(state, doc_name, toc_hint, enhancement_intent, capacity_fingerprint):
        source = state.source
        value = ["enhancement-consent-v2", doc_name, source.namespace,
                 source.document_id, source.structure_version,
                 source.extraction_identity.to_dict(), source.text_sha256,
                 state.interaction_revision, "llm_enhanced", str(toc_hint),
                 str(enhancement_intent), source.language, capacity_fingerprint]
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def preflight(self, *, doc_name, toc_hint="auto", enhancement_intent="restore_source"):
        result = dict(ready=False, reason=None, structure_version=None,
                      interaction_count=None, includes_user_notes=None,
                      cleanup_scope="all_document_interactions_and_local_drafts",
                      cleanup_condition="structure_changed_only",
                      confirmation_token=None, expires_at=None, execution_available=False, capacity=None)
        try:
            state = self._state(doc_name)
        except (EnhancementConflict, SourceNotReady) as error:
            return {**result, "reason": error.reason}
        result.update(ready=True, structure_version=state.source.structure_version,
                      interaction_count=state.interaction_count, includes_user_notes=state.includes_user_notes)
        unavailable = self.execution_unavailable_reason(doc_name)
        if unavailable:
            return {**result, "reason": unavailable}
        report = self._capacity(state, toc_hint, enhancement_intent)
        result["capacity"] = report["capacity"]
        if report["capacity"]["status"] != "fits":
            return {**result, "reason": report["capacity"]["reason_code"]}
        expires = int(self.clock()) + 300
        binding = self._binding(state, doc_name, toc_hint, enhancement_intent, report["fingerprint"])
        body = f"{expires}.{binding}.{secrets.token_hex(16)}"
        signature = hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()
        token = base64.urlsafe_b64encode(f"{body}.{signature}".encode()).decode()
        try:
            self.operations.register(dict(doc_name=doc_name, toc_hint=toc_hint,
                enhancement_intent=enhancement_intent, confirmation_token=token))
        except EnhancementConflict as error:
            return {**result, "reason": error.reason}
        return dict(result, ready=True, execution_available=unavailable is None, reason=unavailable,
                    structure_version=state.source.structure_version,
                    interaction_count=state.interaction_count,
                    includes_user_notes=state.includes_user_notes,
                    confirmation_token=token, expires_at=expires)

    def verify_consent(self, *, doc_name, discard_interactions=False, confirmation_token=None,
                 toc_hint="auto", enhancement_intent="restore_source"):
        if discard_interactions is not True or not confirmation_token:
            raise EnhancementConflict("confirmation_required")
        try:
            if len(confirmation_token) > 1024:
                raise ValueError("oversized token")
            body, signature = base64.b64decode(
                confirmation_token, altchars=b"-_", validate=True).decode().rsplit(".", 1)
            expected = hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise ValueError("invalid signature")
            expires, binding, nonce = body.split(".")
            expires = int(expires)
        except (ValueError, UnicodeError, TypeError) as error:
            raise EnhancementConflict("confirmation_invalid") from error
        if self.clock() >= expires:
            raise EnhancementConflict("confirmation_expired")
        return binding

    def _validated(self, **request):
        binding = self.verify_consent(**request)
        doc_name = request["doc_name"]
        toc_hint = request.get("toc_hint", "auto")
        enhancement_intent = request.get("enhancement_intent", "restore_source")
        try:
            state = self._state(doc_name)
        except SourceNotReady as error:
            raise EnhancementConflict(error.reason) from error
        unavailable = self.execution_unavailable_reason(doc_name)
        if unavailable:
            raise EnhancementConflict(unavailable, 503)
        report = self._capacity(state, toc_hint, enhancement_intent)
        capacity = report["capacity"]
        if capacity["status"] != "fits":
            raise EnhancementConflict(capacity["reason_code"],
                413 if capacity["status"] in {"exceeded", "resource_limited"} else 503, capacity)
        if not hmac.compare_digest(binding, self._binding(state, doc_name, toc_hint, enhancement_intent, report["fingerprint"])):
            raise EnhancementConflict("confirmation_stale")
        return state, report["fingerprint"]

    def validate(self, **request):
        return self._validated(**request)[0]

    @contextmanager
    def authorize_execution(self, **request):
        # Verify signature/expiry cheaply, then re-read and budget under the lease.
        self.verify_consent(**request)
        if self.lease is None:
            raise EnhancementConflict("enhancement_execution_unavailable", 503)
        try:
            with self.lease(request["doc_name"]):
                state, fingerprint = self._validated(**request)
                with require_capacity_identity(fingerprint):
                    yield state
        except BlockingIOError as error:
            raise EnhancementConflict("enhancement_in_progress") from error


enhancement_preflight_service = EnhancementPreflightService()
