"""Bounded, request-local PDF evidence reuse and subprocess accounting."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
import logging
import math
import sys
import subprocess
import time
from collections.abc import Callable

from logging_config import request_id_context


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfExecutionLimits:
    text_render_attempts: int = 1024
    text_ocr_attempts: int = 4096
    text_seconds: float = 1800.0
    layout_render_attempts: int = 30
    layout_ocr_attempts: int = 120
    layout_seconds: float = 60.0
    metadata_attempts: int = 4
    cache_entries: int = 512
    cache_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for name, value in vars(self).items():
            if name.endswith("seconds"):
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            else:
                valid = type(value) is int and value >= 0
            if not valid:
                raise ValueError(f"{name} must be finite and non-negative")


class PdfBudgetExceeded(RuntimeError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def in_pdf_pass(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.preparation_scope():
            return method(self, *args, **kwargs)
    return wrapped


def _size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_size(k, seen) + _size(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list, set)):
        size += sum(_size(item, seen) for item in value)
    elif hasattr(value, "__dict__"):
        size += _size(vars(value), seen)
    return size


@dataclass
class PdfExecutionPass:
    limits: PdfExecutionLimits
    clock: Callable[[], float] = time.monotonic
    cache: OrderedDict = field(default_factory=OrderedDict)
    cache_bytes: int = 0
    counters: dict = field(default_factory=dict)
    started: dict = field(default_factory=dict)
    provenance: dict | None = None
    pages: list | None = None
    quality: list = field(default_factory=list)

    def clear_cache(self):
        self.cache.clear()
        self.cache_bytes = 0

    def get(self, key, *, stage, page=-1, purpose="inspection"):
        hit = key is not None and key in self.cache
        logger.info("pdf_evidence_cache request_id=%s stage=%s page=%s purpose=%s outcome=%s",
                    request_id_context.get(), stage, page, purpose,
                    "hit" if hit else "miss" if key is not None else "unknown_identity")
        if not hit:
            return None
        value, size = self.cache.pop(key)
        self.cache[key] = (value, size)
        return deepcopy(value)

    def put(self, key, value):
        if key is None or not self.limits.cache_entries:
            return
        size = _size((key, value))
        if size > self.limits.cache_bytes:
            logger.info("pdf_evidence_cache outcome=not_retained reason=entry_too_large")
            return
        if key in self.cache:
            self.cache_bytes -= self.cache.pop(key)[1]
        while self.cache and (len(self.cache) >= self.limits.cache_entries
                              or self.cache_bytes + size > self.limits.cache_bytes):
            self.cache_bytes -= self.cache.popitem(last=False)[1][1]
        self.cache[key] = (deepcopy(value), size)
        self.cache_bytes += size

    def remaining(self, purpose):
        prefix = "text" if purpose == "text_recovery" else "layout"
        limit = getattr(self.limits, f"{prefix}_seconds")
        now = self.clock()
        return limit - (now - self.started.setdefault(purpose, now))

    def exhausted(self, purpose, stage, page, reason, count):
        logger.warning("pdf_budget_exhausted request_id=%s purpose=%s stage=%s page=%s reason=%s attempts=%s",
                       request_id_context.get(), purpose, stage, page, reason, count)
        return PdfBudgetExceeded(f"{purpose}_{reason}_budget_exhausted")

    def run(self, runner, command, *, purpose, stage, page, timeout):
        prefix = "text" if purpose == "text_recovery" else "layout"
        limit = (self.limits.metadata_attempts if stage == "metadata"
                 else getattr(self.limits, f"{prefix}_{stage}_attempts"))
        count_key = (purpose, stage)
        count = self.counters.get(count_key, 0)
        remaining = self.remaining(purpose)
        reason = "time" if remaining <= 0 else stage if count >= limit else None
        if reason:
            raise self.exhausted(purpose, stage, page, reason, count)
        self.counters[count_key] = count + 1
        started = self.clock()
        outcome = "failed"
        try:
            try:
                result = runner(command, check=False, capture_output=True, text=True,
                                timeout=min(timeout, remaining))
            except subprocess.TimeoutExpired:
                if remaining <= timeout or self.remaining(purpose) <= 0:
                    raise self.exhausted(purpose, stage, page, "time", count + 1) from None
                raise
            if self.remaining(purpose) <= 0:
                raise self.exhausted(purpose, stage, page, "time", count + 1)
            outcome = "completed" if result.returncode == 0 else "nonzero_exit"
            return result
        finally:
            logger.info("pdf_process request_id=%s purpose=%s stage=%s page=%s attempt=%s duration_ms=%.1f outcome=%s",
                        request_id_context.get(), purpose, stage, page, count + 1,
                        (self.clock() - started) * 1000, outcome)
