"""Model-matched local counting. Asset acquisition belongs to explicit setup only."""

import base64
from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
from threading import Lock
from time import perf_counter

LOGGER = logging.getLogger(__name__)
RECIPE_VERSION = "offline_o200k_v1"


@dataclass(frozen=True)
class EncodingAsset:
    name: str
    filename: str
    url: str
    sha256: str
    size: int


_O200K = EncodingAsset(
    "o200k_base", "o200k_base.tiktoken",
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d", 3613922,
)
SUPPORTED_MODELS = ("gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini")


def encoding_asset_for_model(model: str) -> EncodingAsset | None:
    """Single manifest for both the runtime counter and setup tooling."""
    return _O200K if model in SUPPORTED_MODELS else None


# Pinned o200k_base recipe from tiktoken_ext.openai_public, using Encoding's public
# constructor instead of registry constructors that can download missing assets.
_PATTERN = "|".join([
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n/]*", r"\s*[\r\n]+", r"\s+(?!\S)", r"\s+",
])


class _Unavailable(Exception):
    pass


class OfflineTokenCounter:
    def __init__(self, asset_dir: str | Path | None = None):
        # Construction reads configuration only; assets are verified on first use.
        self.asset_dir = Path(asset_dir if asset_dir is not None else
                              os.environ.get("DEEP_READER_TOKENIZER_ASSET_DIR") or
                              Path(__file__).with_name("tokenizer_assets"))
        self._encodings = {}
        self._lock = Lock()

    def identity(self, model: str) -> str | None:
        asset = encoding_asset_for_model(model)
        if asset is None:
            return None
        try:
            import tiktoken
            return f"{RECIPE_VERSION}:{asset.name}:{asset.sha256}:tiktoken-{tiktoken.__version__}"
        except ImportError:
            return None

    def _encoding(self, asset):
        with self._lock:
            if asset.name in self._encodings:
                return self._encodings[asset.name], "memory"
            started = perf_counter()
            status = "asset_unavailable"
            try:
                path = self.asset_dir / asset.filename
                if not path.is_file():
                    raise _Unavailable(status)
                with path.open("rb") as stream:
                    data = stream.read(asset.size + 1)
                if len(data) != asset.size or hashlib.sha256(data).hexdigest() != asset.sha256:
                    status = "asset_corrupt"
                    raise _Unavailable(status)
                status = "verified"
            except OSError:
                raise _Unavailable(status) from None
            finally:
                LOGGER.info("tokenizer stage=asset_verify encoding=%s recipe=%s status=%s elapsed_ms=%.1f",
                            asset.name, RECIPE_VERSION, status, (perf_counter() - started) * 1000)
            started = perf_counter()
            status = "initialization_failed"
            try:
                import tiktoken
                ranks = {base64.b64decode(token, validate=True): int(rank)
                         for token, rank in (line.split() for line in data.splitlines() if line)}
                encoding = tiktoken.Encoding(name=asset.name, pat_str=_PATTERN,
                    mergeable_ranks=ranks, special_tokens={"<|endoftext|>": 199999, "<|endofprompt|>": 200018})
                self._encodings[asset.name] = encoding
                status = "ready"
                return encoding, "cold"
            finally:
                LOGGER.info("tokenizer stage=initialize encoding=%s recipe=%s status=%s elapsed_ms=%.1f",
                            asset.name, RECIPE_VERSION, status, (perf_counter() - started) * 1000)

    def count(self, model: str, text: str) -> int | None:
        started = perf_counter()
        asset = encoding_asset_for_model(model)
        status, cache, count = "unsupported_model", "none", None
        try:
            if asset is None:
                return None
            status = "tokenizer_unavailable"
            import tiktoken
            # Pure mapping lookup, never encoding_for_model/get_encoding.
            if tiktoken.encoding_name_for_model(model) != asset.name:
                status = "encoding_mismatch"
                return None
            status = "count_unavailable"
            encoding, cache = self._encoding(asset)
            count = len(encoding.encode(text, disallowed_special=()))
            status = "ready"
            return count
        except _Unavailable as error:
            status = str(error)
            return None
        except Exception:
            return None
        finally:
            LOGGER.info("tokenizer stage=count model=%s encoding=%s recipe=%s status=%s cache=%s chars=%s tokens=%s elapsed_ms=%.1f",
                        model if asset else "unsupported", asset.name if asset else "none", RECIPE_VERSION,
                        status, cache, len(text), count, (perf_counter() - started) * 1000)


TOKEN_COUNTER = OfflineTokenCounter()
