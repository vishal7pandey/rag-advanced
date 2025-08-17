from __future__ import annotations
import hashlib
from pathlib import Path
import time
from typing import List, Tuple
import os
import random
from threading import Lock

import numpy as np
from dotenv import load_dotenv
from app.core.utils import get_logger

try:
    from openai import OpenAI  # type: ignore

    try:
        # v1 SDK exception types (optional)
        from openai import (
            RateLimitError,
            APITimeoutError,
            APIConnectionError,
            APIStatusError,
            BadRequestError,
        )  # type: ignore
    except Exception:  # pragma: no cover - older/newer SDKs
        RateLimitError = Exception  # type: ignore
        APITimeoutError = Exception  # type: ignore
        APIConnectionError = Exception  # type: ignore
        APIStatusError = Exception  # type: ignore
        BadRequestError = Exception  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
    RateLimitError = Exception  # type: ignore
    APITimeoutError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore
    APIStatusError = Exception  # type: ignore
    BadRequestError = Exception  # type: ignore

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

# Legacy in-memory cache handle kept for backward compatibility in tests.
# The current implementation uses app.core.models.get_sentence_transformer when available.
_ST_MODELS: dict[str, object] = {}

# Cached loader (Streamlit-aware) for SentenceTransformer
try:
    from app.core.models import get_sentence_transformer as _get_st_cached
except Exception:  # pragma: no cover - fallback if models.py missing
    _get_st_cached = None  # type: ignore

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "emb_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Central logger
logger = get_logger()


# Centralized limiters (process-wide)
class _RequestLimiter:
    def __init__(self, rpm: float):
        self.min_spacing = 60.0 / max(1.0, float(rpm))
        self._next = 0.0
        self._lock = Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                wait = max(0.0, self._next - now)
                if wait <= 0:
                    # schedule next slot
                    self._next = max(self._next, now) + self.min_spacing
                    return
            # sleep outside lock
            time.sleep(wait + random.uniform(0, 0.02))


class _TokenBucket:
    def __init__(self, capacity: float, refill_per_sec: float):
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self.tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = Lock()

    def acquire(self, amount: float) -> None:
        # Blocks until enough tokens are available, then consumes them
        amount = float(max(0.0, amount))
        if amount == 0.0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill)
                    self._last = now
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                needed = amount - self.tokens
                wait = needed / max(1e-6, self.refill)
            # Clamp wait to avoid excessively long sleeps under misconfiguration
            time.sleep(min(wait, 5.0) + random.uniform(0, 0.02))


_REQ_LIMITER: _RequestLimiter | None = None
_TPM_LIMITER: _TokenBucket | None = None
_ENCODER = None


def _ensure_limiters() -> tuple[_RequestLimiter, _TokenBucket | None]:
    global _REQ_LIMITER, _TPM_LIMITER, _ENCODER
    # RPM limiter
    if _REQ_LIMITER is None:
        try:
            rpm_cap = float(os.getenv("OPENAI_EMBED_RPM_CAP", "90").strip() or 90)
        except Exception:
            rpm_cap = 90.0
        _REQ_LIMITER = _RequestLimiter(rpm_cap)
    # TPM limiter (optional)
    if _TPM_LIMITER is None:
        tpm_env = os.getenv("OPENAI_EMBED_TPM_CAP", "").strip() or ""
        if tpm_env:
            try:
                tpm_cap = float(tpm_env)
                if tpm_cap <= 0:
                    # Non-positive TPM disables limiter to avoid indefinite waits
                    _TPM_LIMITER = None
                else:
                    _TPM_LIMITER = _TokenBucket(capacity=tpm_cap, refill_per_sec=tpm_cap / 60.0)
            except Exception:
                _TPM_LIMITER = None
    # Token encoder (optional)
    if _ENCODER is None and tiktoken is not None:
        try:
            _ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _ENCODER = None
    return _REQ_LIMITER, _TPM_LIMITER


def _count_tokens(text: str) -> int:
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:
            pass
    # Fallback heuristic: ~4 chars per token
    return max(1, (len(text) // 4) + 1)


def _yield_batches_token_aware(
    items: List[str], max_items: int, per_req_token_budget: int
) -> List[List[str]]:
    if max_items <= 0:
        max_items = 256
    if per_req_token_budget and per_req_token_budget > 0:
        out: List[List[str]] = []
        cur: List[str] = []
        cur_tok = 0
        for t in items:
            t_tok = _count_tokens(t)
            # If adding this item exceeds either limit, flush first (ensure at least one per batch)
            if cur and (len(cur) >= max_items or cur_tok + t_tok > per_req_token_budget):
                out.append(cur)
                cur = []
                cur_tok = 0
            cur.append(t)
            cur_tok += t_tok
            if len(cur) >= max_items:
                out.append(cur)
                cur = []
                cur_tok = 0
        if cur:
            out.append(cur)
        return out
    # Fallback: fixed-size batches
    out2: List[List[str]] = []
    for i in range(0, len(items), max_items):
        out2.append(items[i : i + max_items])
    return out2


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _cache_path(model: str, text: str) -> Path:
    return CACHE_DIR / f"{model}_{_sha(text)}.npy"


def embed_openai(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed texts using OpenAI with caching, token-aware micro-batching, centralized throttling, and robust backoff.

    - On-disk per-text cache to avoid redundant requests
    - Token-aware batching (uses tiktoken when available) with item cap
    - Centralized RPM limiter and optional TPM limiter via env vars
    - Retries only on retriable errors; honors Retry-After with jitter; re-raises last error with details
    """
    if OpenAI is None:
        raise RuntimeError("openai package not available")
    client = OpenAI()

    # Resolve which items are already cached
    cached: List[tuple[int, np.ndarray]] = []
    missing_idx: List[int] = []
    missing_texts: List[str] = []
    for i, t in enumerate(texts):
        cp = _cache_path(model, t)
        if cp.exists():
            try:
                cached.append((i, np.load(cp)))
                continue
            except Exception:
                pass  # fall through to recompute
        missing_idx.append(i)
        missing_texts.append(t)

    # Configurable knobs
    max_items = int(os.getenv("OPENAI_EMBED_MAX_ITEMS", "256").strip() or 256)
    # Per-request token budget (token-aware micro-batching)
    # Default keeps each embeddings.create request well under common org TPM caps.
    # Override via OPENAI_EMBED_PER_REQ_TOKENS if desired.
    try:
        per_req_token_budget = int(
            os.getenv("OPENAI_EMBED_PER_REQ_TOKENS", "20000").strip() or 20000
        )
    except Exception:
        per_req_token_budget = 20000
    per_req_token_budget = max(1024, per_req_token_budget)

    # Ensure limiters/encoder are initialized
    req_limiter, tpm_limiter = _ensure_limiters()

    def _call_with_backoff(batch: List[str], max_retries: int = 8):
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                payload = batch[0] if len(batch) == 1 else batch
                return client.embeddings.create(model=model, input=payload)
            except BadRequestError as e:  # non-retriable (e.g., too-long input)
                raise e
            except RateLimitError as e:  # 429
                last_err = e
                retry_after = 0.0
                try:
                    headers = getattr(getattr(e, "response", None), "headers", {}) or {}
                    retry_after = float(
                        headers.get("retry-after") or headers.get("Retry-After") or 0
                    )
                except Exception:
                    retry_after = 0.0
                sleep_s = max(retry_after, 0.5 * (2**attempt)) + random.uniform(0, 0.25)
                try:
                    logger.warning(
                        "embed.retry",
                        attempt=attempt,
                        sleep_s=round(sleep_s, 3),
                        reason="rate_limit",
                    )
                except Exception:
                    pass
                time.sleep(min(sleep_s, 15.0))
            except APIStatusError as e:  # server-side status
                last_err = e
                status = getattr(e, "status", None)
                if status == 429 or (isinstance(status, int) and 500 <= status < 600):
                    sleep_s = 0.5 * (2**attempt) + random.uniform(0, 0.25)
                    try:
                        logger.warning(
                            "embed.retry",
                            attempt=attempt,
                            sleep_s=round(sleep_s, 3),
                            reason="api_status",
                            status=int(status) if isinstance(status, int) else status,
                        )
                    except Exception:
                        pass
                    time.sleep(min(sleep_s, 15.0))
                else:
                    raise e
            except (APITimeoutError, APIConnectionError) as e:
                last_err = e
                sleep_s = 0.5 * (2**attempt) + random.uniform(0, 0.25)
                try:
                    logger.warning(
                        "embed.retry",
                        attempt=attempt,
                        sleep_s=round(sleep_s, 3),
                        reason="timeout_or_connection",
                    )
                except Exception:
                    pass
                time.sleep(min(sleep_s, 15.0))
        # Preserve original error details
        if last_err is not None:
            raise last_err
        raise RuntimeError("OpenAI embeddings.create failed without exception")

    # Perform requests only for missing texts
    if missing_texts:
        batches = _yield_batches_token_aware(missing_texts, max_items, per_req_token_budget)
        for batch in batches:
            batch_tokens = sum(_count_tokens(t) for t in batch)
            t0 = time.monotonic()
            try:
                logger.info(
                    "embed.batch_start",
                    model=model,
                    items=len(batch),
                    tokens_in=batch_tokens,
                    per_req_token_budget=per_req_token_budget,
                    max_items=max_items,
                )
            except Exception:
                pass
            # centralized RPM limiter
            req_limiter.acquire()
            # optional TPM limiter
            if tpm_limiter is not None:
                tpm_limiter.acquire(batch_tokens)
            resp = _call_with_backoff(batch)
            data = resp.data
            # Save to cache
            for t, d in zip(batch, data):
                arr = np.array(d.embedding, dtype=np.float32)
                np.save(_cache_path(model, t), arr)
            t1 = time.monotonic()
            try:
                logger.info(
                    "embed.batch_end",
                    model=model,
                    items=len(batch),
                    tokens_in=batch_tokens,
                    duration_ms=int((t1 - t0) * 1000),
                )
            except Exception:
                pass

    # Reassemble outputs in input order
    out: List[np.ndarray] = [None] * len(texts)  # type: ignore
    for i, vec in cached:
        out[i] = vec
    # reload from cache for computed ones to be safe
    for i in missing_idx:
        out[i] = np.load(_cache_path(model, texts[i]))
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def get_st_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer instance for the given model name.
    Uses Streamlit cache if available (via app.core.models), else a local instance.
    """
    # 1) Legacy in-memory cache for tests/back-compat
    if model_name in _ST_MODELS:
        return _ST_MODELS[model_name]

    # 2) Prefer local constructor so unit tests can patch app.core.embeddings.SentenceTransformer
    #    without requiring the package to be installed (models.py uses its own import).
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer(model_name)
            _ST_MODELS[model_name] = model
            return model
        except Exception:
            # Fall back to Streamlit-cached loader if local instantiation fails.
            pass

    # 3) Streamlit-cached loader (may raise if sentence-transformers truly unavailable)
    if _get_st_cached is not None:
        return _get_st_cached(model_name)

    # 4) No available loader
    raise RuntimeError("sentence-transformers not available")


def embed_st(texts: List[str], model_name: str = "intfloat/e5-small-v2") -> np.ndarray:
    """Embed texts with SentenceTransformer, using on-disk per-text cache and batching for misses."""
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")
    model = get_st_model(model_name)
    # Split into cached vs missing
    cached_vecs: List[Tuple[int, np.ndarray]] = []
    missing_indices: List[int] = []
    missing_texts: List[str] = []
    for i, t in enumerate(texts):
        cp = _cache_path(model_name, t)
        if cp.exists():
            cached_vecs.append((i, np.load(cp)))
        else:
            missing_indices.append(i)
            missing_texts.append(t)
    # Batch-encode missing
    computed: List[np.ndarray] = []
    if missing_texts:
        # Call encode with scalar string when only one text is missing to match tests
        if len(missing_texts) == 1:
            vs = model.encode(missing_texts[0], normalize_embeddings=True)
            arrs = [np.array(vs, dtype=np.float32)]
        else:
            vs = model.encode(missing_texts, normalize_embeddings=True)
            # Ensure 2D array list
            if isinstance(vs, list):
                arrs = [np.array(v, dtype=np.float32) for v in vs]
            else:
                arrs = [np.array(v, dtype=np.float32) for v in list(vs)]
        for idx, arr in zip(missing_indices, arrs):
            # Save to disk cache
            cp = _cache_path(model_name, texts[idx])
            np.save(cp, arr)
            computed.append(arr)
    # Reassemble in original order
    out: List[np.ndarray] = [None] * len(texts)  # type: ignore
    for i, vec in cached_vecs:
        out[i] = vec
    for i, arr in zip(missing_indices, computed):
        out[i] = arr
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def get_default_embedder(offline: bool, st_model: str | None = None, oa_model: str | None = None):
    if offline:

        def fn(texts: List[str]) -> np.ndarray:
            return embed_st(texts, st_model or "intfloat/e5-small-v2")

        return fn
    else:
        load_dotenv()
        if os.getenv("OPENAI_API_KEY", "").strip() == "":
            # fallback to offline if key missing
            def fn(texts: List[str]) -> np.ndarray:
                return embed_st(texts, st_model or "intfloat/e5-small-v2")

            return fn

        def fn(texts: List[str]) -> np.ndarray:
            return embed_openai(texts, oa_model or "text-embedding-3-small")

        return fn


def effective_embedder_id(
    offline: bool, st_model: str | None = None, oa_model: str | None = None
) -> str:
    """Return a stable identifier for the actual embedder used.
    Examples: "st:intfloat/e5-small-v2" or "openai:text-embedding-3-small".
    Mirrors the decision logic in get_default_embedder().
    """
    if offline:
        return f"st:{st_model or 'intfloat/e5-small-v2'}"
    load_dotenv()
    if os.getenv("OPENAI_API_KEY", "").strip() == "":
        return f"st:{st_model or 'intfloat/e5-small-v2'}"
    return f"openai:{oa_model or 'text-embedding-3-small'}"
