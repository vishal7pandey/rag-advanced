from __future__ import annotations
from typing import Tuple

# Try to use tiktoken if available for accurate OpenAI token counts; fallback to heuristic
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

_MODEL_DEFAULT = "gpt-4o-mini"


def _encode_len(model: str, text: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        # Heuristic: ~4 chars per token
        return max(1, int(len(text) / 4))
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")  # type: ignore[attr-defined]
    try:
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


def estimate_tokens(model: str | None, prompt_text: str, completion_text: str) -> Tuple[int, int]:
    """Estimate prompt and completion tokens for cost calculation.
    Uses tiktoken when available; otherwise falls back to a simple heuristic.
    """
    m = model or _MODEL_DEFAULT
    pt = _encode_len(m, prompt_text or "")
    ct = _encode_len(m, completion_text or "")
    return pt, ct
