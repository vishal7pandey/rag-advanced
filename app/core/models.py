from __future__ import annotations
from typing import Optional

# Optional dependency: Streamlit for cross-rerun resource caching
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    st = None  # type: ignore

# Optional deps: model libraries
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    SentenceTransformer = None  # type: ignore

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    CrossEncoder = None  # type: ignore

# Fallback cache if Streamlit is unavailable
_ST_LOCAL: dict[tuple[str, Optional[str]], object] = {}
_CE_LOCAL: dict[tuple[str, str], object] = {}


def _cache_resource(func):
    """Use st.cache_resource if available, else a no-op wrapper with local dicts."""
    if st is None:
        return func
    return st.cache_resource(func)


@_cache_resource
def get_sentence_transformer(model_name: str, device: Optional[str] = None):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")
    # SentenceTransformer handles device internally via .to(); pass device if provided later by caller
    return SentenceTransformer(model_name)


@_cache_resource
def get_cross_encoder(model_name: str, device: str = "cpu"):
    if CrossEncoder is None:
        return None
    return CrossEncoder(model_name, device=device)
