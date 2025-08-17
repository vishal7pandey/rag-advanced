from __future__ import annotations
from typing import List, Tuple
import numpy as np
import json

from app.core.types import RetrievedDoc

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None  # type: ignore

# Prefer Streamlit-cached loader if available
try:
    from app.core.models import get_cross_encoder as _get_ce_cached
except Exception:  # pragma: no cover - optional at runtime
    _get_ce_cached = None  # type: ignore

# Lazy import to avoid hard dependency at import time
try:
    from app.core.utils import load_config
except Exception:  # pragma: no cover - defensive
    load_config = None  # type: ignore

_model = None
_model_name = None

_CE_MODELS: dict[str, object] = {}


def _get_ce_local(model_name: str, device: str = "cpu"):
    global _CE_MODELS
    if CrossEncoder is None:
        return None
    key = f"{model_name}::{device}"
    if key not in _CE_MODELS:
        try:
            _CE_MODELS[key] = CrossEncoder(model_name, device=device)
        except Exception:
            return None
    return _CE_MODELS[key]


def get_cross_encoder(model_name: str, device: str = "cpu"):
    """Return a cached CrossEncoder instance (Streamlit cache if available)."""
    if _get_ce_cached is not None:
        try:
            return _get_ce_cached(model_name, device=device)
        except Exception:
            pass
    return _get_ce_local(model_name, device=device)


def _get_rerank_config() -> tuple[str, str, int]:
    """Return (model_name, device, batch_size) from config or safe defaults."""
    # Defaults chosen for Phase 1 upgrade
    model_name = "BAAI/bge-reranker-v2-m3"
    device = "cpu"
    batch_size = 16
    try:
        if load_config is not None:
            cfg = load_config()
            r = cfg.get("retrieval", {})
            model_name = r.get("rerank_model", model_name)
            device = r.get("rerank_device", device)
            bs = r.get("rerank_batch_size", batch_size)
            try:
                batch_size = int(bs)
            except Exception:
                pass
    except Exception:
        # If config fails to load, keep defaults
        pass
    return model_name, device, batch_size


def _ensure_model(target_model: str, device: str):
    """Backwards-compatible wrapper using singleton cache."""
    model = get_cross_encoder(target_model, device=device)
    return model


def rerank_bge(
    query: str, docs: List[RetrievedDoc], model_name: str | None = None
) -> List[RetrievedDoc]:
    """Fully rerank docs using a cross-encoder. Falls back to original order on failure.

    Note: This requires sentence-transformers to be installed (optional dependency).
    """
    if not docs:
        return docs
    if CrossEncoder is None:
        # Graceful fallback when sentence-transformers not available
        return docs
    cfg_model, device, batch_size = _get_rerank_config()
    use_model = model_name or cfg_model
    model = _ensure_model(use_model, device)
    if model is None:
        return docs
    pairs = [(query, d.chunk.text) for d in docs]
    try:
        scores = model.predict(pairs, batch_size=batch_size)  # type: ignore[attr-defined]
    except Exception:
        return docs
    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: float(x[1]), reverse=True)
    return [RetrievedDoc(chunk=d.chunk, score=float(s)) for d, s in scored]


def rerank_bge_topn(
    query: str, docs: List[RetrievedDoc], top_n: int, model_name: str | None = None
) -> List[RetrievedDoc]:
    """Rerank only the top_n docs by cross-encoder, keep remainder order.
    Returns a list with reranked head followed by untouched tail. Falls back gracefully.
    """
    if not docs or top_n <= 0:
        return docs
    head = docs[: min(top_n, len(docs))]
    tail = docs[len(head) :]
    head_rr = rerank_bge(query, head, model_name=model_name)
    return head_rr + tail


def mmr_rerank(
    query_vec: np.ndarray,
    docs: List[RetrievedDoc],
    doc_vecs: np.ndarray,
    lambda_param: float = 0.5,
    top_n: int | None = None,
) -> List[RetrievedDoc]:
    """Maximal Marginal Relevance reranking using embeddings (API-only approach).

    Args:
        query_vec: Query embedding vector [D]
        docs: List of retrieved documents
        doc_vecs: Document embedding matrix [N, D] in same order as docs
        lambda_param: Balance between relevance (1.0) and diversity (0.0)
        top_n: Number of documents to return (None = all)

    Returns:
        Reranked documents with MMR scores
    """
    if not docs or len(docs) != len(doc_vecs):
        return docs

    n_docs = len(docs)
    if top_n is None:
        top_n = n_docs
    top_n = min(top_n, n_docs)

    # Normalize vectors for cosine similarity
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-12)

    # Compute relevance scores (cosine similarity)
    relevance_scores = doc_norms @ query_norm

    selected_indices: List[int] = []
    remaining_indices: List[int] = list(range(n_docs))

    for _ in range(top_n):
        if not remaining_indices:
            break

        mmr_scores: List[Tuple[int, float]] = []
        for i in remaining_indices:
            # Relevance component
            relevance = relevance_scores[i]

            # Diversity component (max similarity to already selected)
            if selected_indices:
                selected_vecs = doc_norms[selected_indices]
                similarities = selected_vecs @ doc_norms[i]
                max_similarity = np.max(similarities)
            else:
                max_similarity = 0.0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((i, mmr_score))

        # Select document with highest MMR score
        best_idx, best_score = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    # Return reranked documents with MMR scores
    result: List[RetrievedDoc] = []
    for idx in selected_indices:
        doc = docs[idx]
        mmr_score = lambda_param * relevance_scores[idx] - (1 - lambda_param) * (
            np.max(doc_norms[selected_indices[: selected_indices.index(idx)]] @ doc_norms[idx])
            if selected_indices.index(idx) > 0
            else 0.0
        )
        result.append(RetrievedDoc(chunk=doc.chunk, score=float(mmr_score)))

    return result


def llm_rerank(
    query: str, docs: List[RetrievedDoc], model: str = "gpt-4o-mini", top_n: int | None = None
) -> Tuple[List[RetrievedDoc], dict]:
    """LLM-judge reranker using OpenAI for scoring relevance.

    Args:
        query: User query
        docs: List of retrieved documents
        model: OpenAI model to use for scoring
        top_n: Number of documents to return (None = all)

    Returns:
        Tuple of (reranked_docs, cost_info)
    """
    if not docs:
        return docs, {"tokens": 0, "cost": 0.0}

    try:
        from openai import OpenAI
    except ImportError:
        return docs, {"tokens": 0, "cost": 0.0, "error": "openai not available"}

    try:
        client = OpenAI()

        # Build scoring prompt
        scoring_prompt = _build_scoring_prompt(query, docs)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a relevance scoring assistant. Score each document's relevance to the query on a scale of 0-10. Return only a JSON array of scores.",
                },
                {"role": "user", "content": scoring_prompt},
            ],
            temperature=0.1,
            max_tokens=200,
        )

        # Parse scores
        try:
            scores_text_raw = response.choices[0].message.content
            scores_text = (scores_text_raw or "").strip()
            scores = json.loads(scores_text)
            if not isinstance(scores, list) or len(scores) != len(docs):
                raise ValueError("Invalid scores format")
        except (json.JSONDecodeError, ValueError):
            # Fallback: return original order
            return docs, {
                "tokens": response.usage.total_tokens if response.usage else 0,
                "cost": 0.0,
                "error": "score_parsing_failed",
            }

        # Sort by scores
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: float(x[1]), reverse=True)

        # Apply top_n limit
        if top_n is not None:
            scored_docs = scored_docs[:top_n]

        # Return reranked docs with LLM scores
        result = [RetrievedDoc(chunk=doc.chunk, score=float(score)) for doc, score in scored_docs]

        # Calculate cost (rough estimate)
        tokens = response.usage.total_tokens if response.usage else 0
        cost = _estimate_cost(model, tokens)

        return result, {"tokens": tokens, "cost": cost}

    except Exception as e:
        return docs, {"tokens": 0, "cost": 0.0, "error": str(e)}


def _build_scoring_prompt(query: str, docs: List[RetrievedDoc]) -> str:
    """Build a compact prompt for LLM scoring."""
    prompt_parts = [f"Query: {query}\n\nScore each document's relevance (0-10):\n"]

    for i, doc in enumerate(docs):
        # Truncate long documents to keep prompt manageable
        text = doc.chunk.text[:200] + "..." if len(doc.chunk.text) > 200 else doc.chunk.text
        prompt_parts.append(f"{i + 1}. {text}")

    prompt_parts.append("\nReturn only a JSON array of scores: [score1, score2, ...]")
    return "\n".join(prompt_parts)


def _estimate_cost(model: str, tokens: int) -> float:
    """Rough cost estimation for OpenAI models."""
    # Approximate pricing (as of 2024)
    pricing = {
        "gpt-4o-mini": 0.00015 / 1000,  # $0.15 per 1M tokens
        "gpt-4o": 0.005 / 1000,  # $5 per 1M tokens
        "gpt-3.5-turbo": 0.001 / 1000,  # $1 per 1M tokens
    }

    rate = pricing.get(model, 0.001 / 1000)  # Default fallback
    return tokens * rate
