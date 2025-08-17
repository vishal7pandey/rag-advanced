from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set, cast
import re
import os

from app.core.retrievers.rerank import rerank_bge_topn
from app.core.types import RetrievedDoc, CitationUIMap, CitationEnvMap


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_claims(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    # Clean and keep non-trivial claims
    claims = [p.strip() for p in parts if p.strip()]
    return claims


@dataclass
class ClaimEvidence:
    claim: str
    supports: List[Tuple[int, RetrievedDoc]]  # (rank, doc)


def per_claim_citations(
    answer_md: str, retrieved: List[RetrievedDoc], top_n: int = 1
) -> List[ClaimEvidence]:
    """Map simple claims to best supporting retrieved docs.

    Uses BGE cross-encoder reranker when available; otherwise falls back to
    original order. Returns top_n supports per claim.
    """
    claims = _split_claims(answer_md)
    if not claims or not retrieved:
        return []
    results: List[ClaimEvidence] = []
    for c in claims:
        try:
            ranked = rerank_bge_topn(c, retrieved, top_n=top_n)
            supports = [(i + 1, ranked[i]) for i in range(min(top_n, len(ranked)))]
        except Exception:
            # Fallback: take first N retrieved if reranker unavailable
            supports = [(i + 1, retrieved[i]) for i in range(min(top_n, len(retrieved)))]
        results.append(ClaimEvidence(claim=c, supports=supports))
    return results


# Deterministic citation mapping for UI and Envelope
@dataclass
class CitationUI:
    marker: int
    title: str
    doc_short_id: str
    ord: Optional[int] = None


@dataclass
class CitationEnv:
    marker: int
    doc_id: str
    chunk_id: str
    ord: Optional[int] = None
    path: Optional[str] = None


def build_citation_map(
    retrieved: List[RetrievedDoc],
) -> Tuple[List[CitationUIMap], List[CitationEnvMap], Dict[str, int]]:
    """
    Given merged, de-duplicated retrieved docs, assign stable 1-based markers and
    build two parallel representations plus a key->marker index map.

    Returns: (citations_ui, citations_env, index_map)
    - citations_ui: [{marker, title, doc_short_id, ord}]
    - citations_env: [{marker, doc_id, chunk_id, ord, path}]
    - index_map: {chunk_id or "doc_id:ord" -> marker}
    """
    seen: Set[object] = set()
    ui: List[CitationUIMap] = []
    env: List[CitationEnvMap] = []
    index: Dict[str, int] = {}
    marker = 1
    for rd in retrieved or []:
        try:
            cid = getattr(rd.chunk, "id", None)
            key = cid or (rd.chunk.doc_id, rd.chunk.ord)
            if key in seen:
                continue
            seen.add(key)
            path = cast(str, (rd.chunk.meta or {}).get("path", ""))
            title = os.path.basename(path) or rd.chunk.doc_id[:8]
            cui = CitationUI(
                marker=marker, title=title, doc_short_id=str(rd.chunk.doc_id)[:8], ord=rd.chunk.ord
            )
            cen = CitationEnv(
                marker=marker,
                doc_id=str(rd.chunk.doc_id),
                chunk_id=str(rd.chunk.id),
                ord=rd.chunk.ord,
                path=path,
            )
            ui.append(cast(CitationUIMap, asdict(cui)))
            env.append(cast(CitationEnvMap, asdict(cen)))
            # Index by multiple keys
            if cid:
                index[str(cid)] = marker
            index[f"{rd.chunk.doc_id}:{rd.chunk.ord}"] = marker
            marker += 1
        except Exception:
            continue
    return ui, env, index
