from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from datetime import datetime, timezone

import numpy as np  # type: ignore

from app.core.types import RetrievedDoc
from app.core.utils import load_config


@dataclass
class HybridConfig:
    k_bm25: int = 20
    k_dense: int = 20
    rrf_k: int = 10


def _apply_recency_adjustments(items: List[RetrievedDoc]) -> List[RetrievedDoc]:
    """Apply optional recency filter and exponential decay based on config."""
    try:
        rcfg = load_config().get("retrieval", {})
        days = int(rcfg.get("recency_filter_days", 0) or 0)
        lam = float(rcfg.get("recency_decay_lambda", 0.0) or 0.0)
    except Exception:
        days = 0
        lam = 0.0
    now = datetime.now(tz=timezone.utc)
    fused: List[RetrievedDoc] = list(items)
    if days > 0:
        cutoff = now.timestamp() - days * 86400

        def _ok(d: RetrievedDoc) -> bool:
            mt = str(d.chunk.meta.get("mtime", ""))
            try:
                t = datetime.fromisoformat(mt).timestamp() if mt else 0.0
            except Exception:
                t = 0.0
            return t >= cutoff

        fused = [d for d in fused if _ok(d)]
    if lam > 0.0 and fused:

        def _decay(d: RetrievedDoc) -> float:
            mt = str(d.chunk.meta.get("mtime", ""))
            try:
                t = datetime.fromisoformat(mt)
                age_days = max(
                    0.0,
                    (now - (t if t.tzinfo else t.replace(tzinfo=timezone.utc))).total_seconds()
                    / 86400.0,
                )
            except Exception:
                age_days = 0.0
            return float(d.score) * float(np.exp(-lam * age_days))

        fused = [RetrievedDoc(chunk=d.chunk, score=_decay(d)) for d in fused]
        fused.sort(key=lambda r: r.score, reverse=True)
    return fused


def rrf_fuse(
    bm25: List[RetrievedDoc],
    dense: List[RetrievedDoc],
    k: int = 10,
    w_bm25: float = 1.0,
    w_dense: float = 1.0,
) -> List[RetrievedDoc]:
    """Reciprocal Rank Fusion with optional weights for bm25 and dense lists.

    Backwards compatible: callers not passing weights will behave as before.
    """
    scores: Dict[str, float] = {}
    for i, r in enumerate(bm25):
        scores[r.chunk.id] = scores.get(r.chunk.id, 0.0) + float(w_bm25) * (1.0 / (k + i + 1))
    for i, r in enumerate(dense):
        scores[r.chunk.id] = scores.get(r.chunk.id, 0.0) + float(w_dense) * (1.0 / (k + i + 1))
    # build from union preserving any order by score
    id_to_doc = {r.chunk.id: r for r in dense + bm25}
    fused_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = [RetrievedDoc(chunk=id_to_doc[i].chunk, score=scores[i]) for i in fused_ids]
    return _apply_recency_adjustments(fused)


def rrf_fuse_multi(
    bm25_lists: List[List[RetrievedDoc]],
    dense_lists: List[List[RetrievedDoc]],
    k: int = 10,
    w_bm25: float = 1.0,
    w_dense: float = 1.0,
) -> List[RetrievedDoc]:
    """RRf over multiple bm25 and dense result lists (e.g., multi-query expansion).

    Each list contributes scores independently so later queries are not penalized
    by list concatenation. Applies the same recency adjustments as rrf_fuse.
    """
    scores: Dict[str, float] = {}
    id_to_doc: Dict[str, RetrievedDoc] = {}

    def _accumulate(docs: List[RetrievedDoc], weight: float):
        for i, r in enumerate(docs):
            cid = r.chunk.id
            id_to_doc[cid] = r
            scores[cid] = scores.get(cid, 0.0) + float(weight) * (1.0 / (k + i + 1))

    for lst in bm25_lists:
        _accumulate(lst, w_bm25)
    for lst in dense_lists:
        _accumulate(lst, w_dense)

    fused_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = [RetrievedDoc(chunk=id_to_doc[i].chunk, score=scores[i]) for i in fused_ids]
    return _apply_recency_adjustments(fused)
