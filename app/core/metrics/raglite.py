from __future__ import annotations
from typing import List
import numpy as np

from app.core.types import RetrievedDoc


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def lite_metrics(answer_text: str, docs: List[RetrievedDoc]) -> dict:
    # Placeholder heuristic metrics in [0,1]
    if not answer_text or not docs:
        return {
            "answer_relevancy_lite": 0.0,
            "context_precision_lite": 0.0,
            "groundedness_lite": 0.0,
        }
    # crude: longer answer and more docs => higher coverage (placeholder)
    ar = min(1.0, len(answer_text) / 1000)
    cp = min(1.0, sum(len(d.chunk.text) for d in docs) / 5000)
    gr = min(1.0, len(docs) / 10)
    return {"answer_relevancy_lite": ar, "context_precision_lite": cp, "groundedness_lite": gr}


def delta_precision_lite(
    before: List[RetrievedDoc], after: List[RetrievedDoc], n: int = 10
) -> float:
    """Rank-based lightweight delta precision proxy.
    Computes average normalized rank improvement for items appearing in both top-n lists.
    Positive value indicates improved ranking after rerank.
    Returns value in approximately [-1, 1].
    """
    if not before or not after:
        return 0.0
    n_b = min(n, len(before))
    n_a = min(n, len(after))

    # build ids
    def _cid(d: RetrievedDoc) -> str:
        return getattr(d.chunk, "id", None) or f"{d.chunk.meta.get('path', '')}::{d.chunk.ord}"

    idx_b = {_cid(d): i for i, d in enumerate(before[:n_b])}
    idx_a = {_cid(d): i for i, d in enumerate(after[:n_a])}
    shared = [cid for cid in idx_b.keys() if cid in idx_a]
    if not shared:
        return 0.0
    improvements = []
    denom = float(max(n_b, n_a))
    for cid in shared:
        rb = idx_b[cid]
        ra = idx_a[cid]
        # improvement: positive if moved up
        improvements.append((rb - ra) / denom)
    return float(np.clip(np.mean(improvements), -1.0, 1.0))
