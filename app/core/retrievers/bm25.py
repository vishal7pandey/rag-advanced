from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from functools import lru_cache
import hashlib

from rank_bm25 import BM25Okapi  # type: ignore
import json

from app.core.storage import connect
from app.core.types import Chunk, RetrievedDoc


@dataclass
class BM25Config:
    top_k: int = 10


def retrieve_bm25(query: str, cfg: BM25Config) -> List[RetrievedDoc]:
    conn = connect()
    # Precompute a simple chunks version for invalidation
    ver_row = conn.execute("SELECT COUNT(*), MAX(id) FROM chunks").fetchone()
    # id is TEXT (UUID). Avoid casting to int; use a string-based version key instead.
    total = int(ver_row[0]) if ver_row and ver_row[0] is not None else 0
    max_id = str(ver_row[1]) if ver_row and ver_row[1] is not None else ""
    chunks_version = f"{total}:{max_id}"
    # Load chunks for mapping results back
    cur = conn.execute("SELECT id, doc_id, ord, text, meta_json FROM chunks ORDER BY doc_id, ord")
    rows = cur.fetchall()
    chunks: List[Chunk] = []
    for r in rows:
        meta = json.loads((r[4] or "{}"))
        chunks.append(Chunk(id=r[0], doc_id=r[1], ord=r[2], text=r[3], meta=meta))
    conn.close()
    if not chunks:
        return []
    qsha = hashlib.sha1(query.encode("utf-8")).hexdigest()
    idxs, scores = _bm25_search_cached(qsha, cfg.top_k, chunks_version, query)
    return [
        RetrievedDoc(chunk=chunks[i], score=float(scores[pos]))
        for pos, i in enumerate(idxs)
        if 0 <= i < len(chunks)
    ]


@lru_cache(maxsize=256)
def _bm25_search_cached(
    query_sha: str, top_k: int, chunks_version: str, query: str
) -> Tuple[List[int], List[float]]:
    """Run BM25 over current chunks snapshot and cache indices/scores.
    Uses COUNT and MAX(id) (TEXT-safe) as a coarse invalidation key.
    """
    conn = connect()
    cur = conn.execute("SELECT text FROM chunks ORDER BY doc_id, ord")
    texts = [r[0] for r in cur.fetchall()]
    conn.close()
    if not texts:
        return [], []
    corpus_tokens = [t.split() for t in texts]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(query.split())
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return idxs, [float(scores[i]) for i in idxs]
