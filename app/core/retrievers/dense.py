from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
from functools import lru_cache
import hashlib
import os
from datetime import datetime, timezone

# FAISS is optional. On Python 3.13 we may not have wheels available.
try:
    import faiss  # type: ignore

    HAVE_FAISS = True
except Exception:  # pragma: no cover - environment dependent
    faiss = None  # type: ignore
    HAVE_FAISS = False
import numpy as np
import json

from app.core.embeddings import get_default_embedder, effective_embedder_id
from app.core.storage import connect
from app.core.types import Chunk, RetrievedDoc
from app.core.utils import load_config


@dataclass
class DenseConfig:
    top_k: int = 6
    offline: bool = False
    st_model: str | None = None
    oa_model: str | None = None
    index_name: str | None = None


def _load_index(path: Path):
    """Load either a FAISS index or a NumPy fallback matrix depending on file type.

    Returns a tuple (kind, obj):
      - kind == 'faiss', obj is a FAISS index
      - kind == 'np', obj is an L2-normalized np.ndarray of shape (N, D)
    """
    if path.suffix == ".faiss":
        if not HAVE_FAISS:
            raise RuntimeError(
                "FAISS index present but faiss module is not available. Rebuild index on this environment or install faiss."
            )
        return "faiss", faiss.read_index(str(path))  # type: ignore
    elif path.suffix == ".npz":
        data = np.load(path)
        X = data["X"]
        return "np", X
    else:
        raise RuntimeError(f"Unsupported index file: {path}")


def retrieve_dense(query: str, cfg: DenseConfig) -> Tuple[List[RetrievedDoc], List[Chunk]]:
    conn = connect()
    # load chunks in order
    cur = conn.execute("SELECT id, doc_id, ord, text, meta_json FROM chunks ORDER BY doc_id, ord")
    rows = cur.fetchall()
    chunks: List[Chunk] = []
    for cid, did, ordv, text, meta_json in rows:
        meta = json.loads(meta_json or "{}")
        chunks.append(Chunk(id=cid, doc_id=did, ord=ordv, text=text, meta=meta))
    # find index path
    cur = conn.execute(
        "SELECT index_name, faiss_path, dim FROM indices ORDER BY created_at DESC LIMIT 1"
    )
    r = cur.fetchone()
    if not r:
        conn.close()
        raise RuntimeError("No FAISS index found. Build index first.")
    index_name, faiss_path, dim = r
    # build index version based on path, dim, and file mtime for invalidation
    mtime = 0
    try:
        st = os.stat(faiss_path)
        mtime = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except Exception:
        mtime = 0
    index_version = f"{faiss_path}:{dim}:{mtime}"
    model_id = effective_embedder_id(cfg.offline, cfg.st_model, cfg.oa_model)
    qsha = hashlib.sha1(query.encode("utf-8")).hexdigest()
    # use cached search results (indices and distances)
    idxs, dists = _dense_search_cached(
        qsha,
        cfg.top_k,
        model_id,
        index_version,
        query,
        faiss_path,
        cfg.offline,
        cfg.st_model,
        cfg.oa_model,
    )
    # Map to RetrievedDoc list
    top: List[RetrievedDoc] = []
    for idx, score in zip(idxs, dists):
        if idx < 0 or idx >= len(chunks):
            continue
        top.append(RetrievedDoc(chunk=chunks[idx], score=float(score)))
    # Recency filter/decay
    try:
        rcfg = load_config().get("retrieval", {})
        days = int(rcfg.get("recency_filter_days", 0) or 0)
        lam = float(rcfg.get("recency_decay_lambda", 0.0) or 0.0)
    except Exception:
        days = 0
        lam = 0.0
    now = datetime.now(tz=timezone.utc)
    if days > 0:
        cutoff = now.timestamp() - days * 86400

        def _ok(d: RetrievedDoc) -> bool:
            mt = str(d.chunk.meta.get("mtime", ""))
            try:
                t = datetime.fromisoformat(mt).timestamp() if mt else 0.0
            except Exception:
                t = 0.0
            return t >= cutoff

        top = [d for d in top if _ok(d)]
    if lam > 0.0 and top:

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

        # apply and re-sort
        top = [RetrievedDoc(chunk=d.chunk, score=_decay(d)) for d in top]
        top.sort(key=lambda r: r.score, reverse=True)
    conn.close()
    return top, chunks


@lru_cache(maxsize=256)
def _dense_search_cached(
    qsha: str,
    top_k: int,
    model_id: str,
    index_version: str,
    query: str,
    index_path: str,
    offline: bool,
    st_model: str | None,
    oa_model: str | None,
):
    """Cache wrapper around the actual dense search to avoid redundant work for identical queries."""
    return _dense_search(
        top_k, model_id, index_version, query, Path(index_path), offline, st_model, oa_model
    )


def _dense_search(
    top_k: int,
    model_id: str,
    index_version: str,
    query: str,
    index_path: Path,
    offline: bool,
    st_model: str | None,
    oa_model: str | None,
):
    embedder = get_default_embedder(offline=offline, st_model=st_model, oa_model=oa_model)
    q = embedder([query])[0]
    # load index or matrix
    kind, obj = _load_index(index_path)
    if kind == "faiss":
        # cosine via inner product on L2-normalized data (index built with normalized X)
        # normalize q
        qn = q / (np.linalg.norm(q) + 1e-12)
        dists, idxs = obj.search(np.asarray([qn], dtype=np.float32), top_k)  # type: ignore
        return idxs[0].tolist(), dists[0].tolist()
    else:
        # NumPy fallback: X is already L2-normalized, compute cosine with normalized q
        X = obj  # np.ndarray [N, D]
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = X @ qn.astype(np.float32)
        # top-k indices
        if top_k >= len(sims):
            top_idx = np.argsort(-sims)
        else:
            top_idx = np.argpartition(-sims, top_k)[:top_k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        return top_idx.tolist(), sims[top_idx].astype(float).tolist()
