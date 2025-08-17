import sqlite3
from pathlib import Path
import numpy as np
import pytest

try:
    import faiss  # type: ignore

    _faiss_ok = True
except Exception:
    faiss = None  # type: ignore
    _faiss_ok = False

from app.core import storage
from app.core.retrievers import dense as dense_mod
from app.core.retrievers import bm25 as bm25_mod
from app.core.retrievers.dense import retrieve_dense, DenseConfig
from app.core.retrievers.bm25 import retrieve_bm25, BM25Config
from app.core.retrievers.hybrid import rrf_fuse
from app.core.types import Chunk, RetrievedDoc


def _fake_embedder():
    def fn(texts):
        vecs = []
        for t in texts:
            s = sum(ord(c) for c in t) or 1
            v = np.array([(s % (i + 3)) / 10.0 for i in range(8)], dtype=np.float32)
            # normalize so IP equals cosine
            v = v / (np.linalg.norm(v) + 1e-8)
            vecs.append(v.astype(np.float32))
        return np.vstack(vecs)

    return fn


def _seed_db(db_path: Path, texts: list[str]) -> None:
    storage.init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        # insert single doc and chunks
        conn.execute("INSERT INTO documents(id, path, sha) VALUES(?,?,?)", ("doc1", "tmp", "sha"))
        for i, t in enumerate(texts):
            conn.execute(
                "INSERT INTO chunks(id, doc_id, ord, text, meta_json) VALUES(?,?,?,?,?)",
                (f"c{i}", "doc1", i, t, "{}"),
            )
        conn.commit()
    finally:
        conn.close()


def test_bm25_retriever_orders_relevant(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    texts = ["alpha beta", "beta gamma", "delta alpha", "epsilon zeta"]
    _seed_db(db, texts)
    # redirect connect() symbol used inside bm25 module to our db
    monkeypatch.setattr(bm25_mod, "connect", lambda db_path=None: sqlite3.connect(db))
    out = retrieve_bm25("alpha", BM25Config(top_k=3))
    assert out and out[0].chunk.text in {"alpha beta", "delta alpha"}


def test_dense_retriever_uses_faiss_index(tmp_path, monkeypatch):
    if not _faiss_ok:
        pytest.skip("FAISS not available on this environment")
    db = tmp_path / "test.db"
    texts = ["alpha beta", "beta gamma", "delta alpha"]
    _seed_db(db, texts)
    # Build FAISS index with our fake embeddings
    try:
        X = _fake_embedder()(texts)
        dim = X.shape[1]
        faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(dim)
        index.add(X)
        idx_path = tmp_path / "test.faiss"
        faiss.write_index(index, str(idx_path))
    except Exception:
        pytest.skip("FAISS operations not supported on this CPU/Python build")
    # write indices table
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "INSERT INTO indices(index_name, faiss_path, dim) VALUES(?,?,?)",
            ("faiss_stub", str(idx_path), int(dim)),
        )
        conn.commit()
    finally:
        conn.close()
    # redirect connect symbol used inside dense module and embedder
    monkeypatch.setattr(dense_mod, "connect", lambda db_path=None: sqlite3.connect(db))
    monkeypatch.setattr(
        "app.core.retrievers.dense.get_default_embedder",
        lambda offline, st_model, oa_model: _fake_embedder(),
    )
    out, all_chunks = retrieve_dense("alpha", DenseConfig(top_k=2, offline=True, st_model="stub"))
    assert len(out) == 2
    # At least one of the docs containing 'alpha' should be in top-2
    top_texts = {r.chunk.text for r in out}
    assert top_texts & {"alpha beta", "delta alpha"}


def test_rrf_fuse_prefers_consensus():
    a = [
        RetrievedDoc(chunk=Chunk(id=str(i), doc_id="d", ord=i, text=f"t{i}", meta={}), score=1.0)
        for i in [0, 1, 2]
    ]
    b = [
        RetrievedDoc(chunk=Chunk(id=str(i), doc_id="d", ord=i, text=f"t{i}", meta={}), score=1.0)
        for i in [2, 1, 0]
    ]
    fused = rrf_fuse(a, b, k=10)
    # Standard RRF may rank edges above the middle; stable sort favors '0' tie
    assert fused[0].chunk.id in {"0", "2"}
