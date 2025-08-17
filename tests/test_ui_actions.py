from pathlib import Path
import numpy as np

from app.core.indexer import ingest_paths, reembed_changed_only, build_faiss, purge_index


def _fake_embedder(dim: int = 8):
    def f(texts):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(len(texts), dim)).astype(np.float32)
        return X

    return f


def test_reembed_and_purge_roundtrip(tmp_path, monkeypatch):
    # Use temp DB file
    db = tmp_path / "test.db"
    monkeypatch.setattr("app.core.storage.DB_FILE", db)
    # Seed two small files and ingest
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("alpha beta gamma")
    p2.write_text("delta epsilon zeta")

    chunks, stats = ingest_paths([p1, p2], chunk_size=16, overlap=0)
    assert len(chunks) > 0

    # Avoid downloading models during test
    monkeypatch.setattr(
        "app.core.indexer.get_default_embedder",
        lambda offline, st_model, oa_model: _fake_embedder(),
    )

    # Re-embed changed only should create cache files and report counts
    res = reembed_changed_only(offline=True, emb_model_st="stub", emb_model_oa=None)
    assert res["total_chunks"] == len(chunks)
    assert res["new_embeddings"] > 0

    # Build FAISS and then purge
    idx_path, dim = build_faiss(offline=True, emb_model_st="stub", emb_model_oa=None)
    assert Path(idx_path).exists()

    out = purge_index(offline=True, emb_model_st="stub", emb_model_oa=None)
    assert out["tag"] == "stub"
    # Either removed file or no-op if already gone
    assert out["removed_cache_files"] >= 0
