import numpy as np

from app.core.indexer import ingest_paths, reembed_changed_only, build_faiss, purge_index


def _fake_embedder():
    def fn(texts):
        # map text to 8-dim vector based on ascii sums for determinism
        vecs = []
        for t in texts:
            s = sum(ord(c) for c in t) or 1
            v = np.array([(s % (i + 3)) / 10.0 for i in range(8)], dtype=np.float32)
            vecs.append(v)
        return np.vstack(vecs)

    return fn


def test_reembed_changed_and_purge(tmp_path, monkeypatch):
    # Create two docs
    p1 = tmp_path / "a.txt"
    p2 = tmp_path / "b.txt"
    p1.write_text("alpha beta gamma")
    p2.write_text("delta epsilon zeta")

    # Ingest
    chunks, stats = ingest_paths([p1, p2], chunk_size=32, overlap=0)
    assert stats["docs_added"] == 2
    assert len(chunks) >= 2

    # Monkeypatch embedder
    monkeypatch.setattr(
        "app.core.indexer.get_default_embedder",
        lambda offline, st_model, oa_model: _fake_embedder(),
    )

    # First re-embed should add embeddings
    res1 = reembed_changed_only(offline=True, emb_model_st="stub", emb_model_oa=None)
    assert res1["new_embeddings"] >= 1

    # Build FAISS
    idx_path, dim = build_faiss(offline=True, emb_model_st="stub", emb_model_oa=None)
    assert idx_path.exists()
    assert dim > 0

    # Second re-embed should mostly skip
    res2 = reembed_changed_only(offline=True, emb_model_st="stub", emb_model_oa=None)
    assert res2["skipped"] >= res1["total_chunks"] - res1["new_embeddings"]

    # Purge
    out = purge_index(offline=True, emb_model_st="stub", emb_model_oa=None)
    assert out["tag"] == "stub"
    assert out["removed_index"] is True
