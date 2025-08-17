from datetime import datetime, timedelta, timezone

from app.core.retrievers.hybrid import rrf_fuse
from app.core.types import Chunk, RetrievedDoc


def mk_doc(id_: str, score: float, days_ago: int) -> RetrievedDoc:
    mt = (datetime.now(tz=timezone.utc) - timedelta(days=days_ago)).isoformat()
    ch = Chunk(id=id_, doc_id="d1", ord=0, text=f"{id_}", meta={"mtime": mt, "path": f"{id_}.txt"})
    return RetrievedDoc(chunk=ch, score=score)


def test_rrf_recency_filter(monkeypatch):
    # Configure recency filter to 1 day; one old doc should be filtered out
    from app.core import retrievers as r

    def fake_load():
        return {"retrieval": {"recency_filter_days": 1, "recency_decay_lambda": 0.0}}

    monkeypatch.setattr(r.hybrid, "load_config", fake_load, raising=False)

    bm25 = [mk_doc("a", 1.0, days_ago=0), mk_doc("b", 1.0, days_ago=10)]
    dense = [mk_doc("a", 1.0, days_ago=0), mk_doc("c", 1.0, days_ago=0)]

    fused = rrf_fuse(bm25, dense, k=10)
    ids = [d.chunk.id for d in fused]
    assert "b" not in ids  # filtered by recency
    assert set(ids) == {"a", "c"}


def test_rrf_recency_decay_sort(monkeypatch):
    # No filter; apply decay so newer doc scores higher after decay
    from app.core import retrievers as r

    def fake_load():
        return {"retrieval": {"recency_filter_days": 0, "recency_decay_lambda": 0.5}}

    monkeypatch.setattr(r.hybrid, "load_config", fake_load, raising=False)

    # Older doc has higher initial fused score but should lose after decay
    bm25 = [mk_doc("x", 1.0, days_ago=10), mk_doc("y", 0.1, days_ago=0)]
    dense = [mk_doc("x", 1.0, days_ago=10)]

    fused = rrf_fuse(bm25, dense, k=10)
    assert fused[0].chunk.id == "y"
