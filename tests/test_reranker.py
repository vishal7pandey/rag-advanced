from app.core.retrievers import rerank as rr
from app.core.types import Chunk, RetrievedDoc
from app.core.metrics.raglite import delta_precision_lite


def _mk_docs(n=5):
    return [
        RetrievedDoc(
            chunk=Chunk(id=f"c{i}", doc_id="d", ord=i, text=f"text {i}", meta={}),
            score=1.0 / (i + 1),
        )
        for i in range(n)
    ]


def test_rerank_topn_uses_custom_sort(monkeypatch):
    docs = _mk_docs(5)

    def fake_rerank_bge(query, docs_in, model_name="BAAI/bge-reranker-base"):
        # reverse order to simulate change
        return list(reversed(docs_in))

    monkeypatch.setattr(rr, "rerank_bge", fake_rerank_bge)
    out = rr.rerank_bge_topn("q", docs, top_n=3)
    # head reversed, tail untouched
    assert [d.chunk.id for d in out[:3]] == ["c2", "c1", "c0"]
    assert [d.chunk.id for d in out[3:]] == ["c3", "c4"]


def test_delta_precision_reports_change():
    before = _mk_docs(5)
    after = list(reversed(before))
    val = delta_precision_lite(before, after, n=5)
    assert -1.0 <= val <= 1.0
