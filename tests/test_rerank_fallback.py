from typing import Any


from app.core.retrievers import rerank as rr
from app.core.types import Chunk, RetrievedDoc


def make_docs(n: int = 3):
    docs = []
    for i in range(n):
        ch = Chunk(id=f"c{i}", doc_id="d1", ord=i, text=f"text-{i}", meta={})
        docs.append(RetrievedDoc(chunk=ch, score=float(i)))
    return docs


def test_get_rerank_config_parsing_error_keeps_defaults(monkeypatch):
    def fake_load():
        return {
            "retrieval": {"rerank_model": "X", "rerank_device": "cpu", "rerank_batch_size": "oops"}
        }

    monkeypatch.setattr(rr, "load_config", fake_load, raising=False)
    model, device, bs = rr._get_rerank_config()
    assert model == "X"
    assert device == "cpu"
    assert isinstance(bs, int) and bs == 16  # default preserved on parse error


def test_rerank_bge_fallback_when_no_model(monkeypatch):
    # Force CrossEncoder absence
    monkeypatch.setattr(rr, "CrossEncoder", None, raising=False)
    docs = make_docs(3)
    out = rr.rerank_bge("q", docs)
    assert out == docs


def test_rerank_bge_predict_exception_returns_original(monkeypatch):
    class DummyCE:
        def __init__(self, *a: Any, **k: Any):
            pass

        def predict(self, pairs, batch_size: int = 16):
            raise RuntimeError("boom")

    monkeypatch.setattr(rr, "CrossEncoder", DummyCE, raising=False)
    docs = make_docs(2)
    out = rr.rerank_bge("q", docs, model_name="dummy")
    assert out == docs


def test_rerank_bge_topn_head_tail(monkeypatch):
    class DummyCE:
        def __init__(self, *a: Any, **k: Any):
            pass

        def predict(self, pairs, batch_size: int = 16):
            # Reverse order by scores
            return list(reversed(range(len(pairs))))

    monkeypatch.setattr(rr, "CrossEncoder", DummyCE, raising=False)
    docs = make_docs(4)
    out = rr.rerank_bge_topn("q", docs, top_n=2, model_name="dummy")
    # Length preserved; first two potentially reranked
    assert len(out) == 4
    assert {d.chunk.id for d in out} == {d.chunk.id for d in docs}
