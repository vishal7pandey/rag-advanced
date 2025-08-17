from app.core.flows.standard import StandardFlow, StandardParams
from app.core.types import Chunk, RetrievedDoc


def _docs():
    A = RetrievedDoc(chunk=Chunk(id="A", doc_id="d", ord=0, text="tA", meta={}), score=0.3)
    B = RetrievedDoc(chunk=Chunk(id="B", doc_id="d", ord=1, text="tB", meta={}), score=0.2)
    C = RetrievedDoc(chunk=Chunk(id="C", doc_id="d", ord=2, text="tC", meta={}), score=0.1)
    return [A, B, C]


def test_standard_flow_no_rerank(monkeypatch):
    pre = _docs()
    # bypass retrieval/generation
    monkeypatch.setattr(
        "app.core.flows.standard.retrieve_dense",
        lambda q, cfg: (list(pre), []),
    )
    monkeypatch.setattr(
        "app.core.flows.standard.generate_answer",
        lambda prompt, model, offline, docs: {
            "answer_md": "ans",
            "citations": [{"marker": 1}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    flow = StandardFlow(
        offline=True,
        gen_model="stub",
        emb_st=None,
        emb_oa=None,
        params=StandardParams(rerank=False),
    )
    out = flow.run("Q?", session_state={"memory_text": "mem"})
    assert out.metrics.get("retrieved")
    assert out.metrics.get("reranked")
    assert "delta_precision_lite" not in out.metrics
    assert out.extras["prompt"]["text"]


def test_standard_flow_with_rerank(monkeypatch):
    pre = _docs()
    # retrieval returns A,B,C
    monkeypatch.setattr(
        "app.core.flows.standard.retrieve_dense",
        lambda q, cfg: (list(pre), []),
    )
    # reranker swaps A and B (improvement for B)
    monkeypatch.setattr(
        "app.core.flows.standard.rerank_bge_topn",
        lambda q, docs, top_n=6: [docs[1], docs[0], docs[2]],
    )
    monkeypatch.setattr(
        "app.core.flows.standard.generate_answer",
        lambda prompt, model, offline, docs: {
            "answer_md": "ans",
            "citations": [{"marker": 1}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    flow = StandardFlow(
        offline=True,
        gen_model="stub",
        emb_st=None,
        emb_oa=None,
        params=StandardParams(rerank=True, rerank_top_n=2, rerank_strategy="cross_encoder"),
    )
    out = flow.run("Q?", session_state={})
    # Rerank branch should compute delta_precision_lite; value may be zero depending on tie behavior
    assert "delta_precision_lite" in out.metrics
    assert isinstance(out.metrics.get("rerank_deltas"), list)
