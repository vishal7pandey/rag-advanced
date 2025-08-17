from app.core.flows import multi_hop as mh
from app.core.types import Chunk, RetrievedDoc


def test_multi_hop_runs_and_populates_extras(monkeypatch):
    # Force two hops
    monkeypatch.setattr(mh, "_gen_subquestions", lambda q, n, m, o: ["h1", "h2"])

    def fake_retrieve(q, cfg):
        docs = [
            RetrievedDoc(
                chunk=Chunk(id=f"{q[:1]}{i}", doc_id="d", ord=i, text=f"{q} ctx {i}", meta={}),
                score=1.0 / (i + 1),
            )
            for i in range(3)
        ]
        return docs, []

    monkeypatch.setattr(mh, "retrieve_dense", fake_retrieve)
    monkeypatch.setattr(
        mh,
        "generate_answer",
        lambda prompt, model, offline, docs: {
            "answer_md": "ans",
            "citations": [{"marker": 1}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        },
    )

    flow = mh.MultiHopFlow(
        offline=True, gen_model="stub", emb_st=None, emb_oa=None, params=mh.MultiHopParams()
    )
    out = flow.run("Q?", session_state={})
    assert out.extras and isinstance(out.extras.get("hops"), list)
    assert len(out.extras["hops"]) >= 2
    assert out.citations
